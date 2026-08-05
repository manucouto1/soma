//! Python environment manager: creates and maintains isolated venvs/conda envs
//! per pipeline, with incremental dependency updates.

use crate::error::{Result, WorkerError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment type preference.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnvType {
    /// Standard-library `python -m venv` (the default — no extra tooling).
    #[default]
    Venv,
    /// A conda environment, for pipelines whose dependencies need conda's
    /// binary packages.
    Conda,
}

/// Lockfile: tracks what's installed in an environment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvLockfile {
    /// Installed packages, name → version.
    pub packages: HashMap<String, String>,
    /// SHA-256 of the normalized requirements the environment was built
    /// from — a matching hash means the env can be reused as is.
    pub requirements_hash: String,
    /// How the environment was created (venv or conda).
    pub env_type: EnvType,
    /// The interpreter version the environment was created with.
    pub python_version: String,
}

/// Manages isolated Python environments for pipeline execution.
pub struct EnvManager {
    base_dir: PathBuf,
    env_type: EnvType,
}

impl EnvManager {
    /// A manager that keeps its environments under `base_dir`, one per
    /// pipeline. The directory is created eagerly; if that fails,
    /// [`EnvManager::ensure_env`] reports the real error at first use.
    pub fn new(base_dir: impl Into<PathBuf>, env_type: EnvType) -> Self {
        let base = base_dir.into();
        std::fs::create_dir_all(&base).ok();
        Self {
            base_dir: base,
            env_type,
        }
    }

    /// Get or create an environment for a pipeline.
    /// Returns the path to the Python binary.
    pub fn ensure_env(&self, pipeline_id: &str, requirements: &str) -> Result<PathBuf> {
        let req_hash = Self::hash_requirements(requirements);
        let env_dir = self.base_dir.join(format!("env-{pipeline_id}"));
        let lockfile_path = env_dir.join("lockfile.json");

        // Check if env exists and is up to date
        if env_dir.exists()
            && let Ok(lockfile) = self.read_lockfile(&lockfile_path)
        {
            if lockfile.requirements_hash == req_hash {
                // Env is up to date, just return python path
                tracing::info!("Reusing env for pipeline {pipeline_id} (hash match)");
                return self.python_path(&env_dir);
            }

            // Requirements changed — do incremental update
            tracing::info!("Updating env for pipeline {pipeline_id} (requirements changed)");
            self.incremental_update(&env_dir, requirements, &lockfile)?;
            self.write_lockfile(&lockfile_path, requirements, &req_hash)?;
            return self.python_path(&env_dir);
        }

        // Create new environment
        tracing::info!("Creating new env for pipeline {pipeline_id}");
        self.create_env(&env_dir)?;
        self.install_requirements(&env_dir, requirements)?;
        self.write_lockfile(&lockfile_path, requirements, &req_hash)?;

        self.python_path(&env_dir)
    }

    /// Remove unused environments older than max_age.
    pub fn cleanup(&self, max_age: std::time::Duration) -> usize {
        let mut removed = 0;
        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata()
                    && let Ok(modified) = meta.modified()
                    && modified.elapsed().unwrap_or_default() > max_age
                {
                    let _ = std::fs::remove_dir_all(entry.path());
                    removed += 1;
                }
            }
        }
        removed
    }

    // ── Internal ──

    /// Put `package_dir` on the venv's import path with a `.pth` file.
    ///
    /// The alternative — `pip install` of the source tree — would rebuild
    /// the compiled extension for every pipeline that asks for a different
    /// requirement set. A single path entry costs nothing and points at the
    /// build that is already there.
    ///
    /// Nothing else is placed on the path, so the venv's own packages
    /// (torch, numpy, whatever the requirements asked for) keep winning.
    fn link_local_package(env_dir: &Path, package_dir: &str) -> Result<()> {
        let lib = env_dir.join("lib");
        let site = std::fs::read_dir(&lib)
            .map_err(|e| WorkerError::Env(format!("reading {}: {e}", lib.display())))?
            .filter_map(|e| e.ok())
            .map(|e| e.path().join("site-packages"))
            .find(|p| p.is_dir())
            .ok_or_else(|| {
                WorkerError::Env(format!(
                    "no site-packages under {} to link the local soma package into",
                    lib.display()
                ))
            })?;
        std::fs::write(site.join("_soma_local.pth"), format!("{package_dir}\n"))
            .map_err(|e| WorkerError::Env(format!("writing _soma_local.pth: {e}")))?;
        tracing::info!(path = %package_dir, "worker venv uses the local soma package");
        Ok(())
    }

    fn create_env(&self, env_dir: &Path) -> Result<()> {
        match self.env_type {
            EnvType::Venv => {
                let output = Command::new("python3")
                    .args(["-m", "venv", &env_dir.to_string_lossy()])
                    .output()
                    .map_err(|e| WorkerError::Env(format!("Failed to create venv: {e}")))?;
                if !output.status.success() {
                    return Err(WorkerError::Env(format!(
                        "venv creation failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )));
                }
            }
            EnvType::Conda => {
                let output = Command::new("conda")
                    .args([
                        "create",
                        "-p",
                        &env_dir.to_string_lossy(),
                        "python=3.11",
                        "-y",
                        "-q",
                    ])
                    .output()
                    .map_err(|e| WorkerError::Env(format!("Failed to create conda env: {e}")))?;
                if !output.status.success() {
                    return Err(WorkerError::Env(format!(
                        "conda create failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )));
                }
            }
        }
        Ok(())
    }

    fn install_requirements(&self, env_dir: &Path, requirements: &str) -> Result<()> {
        let pip = self.pip_path(env_dir);

        // Write requirements to temp file
        let req_file = env_dir.join("requirements.txt");
        std::fs::write(&req_file, requirements)
            .map_err(|e| WorkerError::Env(format!("Failed to write requirements.txt: {e}")))?;

        // The bootstrap in `python_process.rs` opens with
        // `import json, sys, base64, cloudpickle, io, pickle`, so a venv
        // without cloudpickle cannot load a single filter — the child dies
        // on its first line and the worker reports "python process closed
        // stdout (crashed?)", which names neither the module nor the venv.
        //
        // This used to install "soma", which is a DIFFERENT project on
        // PyPI: this one publishes as `somatize`. The result was discarded
        // with `let _`, so installing the wrong package, or failing to
        // install anything, was indistinguishable from success.
        // Pinned to THIS build's version where PyPI has it. An unpinned
        // install put somatize 0.3.1 in the venv of a 0.4.0 worker, so the
        // subprocess ran a months-old `_composite.py` against filters
        // pickled by the current build — and the failure surfaced as
        // `'NoneType' object has no attribute 'size'` inside the user's
        // fit, with nothing anywhere mentioning a version.
        //
        // It falls back rather than refusing, because a build ahead of the
        // last release is the normal state of a repository and a mismatch
        // is harmless for a plain filter. It is not harmless for a
        // differentiable one, so the fallback warns loudly.
        // `$SOMA_LOCAL_PACKAGE` short-circuits all of that: it names the
        // directory holding the `soma` package this build belongs to, and
        // is how a working tree runs its OWN Python layer on a worker
        // instead of whatever the last release put on PyPI. The Python
        // `Worker` sets it automatically. It is added by a `.pth` rather
        // than installed, because installing it would mean compiling the
        // extension module once per venv.
        let version = env!("CARGO_PKG_VERSION");
        let local_package = std::env::var("SOMA_LOCAL_PACKAGE").ok().filter(|p| {
            let ok = std::path::Path::new(p).join("soma").is_dir();
            if !ok {
                tracing::warn!(
                    path = %p,
                    "SOMA_LOCAL_PACKAGE does not contain a `soma` directory; \
                     falling back to installing somatize from PyPI"
                );
            }
            ok
        });
        let mut bootstrap = if let Some(dir) = &local_package {
            let out = Command::new(&pip)
                .args(["install", "-q", "cloudpickle"])
                .output()
                .map_err(|e| WorkerError::Env(format!("pip install cloudpickle failed: {e}")))?;
            if out.status.success() {
                Self::link_local_package(env_dir, dir)?;
            }
            out
        } else {
            Command::new(&pip)
                .args([
                    "install",
                    "-q",
                    "cloudpickle",
                    &format!("somatize=={version}"),
                ])
                .output()
                .map_err(|e| {
                    WorkerError::Env(format!("pip install (bootstrap deps) failed: {e}"))
                })?
        };
        if !bootstrap.status.success() && local_package.is_none() {
            tracing::warn!(
                version,
                "PyPI has no somatize {version}; installing the latest instead. A \
                 filter pickled by this build and unpickled against a different \
                 somatize can fail deep inside its own fit, naming no version. \
                 Set SOMA_LOCAL_PACKAGE to this build's Python package \
                 directory to run the real thing instead",
            );
            bootstrap = Command::new(&pip)
                .args(["install", "-q", "cloudpickle", "somatize"])
                .output()
                .map_err(|e| {
                    WorkerError::Env(format!("pip install (bootstrap deps) failed: {e}"))
                })?;
        }
        if !bootstrap.status.success() {
            return Err(WorkerError::Env(format!(
                "installing the bootstrap dependencies (cloudpickle, somatize) failed in {}:\n{}",
                env_dir.display(),
                String::from_utf8_lossy(&bootstrap.stderr)
            )));
        }

        let output = Command::new(&pip)
            .args(["install", "-r", &req_file.to_string_lossy(), "-q"])
            .output()
            .map_err(|e| WorkerError::Env(format!("pip install failed: {e}")))?;

        if !output.status.success() {
            return Err(WorkerError::Env(format!(
                "pip install failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(())
    }

    fn incremental_update(
        &self,
        env_dir: &Path,
        new_requirements: &str,
        old_lockfile: &EnvLockfile,
    ) -> Result<()> {
        let new_packages = Self::parse_requirements(new_requirements);
        let pip = self.pip_path(env_dir);

        // Find packages to install/upgrade
        let mut to_install = Vec::new();
        for (name, version) in &new_packages {
            match old_lockfile.packages.get(name) {
                None => {
                    // New package
                    tracing::info!("  + {name}=={version}");
                    to_install.push(format!("{name}=={version}"));
                }
                Some(old_ver) if old_ver != version => {
                    // Version changed
                    tracing::info!("  ↑ {name}: {old_ver} → {version}");
                    to_install.push(format!("{name}=={version}"));
                }
                _ => {} // Same version, skip
            }
        }

        // Find packages to remove
        for name in old_lockfile.packages.keys() {
            if !new_packages.contains_key(name) {
                tracing::info!("  - {name}");
                let _ = Command::new(&pip)
                    .args(["uninstall", name, "-y", "-q"])
                    .output();
            }
        }

        // Install new/updated packages
        if !to_install.is_empty() {
            let output = Command::new(&pip)
                .args(["install"])
                .args(&to_install)
                .arg("-q")
                .output()
                .map_err(|e| WorkerError::Env(format!("pip install failed: {e}")))?;

            if !output.status.success() {
                return Err(WorkerError::Env(format!(
                    "pip install failed:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
        }

        Ok(())
    }

    fn python_path(&self, env_dir: &Path) -> Result<PathBuf> {
        let path = env_dir.join("bin").join("python");
        if path.exists() {
            Ok(path)
        } else {
            Err(WorkerError::Env(format!(
                "Python not found at {}",
                path.display()
            )))
        }
    }

    fn pip_path(&self, env_dir: &Path) -> PathBuf {
        env_dir.join("bin").join("pip")
    }

    /// A stable environment id for a set of requirements.
    ///
    /// Callers with no durable pipeline identity — a one-off plan, whose id
    /// is a fresh timestamp — must key on this instead. Keying on the plan
    /// id gave every plan its own venv and its own `pip install`, which is
    /// unbounded: one short test suite left 17 GB of near-identical
    /// environments behind.
    pub fn env_id_for(requirements: &str) -> String {
        format!("reqs-{}", &Self::hash_requirements(requirements)[..16])
    }

    fn hash_requirements(requirements: &str) -> String {
        let mut hasher = Sha256::new();
        // Normalize: sort lines, trim whitespace, ignore comments
        let mut lines: Vec<&str> = requirements
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        lines.sort();
        for line in &lines {
            hasher.update(line.as_bytes());
            hasher.update(b"\n");
        }
        hex::encode(hasher.finalize())
    }

    fn parse_requirements(requirements: &str) -> HashMap<String, String> {
        let mut packages = HashMap::new();
        for line in requirements.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Parse "package==version", "package>=version", "package"
            let (name, version) = if let Some((n, v)) = line.split_once("==") {
                (n.trim().to_lowercase(), v.trim().to_string())
            } else if let Some((n, v)) = line.split_once(">=") {
                (n.trim().to_lowercase(), format!(">={v}"))
            } else if let Some((n, v)) = line.split_once("<=") {
                (n.trim().to_lowercase(), format!("<={v}"))
            } else {
                (line.to_lowercase(), "latest".to_string())
            };
            packages.insert(name, version);
        }
        packages
    }

    fn read_lockfile(&self, path: &Path) -> Result<EnvLockfile> {
        let content = std::fs::read_to_string(path).map_err(|e| WorkerError::Env(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| WorkerError::Encoding(e.to_string()))
    }

    fn write_lockfile(&self, path: &Path, requirements: &str, hash: &str) -> Result<()> {
        let lockfile = EnvLockfile {
            packages: Self::parse_requirements(requirements),
            requirements_hash: hash.to_string(),
            env_type: self.env_type.clone(),
            python_version: "3.11".to_string(),
        };
        let json =
            serde_json::to_string_pretty(&lockfile).map_err(|e| WorkerError::Env(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| WorkerError::Env(format!("Failed to write lockfile: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_requirements_stable() {
        let r1 = "numpy==1.26\nscikit-learn==1.4\n";
        let r2 = "scikit-learn==1.4\nnumpy==1.26\n"; // different order
        assert_eq!(
            EnvManager::hash_requirements(r1),
            EnvManager::hash_requirements(r2)
        );
    }

    /// Two plans with the same dependencies share one environment.
    ///
    /// The env id used to be the plan id, which is a fresh timestamp per
    /// plan: nothing was ever reused, every plan paid a full pip install,
    /// and the environments accumulated without bound.
    #[test]
    fn the_env_id_follows_the_requirements_not_the_caller() {
        let a = EnvManager::env_id_for("numpy==1.26\nscikit-learn==1.4\n");
        let b = EnvManager::env_id_for("scikit-learn==1.4\n numpy==1.26\n");
        assert_eq!(a, b, "the same dependency set must reuse one environment");

        let other = EnvManager::env_id_for("numpy==1.26\n");
        assert_ne!(a, other, "a different dependency set needs its own");
    }

    #[test]
    fn hash_requirements_ignores_comments() {
        let r1 = "numpy==1.26\n# comment\nscikit-learn==1.4\n";
        let r2 = "numpy==1.26\nscikit-learn==1.4\n";
        assert_eq!(
            EnvManager::hash_requirements(r1),
            EnvManager::hash_requirements(r2)
        );
    }

    #[test]
    fn hash_changes_on_version_change() {
        let r1 = "numpy==1.26\n";
        let r2 = "numpy==1.27\n";
        assert_ne!(
            EnvManager::hash_requirements(r1),
            EnvManager::hash_requirements(r2)
        );
    }

    #[test]
    fn parse_requirements_formats() {
        let pkgs = EnvManager::parse_requirements("numpy==1.26\nsklearn>=1.4\npandas\n");
        assert_eq!(pkgs["numpy"], "1.26");
        assert_eq!(pkgs["sklearn"], ">=1.4");
        assert_eq!(pkgs["pandas"], "latest");
    }
}
