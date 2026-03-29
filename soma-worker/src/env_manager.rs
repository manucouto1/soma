//! Python environment manager: creates and maintains isolated venvs/conda envs
//! per pipeline, with incremental dependency updates.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment type preference.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnvType {
    #[default]
    Venv,
    Conda,
}

/// Lockfile: tracks what's installed in an environment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvLockfile {
    pub packages: HashMap<String, String>, // name → version
    pub requirements_hash: String,
    pub env_type: EnvType,
    pub python_version: String,
}

/// Manages isolated Python environments for pipeline execution.
pub struct EnvManager {
    base_dir: PathBuf,
    env_type: EnvType,
}

impl EnvManager {
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
    pub fn ensure_env(
        &self,
        pipeline_id: &str,
        requirements: &str,
    ) -> Result<PathBuf, String> {
        let req_hash = Self::hash_requirements(requirements);
        let env_dir = self.base_dir.join(format!("env-{pipeline_id}"));
        let lockfile_path = env_dir.join("lockfile.json");

        // Check if env exists and is up to date
        if env_dir.exists() {
            if let Ok(lockfile) = self.read_lockfile(&lockfile_path) {
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
                if let Ok(meta) = entry.metadata() {
                    if let Ok(modified) = meta.modified() {
                        if modified.elapsed().unwrap_or_default() > max_age {
                            let _ = std::fs::remove_dir_all(entry.path());
                            removed += 1;
                        }
                    }
                }
            }
        }
        removed
    }

    // ── Internal ──

    fn create_env(&self, env_dir: &Path) -> Result<(), String> {
        match self.env_type {
            EnvType::Venv => {
                let output = Command::new("python3")
                    .args(["-m", "venv", &env_dir.to_string_lossy()])
                    .output()
                    .map_err(|e| format!("Failed to create venv: {e}"))?;
                if !output.status.success() {
                    return Err(format!(
                        "venv creation failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
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
                    .map_err(|e| format!("Failed to create conda env: {e}"))?;
                if !output.status.success() {
                    return Err(format!(
                        "conda create failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
            }
        }
        Ok(())
    }

    fn install_requirements(&self, env_dir: &Path, requirements: &str) -> Result<(), String> {
        let pip = self.pip_path(env_dir);

        // Write requirements to temp file
        let req_file = env_dir.join("requirements.txt");
        std::fs::write(&req_file, requirements)
            .map_err(|e| format!("Failed to write requirements.txt: {e}"))?;

        // Always ensure soma is installed
        let _ = Command::new(&pip)
            .args(["install", "soma"])
            .output();

        let output = Command::new(&pip)
            .args(["install", "-r", &req_file.to_string_lossy(), "-q"])
            .output()
            .map_err(|e| format!("pip install failed: {e}"))?;

        if !output.status.success() {
            return Err(format!(
                "pip install failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    fn incremental_update(
        &self,
        env_dir: &Path,
        new_requirements: &str,
        old_lockfile: &EnvLockfile,
    ) -> Result<(), String> {
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
                .map_err(|e| format!("pip install failed: {e}"))?;

            if !output.status.success() {
                return Err(format!(
                    "pip install failed:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }

        Ok(())
    }

    fn python_path(&self, env_dir: &Path) -> Result<PathBuf, String> {
        let path = env_dir.join("bin").join("python");
        if path.exists() {
            Ok(path)
        } else {
            Err(format!("Python not found at {}", path.display()))
        }
    }

    fn pip_path(&self, env_dir: &Path) -> PathBuf {
        env_dir.join("bin").join("pip")
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

    fn read_lockfile(&self, path: &Path) -> Result<EnvLockfile, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&content).map_err(|e| e.to_string())
    }

    fn write_lockfile(
        &self,
        path: &Path,
        requirements: &str,
        hash: &str,
    ) -> Result<(), String> {
        let lockfile = EnvLockfile {
            packages: Self::parse_requirements(requirements),
            requirements_hash: hash.to_string(),
            env_type: self.env_type.clone(),
            python_version: "3.11".to_string(),
        };
        let json = serde_json::to_string_pretty(&lockfile).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("Failed to write lockfile: {e}"))
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
