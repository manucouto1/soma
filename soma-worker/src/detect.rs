//! Auto-detection of hardware capabilities and resource limiting.
//!
//! Scans the system for CPU cores, RAM, GPUs, and Python environments.
//! Users can apply [`ResourceLimits`] to restrict what fraction of the
//! hardware a worker exposes (Slurm-style).

use crate::protocol::{Capabilities, GpuInfo};
use std::path::Path;
use std::process::Command;

/// Limits on what fraction of detected hardware the worker may use.
///
/// Any `None` field means "use all available".
///
/// ```bash
/// soma-worker --cpus 4 --memory 8G --gpus 1 --max-concurrent 2
/// ```
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Max CPU cores to expose (None = all detected).
    pub max_cpus: Option<usize>,
    /// Max RAM in bytes (None = all detected).
    pub max_memory_bytes: Option<u64>,
    /// Max GPUs to expose (None = all detected).
    pub max_gpus: Option<usize>,
    /// Max concurrent plans this worker will accept.
    pub max_concurrent: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpus: None,
            max_memory_bytes: None,
            max_gpus: None,
            max_concurrent: 4,
        }
    }
}

impl Capabilities {
    /// Auto-detect hardware capabilities of the current machine.
    pub fn detect() -> Self {
        let sys = sysinfo::System::new_all();

        let cpu_cores = sys.cpus().len();
        let ram_bytes = sys.total_memory();
        let gpus = detect_gpus();
        let python_envs = detect_python_envs();

        // Auto-tag based on detected hardware
        let mut tags = Vec::new();
        if !gpus.is_empty() {
            tags.push("gpu".to_string());
        }
        tags.push("cpu".to_string());

        Self {
            cpu_cores,
            ram_bytes,
            gpus,
            python_envs,
            tags,
        }
    }

    /// Apply resource limits: effective = min(detected, limit).
    pub fn with_limits(mut self, limits: &ResourceLimits) -> Self {
        if let Some(max_cpus) = limits.max_cpus {
            self.cpu_cores = self.cpu_cores.min(max_cpus);
        }
        if let Some(max_mem) = limits.max_memory_bytes {
            self.ram_bytes = self.ram_bytes.min(max_mem);
        }
        if let Some(max_gpus) = limits.max_gpus {
            self.gpus.truncate(max_gpus);
        }
        self
    }

    /// Summary string for logging.
    pub fn summary(&self) -> String {
        let gpu_str = if self.gpus.is_empty() {
            "none".to_string()
        } else {
            self.gpus
                .iter()
                .map(|g| {
                    format!(
                        "{} ({:.1} GB)",
                        g.name,
                        g.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "{} CPUs, {:.1} GB RAM, GPUs: {}, Python: {:?}, tags: {:?}",
            self.cpu_cores,
            self.ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            gpu_str,
            self.python_envs,
            self.tags,
        )
    }
}

/// Detect NVIDIA GPUs via nvidia-smi.
fn detect_gpus() -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return vec![], // no nvidia-smi or no GPUs
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(2, ',').map(|s| s.trim()).collect();
            if parts.len() == 2 {
                let name = parts[0].to_string();
                let memory_mb: u64 = parts[1].parse().unwrap_or(0);
                Some(GpuInfo {
                    name,
                    memory_bytes: memory_mb * 1024 * 1024,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Detect available Python interpreters.
fn detect_python_envs() -> Vec<String> {
    let candidates = ["python3", "python"];
    let mut envs = Vec::new();

    for cmd in &candidates {
        let Ok(output) = Command::new(cmd).args(["--version"]).output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let version = String::from_utf8_lossy(&output.stdout);
        let version = version.trim();
        if let Ok(which) = Command::new("which").arg(cmd).output() {
            let path = String::from_utf8_lossy(&which.stdout).trim().to_string();
            envs.push(format!("{version} ({path})"));
        } else {
            envs.push(version.to_string());
        }
    }

    // Detect conda envs
    if let Ok(output) = Command::new("conda")
        .args(["env", "list", "--json"])
        .output()
        && output.status.success()
        && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout)
        && let Some(envs_arr) = json.get("envs").and_then(|v| v.as_array())
    {
        for env in envs_arr {
            if let Some(path) = env.as_str() {
                let name = Path::new(path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                if !name.is_empty() {
                    envs.push(format!("conda:{name}"));
                }
            }
        }
    }

    envs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_finds_cpus_and_ram() {
        let caps = Capabilities::detect();
        assert!(caps.cpu_cores > 0, "should detect at least 1 CPU");
        assert!(caps.ram_bytes > 0, "should detect RAM");
        assert!(caps.tags.contains(&"cpu".to_string()));
    }

    #[test]
    fn limits_restrict_capabilities() {
        let caps = Capabilities {
            cpu_cores: 16,
            ram_bytes: 64 * 1024 * 1024 * 1024,
            gpus: vec![
                GpuInfo {
                    name: "A100".into(),
                    memory_bytes: 80_000_000_000,
                },
                GpuInfo {
                    name: "A100".into(),
                    memory_bytes: 80_000_000_000,
                },
            ],
            python_envs: vec![],
            tags: vec![],
        };

        let limited = caps.with_limits(&ResourceLimits {
            max_cpus: Some(4),
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024),
            max_gpus: Some(1),
            max_concurrent: 2,
        });

        assert_eq!(limited.cpu_cores, 4);
        assert_eq!(limited.ram_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(limited.gpus.len(), 1);
    }

    #[test]
    fn limits_none_keeps_all() {
        let caps = Capabilities {
            cpu_cores: 8,
            ram_bytes: 32_000_000_000,
            gpus: vec![],
            python_envs: vec![],
            tags: vec![],
        };

        let limited = caps.with_limits(&ResourceLimits::default());
        assert_eq!(limited.cpu_cores, 8);
        assert_eq!(limited.ram_bytes, 32_000_000_000);
    }

    #[test]
    fn summary_format() {
        let caps = Capabilities {
            cpu_cores: 4,
            ram_bytes: 8 * 1024 * 1024 * 1024,
            gpus: vec![],
            python_envs: vec!["Python 3.11".into()],
            tags: vec!["cpu".into()],
        };
        let s = caps.summary();
        assert!(s.contains("4 CPUs"));
        assert!(s.contains("8.0 GB RAM"));
    }
}
