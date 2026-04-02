//! End-to-end tests for EnvManager: creates real venvs, installs packages.
//!
//! These tests require `python3` to be available on the system.
//! They create temporary directories and real virtual environments.

use soma_worker::env_manager::{EnvManager, EnvType};
use std::path::PathBuf;

fn has_python3() -> bool {
    std::process::Command::new("python3")
        .args(["--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn create_venv_and_get_python() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let mgr = EnvManager::new(tmp.path(), EnvType::Venv);

    // Create env with no requirements
    let python = mgr.ensure_env("test_pipeline", "").unwrap();

    // Python binary should exist
    assert!(
        python.exists(),
        "Python binary should exist at {}",
        python.display()
    );

    // Should be inside the temp dir
    assert!(python.starts_with(tmp.path()));

    // Python should be executable
    let output = std::process::Command::new(&python)
        .args(["--version"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let version = String::from_utf8_lossy(&output.stdout);
    assert!(
        version.contains("Python 3"),
        "Should be Python 3, got: {version}"
    );
}

#[test]
fn reuse_existing_venv() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let mgr = EnvManager::new(tmp.path(), EnvType::Venv);

    // Create env
    let python1 = mgr.ensure_env("reuse_test", "").unwrap();

    // Call again with same requirements → should reuse
    let python2 = mgr.ensure_env("reuse_test", "").unwrap();

    assert_eq!(python1, python2, "should return same python path");
}

#[test]
fn different_requirements_trigger_update() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let mgr = EnvManager::new(tmp.path(), EnvType::Venv);

    // Create with requirements v1
    let _python1 = mgr.ensure_env("update_test", "").unwrap();

    // Update with different requirements (just a comment change triggers hash change)
    let python2 = mgr.ensure_env("update_test", "# updated\n").unwrap();

    // Should still work
    assert!(python2.exists());
}

#[test]
fn install_real_package() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let mgr = EnvManager::new(tmp.path(), EnvType::Venv);

    // Install a small, fast package
    let python = mgr.ensure_env("pkg_test", "six\n").unwrap();

    // Verify the package is importable
    let output = std::process::Command::new(&python)
        .args(["-c", "import six; print(six.__version__)"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "six should be importable. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cleanup_old_envs() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let mgr = EnvManager::new(tmp.path(), EnvType::Venv);

    // Create an env
    let _python = mgr.ensure_env("cleanup_test", "").unwrap();

    // Cleanup with 0 duration → should remove everything
    let removed = mgr.cleanup(std::time::Duration::from_secs(0));
    assert!(removed >= 1, "should have removed at least 1 env");
}

#[test]
fn lockfile_persists_between_calls() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let mgr = EnvManager::new(tmp.path(), EnvType::Venv);

    let _python = mgr.ensure_env("lock_test", "six==1.17.0\n").unwrap();

    // Lockfile should exist
    let lockfile_path = tmp.path().join("env-lock_test").join("lockfile.json");
    assert!(lockfile_path.exists(), "lockfile should be written");

    // Parse and verify
    let content = std::fs::read_to_string(&lockfile_path).unwrap();
    let lockfile: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(lockfile["packages"]["six"].is_string());
}
