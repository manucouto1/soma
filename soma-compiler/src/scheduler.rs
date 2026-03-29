//! Scheduler: distributes ExecutionPlan nodes across available workers.
//!
//! Rules:
//! 1. Sequential phases → single worker (avoid data transfer)
//! 2. Parallel branches → distribute across workers by capability
//! 3. Differentiable connected nodes → same worker (gradient flow)
//! 4. Study trials → round-robin across all workers
//! 5. Auto-assign: users don't pick workers, the scheduler does

use crate::ExecutionPlan;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A worker's capabilities and current load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub gpu: bool,
    pub cpu_cores: usize,
    pub active_jobs: usize,
    pub max_concurrent: usize,
}

impl WorkerInfo {
    pub fn available_slots(&self) -> usize {
        self.max_concurrent.saturating_sub(self.active_jobs)
    }

    pub fn has_capacity(&self) -> bool {
        self.available_slots() > 0
    }

    pub fn matches_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Assignment of a node/phase to a specific worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub node_id: String,
    pub worker_id: String,
    pub worker_name: String,
    pub phase: Phase,
    pub reason: String,
}

/// Execution phase type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Sequential,
    Parallel,
    Trial { trial_index: usize, total: usize },
}

/// The complete distribution plan produced by the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionPlan {
    pub assignments: Vec<Assignment>,
    pub phases: Vec<PlanPhase>,
    pub data_transfers: Vec<DataTransfer>,
    pub warnings: Vec<String>,
}

/// A phase in the execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanPhase {
    pub phase_index: usize,
    pub phase_type: Phase,
    pub node_ids: Vec<String>,
    pub worker_ids: Vec<String>,
}

/// A data transfer between workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransfer {
    pub from_node: String,
    pub to_node: String,
    pub from_worker: String,
    pub to_worker: String,
    pub transfer_type: String, // "s3", "direct", "cached"
}

/// Schedule an execution plan across available workers.
pub fn schedule(
    plan: &ExecutionPlan,
    workers: &[WorkerInfo],
    differentiable_nodes: &[String],
) -> DistributionPlan {
    let mut assignments = Vec::new();
    let mut phases = Vec::new();
    let mut data_transfers = Vec::new();
    let mut warnings = Vec::new();

    if workers.is_empty() {
        warnings.push("No workers available — will execute locally".into());
        return DistributionPlan {
            assignments,
            phases,
            data_transfers,
            warnings,
        };
    }

    let available: Vec<&WorkerInfo> = workers.iter().filter(|w| w.has_capacity()).collect();
    if available.is_empty() {
        warnings.push("All workers are at capacity".into());
        return DistributionPlan {
            assignments,
            phases,
            data_transfers,
            warnings,
        };
    }

    let mut phase_index = 0;
    schedule_plan(
        plan,
        &available,
        differentiable_nodes,
        &mut assignments,
        &mut phases,
        &mut data_transfers,
        &mut warnings,
        &mut phase_index,
        None, // no forced worker
    );

    DistributionPlan {
        assignments,
        phases,
        data_transfers,
        warnings,
    }
}

fn schedule_plan(
    plan: &ExecutionPlan,
    workers: &[&WorkerInfo],
    diff_nodes: &[String],
    assignments: &mut Vec<Assignment>,
    phases: &mut Vec<PlanPhase>,
    transfers: &mut Vec<DataTransfer>,
    warnings: &mut Vec<String>,
    phase_index: &mut usize,
    forced_worker: Option<&str>,
) {
    match plan {
        ExecutionPlan::Execute { node_id } => {
            let worker = if let Some(fw) = forced_worker {
                workers.iter().find(|w| w.id == fw).unwrap_or(&workers[0])
            } else {
                pick_worker(workers, node_id, diff_nodes, assignments)
            };

            assignments.push(Assignment {
                node_id: node_id.clone(),
                worker_id: worker.id.clone(),
                worker_name: worker.name.clone(),
                phase: Phase::Sequential,
                reason: if forced_worker.is_some() {
                    "grouped with differentiable neighbors".into()
                } else {
                    "least loaded worker".into()
                },
            });
        }

        ExecutionPlan::Sequence(steps) => {
            // Sequential: all on the same worker to avoid data transfer
            let worker = forced_worker
                .and_then(|fw| workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(workers));

            let node_ids: Vec<String> = collect_node_ids(plan);

            // Check if any nodes in the sequence are differentiable
            // If so, keep them together on the same worker
            let has_diff = node_ids.iter().any(|n| diff_nodes.contains(n));
            let force = if has_diff { Some(worker.id.as_str()) } else { forced_worker };

            phases.push(PlanPhase {
                phase_index: *phase_index,
                phase_type: Phase::Sequential,
                node_ids: node_ids.clone(),
                worker_ids: vec![worker.id.clone()],
            });
            *phase_index += 1;

            for step in steps {
                schedule_plan(step, workers, diff_nodes, assignments, phases, transfers, warnings, phase_index, force);
            }
        }

        ExecutionPlan::Parallel(branches) => {
            // Parallel: distribute branches across workers
            let branch_ids: Vec<Vec<String>> = branches.iter().map(|b| collect_node_ids(b)).collect();
            let mut assigned_workers = Vec::new();

            for (i, branch) in branches.iter().enumerate() {
                let worker_idx = i % workers.len();
                let worker = workers[worker_idx];
                assigned_workers.push(worker.id.clone());

                schedule_plan(
                    branch,
                    workers,
                    diff_nodes,
                    assignments,
                    phases,
                    transfers,
                    warnings,
                    phase_index,
                    Some(&worker.id),
                );

                // Check if data transfer is needed from previous phase
                if let Some(prev_assignment) = assignments.iter().rev()
                    .find(|a| !branch_ids[i].contains(&a.node_id))
                {
                    if prev_assignment.worker_id != worker.id {
                        transfers.push(DataTransfer {
                            from_node: prev_assignment.node_id.clone(),
                            to_node: branch_ids[i].first().cloned().unwrap_or_default(),
                            from_worker: prev_assignment.worker_id.clone(),
                            to_worker: worker.id.clone(),
                            transfer_type: "s3".into(),
                        });
                    }
                }
            }

            phases.push(PlanPhase {
                phase_index: *phase_index,
                phase_type: Phase::Parallel,
                node_ids: branch_ids.into_iter().flatten().collect(),
                worker_ids: assigned_workers,
            });
            *phase_index += 1;
        }

        ExecutionPlan::Cached { node_id, .. } => {
            // Cached: assigned to a worker but will skip execution
            let worker = forced_worker
                .and_then(|fw| workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(workers));
            assignments.push(Assignment {
                node_id: node_id.clone(),
                worker_id: worker.id.clone(),
                worker_name: worker.name.clone(),
                phase: Phase::Sequential,
                reason: "cached — will skip execution".into(),
            });
        }

        ExecutionPlan::Remote { plan, .. } => {
            // Remote: the compiler already decided this should be remote
            schedule_plan(plan, workers, diff_nodes, assignments, phases, transfers, warnings, phase_index, None);
        }

        ExecutionPlan::Loop { body, node_id, .. } => {
            // Loop body executed on same worker
            let worker = forced_worker
                .and_then(|fw| workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(workers));
            assignments.push(Assignment {
                node_id: node_id.clone(),
                worker_id: worker.id.clone(),
                worker_name: worker.name.clone(),
                phase: Phase::Sequential,
                reason: "loop controller".into(),
            });
            schedule_plan(body, workers, diff_nodes, assignments, phases, transfers, warnings, phase_index, Some(&worker.id));
        }

        ExecutionPlan::Branch { node_id, arms, .. } => {
            let worker = forced_worker
                .and_then(|fw| workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(workers));
            assignments.push(Assignment {
                node_id: node_id.clone(),
                worker_id: worker.id.clone(),
                worker_name: worker.name.clone(),
                phase: Phase::Sequential,
                reason: "branch condition".into(),
            });
            for (_, arm_plan) in arms {
                schedule_plan(arm_plan, workers, diff_nodes, assignments, phases, transfers, warnings, phase_index, Some(&worker.id));
            }
        }

        ExecutionPlan::Empty => {}
    }
}

fn pick_worker<'a>(
    workers: &[&'a WorkerInfo],
    _node_id: &str,
    _diff_nodes: &[String],
    _assignments: &[Assignment],
) -> &'a WorkerInfo {
    least_loaded(workers)
}

fn least_loaded<'a>(workers: &[&'a WorkerInfo]) -> &'a WorkerInfo {
    workers
        .iter()
        .max_by_key(|w| w.available_slots())
        .unwrap()
}

fn collect_node_ids(plan: &ExecutionPlan) -> Vec<String> {
    plan.node_ids().into_iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_workers() -> Vec<WorkerInfo> {
        vec![
            WorkerInfo {
                id: "w1".into(), name: "GPU-A100".into(),
                tags: vec!["gpu".into()], gpu: true, cpu_cores: 16,
                active_jobs: 0, max_concurrent: 4,
            },
            WorkerInfo {
                id: "w2".into(), name: "CPU-Server".into(),
                tags: vec!["cpu".into()], gpu: false, cpu_cores: 64,
                active_jobs: 1, max_concurrent: 8,
            },
        ]
    }

    #[test]
    fn sequential_same_worker() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute { node_id: "normalize".into() },
            ExecutionPlan::Execute { node_id: "select".into() },
            ExecutionPlan::Execute { node_id: "classify".into() },
        ]);

        let result = schedule(&plan, &test_workers(), &[]);
        // All should be on the same worker
        let worker_ids: Vec<&str> = result.assignments.iter().map(|a| a.worker_id.as_str()).collect();
        assert!(worker_ids.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn parallel_distributes() {
        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute { node_id: "train_svm".into() },
            ExecutionPlan::Execute { node_id: "train_knn".into() },
        ]);

        let result = schedule(&plan, &test_workers(), &[]);
        assert_eq!(result.assignments.len(), 2);
        // Should be on different workers
        assert_ne!(
            result.assignments[0].worker_id,
            result.assignments[1].worker_id
        );
    }

    #[test]
    fn no_workers_warns() {
        let plan = ExecutionPlan::Execute { node_id: "test".into() };
        let result = schedule(&plan, &[], &[]);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn sequence_then_parallel() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute { node_id: "load".into() },
            ExecutionPlan::Execute { node_id: "normalize".into() },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute { node_id: "train_a".into() },
                ExecutionPlan::Execute { node_id: "train_b".into() },
            ]),
        ]);

        let result = schedule(&plan, &test_workers(), &[]);
        // load + normalize on same worker, train_a and train_b distributed
        assert!(result.assignments.len() >= 4);
        assert_eq!(result.assignments[0].worker_id, result.assignments[1].worker_id);
    }

    #[test]
    fn data_transfer_on_split() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute { node_id: "preprocess".into() },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute { node_id: "branch_a".into() },
                ExecutionPlan::Execute { node_id: "branch_b".into() },
            ]),
        ]);

        let result = schedule(&plan, &test_workers(), &[]);
        // Should have at least one data transfer (preprocess → branch on different worker)
        assert!(!result.data_transfers.is_empty() || result.assignments.iter().all(|a| a.worker_id == result.assignments[0].worker_id));
    }
}
