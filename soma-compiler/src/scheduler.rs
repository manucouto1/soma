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

/// A worker's capabilities and current load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Stable identifier assignments and transfers refer to.
    pub id: String,
    /// Human-readable name, carried into [`Assignment::worker_name`] so a
    /// distribution plan reads without a worker lookup.
    pub name: String,
    /// Capability tags (e.g. `"gpu"`) matched against a plan's
    /// `RemoteTarget::Tag` requirements.
    pub tags: Vec<String>,
    /// Whether the worker has a GPU.
    pub gpu: bool,
    /// CPU cores available on the worker.
    pub cpu_cores: usize,
    /// Jobs currently running; the load side of [`available_slots`](Self::available_slots).
    pub active_jobs: usize,
    /// Upper bound on concurrent jobs; the capacity side of
    /// [`available_slots`](Self::available_slots).
    pub max_concurrent: usize,
}

impl WorkerInfo {
    /// How many more jobs this worker can take right now. Saturates at
    /// zero: a worker reporting more active jobs than its limit is full,
    /// not underflowed.
    pub fn available_slots(&self) -> usize {
        self.max_concurrent.saturating_sub(self.active_jobs)
    }

    /// Whether the worker can take at least one more job. [`schedule`]
    /// filters on this before any placement is attempted.
    pub fn has_capacity(&self) -> bool {
        self.available_slots() > 0
    }

    /// Whether the worker advertises `tag` among its capability tags.
    pub fn matches_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Assignment of a node/phase to a specific worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    /// The plan node being placed.
    pub node_id: String,
    /// Id of the chosen [`WorkerInfo`].
    pub worker_id: String,
    /// The worker's display name, copied here so the plan is readable on
    /// its own.
    pub worker_name: String,
    /// The kind of phase this node runs in.
    pub phase: Phase,
    /// Why the scheduler chose this worker ("least loaded worker",
    /// "grouped with differentiable neighbors", ...) — diagnostic text,
    /// not machine-read.
    pub reason: String,
}

/// Execution phase type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Nodes run one after another — kept on a single worker to avoid
    /// moving intermediate data.
    Sequential,
    /// Independent branches run concurrently — distributed across workers.
    Parallel,
    /// One trial of a study, placed round-robin across all workers.
    Trial {
        /// Zero-based index of this trial within the study.
        trial_index: usize,
        /// Total number of trials in the study.
        total: usize,
    },
}

/// The complete distribution plan produced by the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionPlan {
    /// One entry per placed node: which worker, and why.
    pub assignments: Vec<Assignment>,
    /// The ordered phases the plan executes in.
    pub phases: Vec<PlanPhase>,
    /// Data movements required where consecutive nodes landed on
    /// different workers.
    pub data_transfers: Vec<DataTransfer>,
    /// Non-fatal conditions ("No workers available — will execute
    /// locally", "All workers are at capacity"). An empty `assignments`
    /// with a warning means: run locally instead.
    pub warnings: Vec<String>,
}

/// A phase in the execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanPhase {
    /// Position of this phase in execution order, starting at 0.
    pub phase_index: usize,
    /// Whether the phase runs its nodes sequentially, in parallel, or as
    /// a study trial.
    pub phase_type: Phase,
    /// Every node the phase covers.
    pub node_ids: Vec<String>,
    /// The workers involved: one id for a sequential phase, one per
    /// branch for a parallel one.
    pub worker_ids: Vec<String>,
}

/// A data transfer between workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransfer {
    /// The node whose output must move.
    pub from_node: String,
    /// The node that consumes it on the other worker.
    pub to_node: String,
    /// Worker id the data currently lives on.
    pub from_worker: String,
    /// Worker id the data must reach.
    pub to_worker: String,
    /// How the data moves: `"s3"`, `"direct"`, or `"cached"`.
    pub transfer_type: String,
}

/// Mutable state accumulated during scheduling.
struct ScheduleState<'a> {
    workers: Vec<&'a WorkerInfo>,
    diff_nodes: &'a [String],
    assignments: Vec<Assignment>,
    phases: Vec<PlanPhase>,
    transfers: Vec<DataTransfer>,
    warnings: Vec<String>,
    phase_index: usize,
}

/// Schedule an execution plan across available workers.
pub fn schedule(
    plan: &ExecutionPlan,
    workers: &[WorkerInfo],
    differentiable_nodes: &[String],
) -> DistributionPlan {
    let mut state = ScheduleState {
        workers: Vec::new(),
        diff_nodes: differentiable_nodes,
        assignments: Vec::new(),
        phases: Vec::new(),
        transfers: Vec::new(),
        warnings: Vec::new(),
        phase_index: 0,
    };

    if workers.is_empty() {
        state
            .warnings
            .push("No workers available — will execute locally".into());
        return DistributionPlan {
            assignments: state.assignments,
            phases: state.phases,
            data_transfers: state.transfers,
            warnings: state.warnings,
        };
    }

    state.workers = workers.iter().filter(|w| w.has_capacity()).collect();
    if state.workers.is_empty() {
        state.warnings.push("All workers are at capacity".into());
        return DistributionPlan {
            assignments: state.assignments,
            phases: state.phases,
            data_transfers: state.transfers,
            warnings: state.warnings,
        };
    }

    schedule_plan(plan, &mut state, None);

    DistributionPlan {
        assignments: state.assignments,
        phases: state.phases,
        data_transfers: state.transfers,
        warnings: state.warnings,
    }
}

fn schedule_plan(plan: &ExecutionPlan, state: &mut ScheduleState<'_>, forced_worker: Option<&str>) {
    match plan {
        // A step schedules like any other single node. Its cost profile is
        // different — latency-bound rather than CPU-bound — which is a
        // reason to weight it differently once the scheduler models cost at
        // all; today it models load, and a step contributes load like the rest.
        ExecutionPlan::Execute { node_id } | ExecutionPlan::Step { node_id, .. } => {
            let worker = if let Some(fw) = forced_worker {
                state
                    .workers
                    .iter()
                    .find(|w| w.id == fw)
                    .unwrap_or(&state.workers[0])
            } else {
                least_loaded(&state.workers)
            };

            state.assignments.push(Assignment {
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
            let worker = forced_worker
                .and_then(|fw| state.workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(&state.workers));

            let node_ids = collect_node_ids(plan);
            let has_diff = node_ids.iter().any(|n| state.diff_nodes.contains(n));
            let force = if has_diff {
                Some(worker.id.as_str())
            } else {
                forced_worker
            };

            state.phases.push(PlanPhase {
                phase_index: state.phase_index,
                phase_type: Phase::Sequential,
                node_ids: node_ids.clone(),
                worker_ids: vec![worker.id.clone()],
            });
            state.phase_index += 1;

            for step in steps {
                schedule_plan(step, state, force);
            }
        }

        ExecutionPlan::Parallel(branches) => {
            let branch_ids: Vec<Vec<String>> = branches.iter().map(collect_node_ids).collect();
            let mut assigned_workers = Vec::new();

            for (i, branch) in branches.iter().enumerate() {
                let worker_idx = i % state.workers.len();
                let worker = state.workers[worker_idx];
                assigned_workers.push(worker.id.clone());

                let worker_id = worker.id.clone();
                schedule_plan(branch, state, Some(&worker_id));

                // Check if data transfer is needed from previous phase
                if let Some(prev) = state
                    .assignments
                    .iter()
                    .rev()
                    .find(|a| !branch_ids[i].contains(&a.node_id))
                    .filter(|prev| prev.worker_id != state.workers[worker_idx].id)
                {
                    state.transfers.push(DataTransfer {
                        from_node: prev.node_id.clone(),
                        to_node: branch_ids[i].first().cloned().unwrap_or_default(),
                        from_worker: prev.worker_id.clone(),
                        to_worker: state.workers[worker_idx].id.clone(),
                        transfer_type: "s3".into(),
                    });
                }
            }

            state.phases.push(PlanPhase {
                phase_index: state.phase_index,
                phase_type: Phase::Parallel,
                node_ids: branch_ids.into_iter().flatten().collect(),
                worker_ids: assigned_workers,
            });
            state.phase_index += 1;
        }

        ExecutionPlan::Remote { plan, .. } => {
            schedule_plan(plan, state, None);
        }

        ExecutionPlan::Loop { body, node_id, .. } => {
            let worker = forced_worker
                .and_then(|fw| state.workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(&state.workers));
            state.assignments.push(Assignment {
                node_id: node_id.clone(),
                worker_id: worker.id.clone(),
                worker_name: worker.name.clone(),
                phase: Phase::Sequential,
                reason: "loop controller".into(),
            });
            let worker_id = worker.id.clone();
            schedule_plan(body, state, Some(&worker_id));
        }

        ExecutionPlan::Branch { node_id, arms, .. } => {
            let worker = forced_worker
                .and_then(|fw| state.workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(&state.workers));
            state.assignments.push(Assignment {
                node_id: node_id.clone(),
                worker_id: worker.id.clone(),
                worker_name: worker.name.clone(),
                phase: Phase::Sequential,
                reason: "branch condition".into(),
            });
            let worker_id = worker.id.clone();
            for (_, arm_plan) in arms {
                schedule_plan(arm_plan, state, Some(&worker_id));
            }
        }

        ExecutionPlan::Composite { node_ids } => {
            let worker = forced_worker
                .and_then(|fw| state.workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(&state.workers));

            state.phases.push(PlanPhase {
                phase_index: state.phase_index,
                phase_type: Phase::Sequential,
                node_ids: node_ids.clone(),
                worker_ids: vec![worker.id.clone()],
            });
            state.phase_index += 1;

            let worker_id = worker.id.clone();
            for nid in node_ids {
                state.assignments.push(Assignment {
                    node_id: nid.clone(),
                    worker_id: worker.id.clone(),
                    worker_name: worker.name.clone(),
                    phase: Phase::Sequential,
                    reason: "composite block — same worker for gradient flow".into(),
                });
            }
            drop(worker_id);
        }

        ExecutionPlan::Stream { node_ids, .. } => {
            // Stream: all filters on the same worker for stateful chunk processing.
            let worker = forced_worker
                .and_then(|fw| state.workers.iter().find(|w| w.id == fw).copied())
                .unwrap_or_else(|| least_loaded(&state.workers));

            state.phases.push(PlanPhase {
                phase_index: state.phase_index,
                phase_type: Phase::Sequential,
                node_ids: node_ids.clone(),
                worker_ids: vec![worker.id.clone()],
            });
            state.phase_index += 1;

            for nid in node_ids {
                state.assignments.push(Assignment {
                    node_id: nid.clone(),
                    worker_id: worker.id.clone(),
                    worker_name: worker.name.clone(),
                    phase: Phase::Sequential,
                    reason: "stream block — same worker for stateful chunk processing".into(),
                });
            }
        }

        ExecutionPlan::Empty => {}
    }
}

/// The worker with the most free slots.
///
/// Precondition: `workers` is non-empty — `schedule()` returns early (with
/// a warning) for both "no workers" and "none with capacity" before any
/// call can reach here.
fn least_loaded<'a>(workers: &[&'a WorkerInfo]) -> &'a WorkerInfo {
    workers
        .iter()
        .max_by_key(|w| w.available_slots())
        .expect("schedule() filters out an empty worker set before placing")
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
                id: "w1".into(),
                name: "GPU-A100".into(),
                tags: vec!["gpu".into()],
                gpu: true,
                cpu_cores: 16,
                active_jobs: 0,
                max_concurrent: 4,
            },
            WorkerInfo {
                id: "w2".into(),
                name: "CPU-Server".into(),
                tags: vec!["cpu".into()],
                gpu: false,
                cpu_cores: 64,
                active_jobs: 1,
                max_concurrent: 8,
            },
        ]
    }

    #[test]
    fn sequential_same_worker() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "normalize".into(),
            },
            ExecutionPlan::Execute {
                node_id: "select".into(),
            },
            ExecutionPlan::Execute {
                node_id: "classify".into(),
            },
        ]);

        let result = schedule(&plan, &test_workers(), &[]);
        // All should be on the same worker
        let worker_ids: Vec<&str> = result
            .assignments
            .iter()
            .map(|a| a.worker_id.as_str())
            .collect();
        assert!(worker_ids.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn parallel_distributes() {
        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute {
                node_id: "train_svm".into(),
            },
            ExecutionPlan::Execute {
                node_id: "train_knn".into(),
            },
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
        let plan = ExecutionPlan::Execute {
            node_id: "test".into(),
        };
        let result = schedule(&plan, &[], &[]);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn sequence_then_parallel() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "load".into(),
            },
            ExecutionPlan::Execute {
                node_id: "normalize".into(),
            },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute {
                    node_id: "train_a".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "train_b".into(),
                },
            ]),
        ]);

        let result = schedule(&plan, &test_workers(), &[]);
        // load + normalize on same worker, train_a and train_b distributed
        assert!(result.assignments.len() >= 4);
        assert_eq!(
            result.assignments[0].worker_id,
            result.assignments[1].worker_id
        );
    }

    #[test]
    fn data_transfer_on_split() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "preprocess".into(),
            },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute {
                    node_id: "branch_a".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "branch_b".into(),
                },
            ]),
        ]);

        let result = schedule(&plan, &test_workers(), &[]);
        // Should have at least one data transfer (preprocess → branch on different worker)
        assert!(
            !result.data_transfers.is_empty()
                || result
                    .assignments
                    .iter()
                    .all(|a| a.worker_id == result.assignments[0].worker_id)
        );
    }
}
