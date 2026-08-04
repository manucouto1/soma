//! Autonomous research over Soma pipelines.
//!
//! One thing lives here: [`ResearchStep`], a [`Step`](somatize_core::step::Step)
//! that proposes an experiment, runs it as an
//! [`Effect::Graph`](somatize_core::effect::Effect), reads the metrics, and
//! decides whether to keep going.
//!
//! Everything else it needs already exists elsewhere and is not
//! reimplemented here: the loop is the effect driver's, the durability is
//! the journal's, the record is `somatize-memory`'s, and running a pipeline
//! is `GraphHandler`'s. What is left is the reasoning, and the reasoning is
//! the model's.
//!
//! This replaces a rule-based planner that varied one parameter through a
//! fixed list of values, with a comment in it saying an LLM would go here.
//!
//! ```ignore
//! let agent = ResearchStep::new("ollama/qwen2.5", "beat 0.8 F1", pipeline)
//!     .with_history(kb.all()?)      // start from what is already known
//!     .with_max_iterations(10);
//!
//! let mut catalog = NodeCatalog::new();
//! catalog.register_step("researcher", Box::new(agent));
//! ```

pub mod action;
pub mod research;

pub use action::Action;
pub use research::ResearchStep;
