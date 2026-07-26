//! Research agent loop for autonomous experimentation.
//!
//! An [`Agent`] iterates through explore → experiment → evaluate cycles,
//! using a [`ResearchPlan`] to generate actions and a `KnowledgeBase`
//! to record results and guide future decisions.

pub mod action;
pub mod agent;
#[cfg(test)]
mod agent_tests;
pub mod planner;

pub use action::Action;
pub use agent::{Agent, Decision};
pub use planner::{ResearchPlan, SimpleResearchPlan};
