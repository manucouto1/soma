//! The side of a graph that decides rather than computes.
//!
//! A [`Step`](crate::graph::step::Step) does not transform its input; it
//! polls, asks the runtime for something it cannot do itself, and says
//! what should happen next. [`Effect`](effect::Effect) is that request —
//! call a model, run a tool, execute another graph, pause for a human —
//! and the runtime is what performs it.
//!
//! The two vocabularies an effect speaks in live beside it:
//! [`Message`](message::Message) for what is said to a model, and
//! [`ToolSpec`](tool::ToolSpec) for what a tool says about itself. Soma is
//! on both ends of the latter — it publishes tools over MCP and calls
//! tools on a model's behalf — so the description is one type, not two.
//!
//! Nothing here performs anything. This crate holds no runtime and opens
//! no socket; `soma-runtime` drives the effects and `soma-llm` reaches the
//! models.

pub mod effect;
pub mod message;
pub mod tool;
