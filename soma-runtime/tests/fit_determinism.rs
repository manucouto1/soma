//! `fit` must answer with the same thing every time it is asked.

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheKey;
use somatize_core::error::Result;
use somatize_core::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::value::Value;
use somatize_runtime::runner::{LocalRunner, RunContext, Runner};
use somatize_runtime::{EventBus, MemoryCache, NodeCatalog};
use std::sync::Arc;

struct Plus(f64);

impl Filter for Plus {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Plus", &self.0.to_le_bytes()])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _s: &Value) -> Result<Value> {
        match x {
            Value::Tensor { values, shape } => Ok(Value::tensor(
                values.iter().map(|v| v + self.0).collect(),
                shape.clone(),
            )),
            _ => Ok(x.clone()),
        }
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "Plus".into(),
            kind: FilterKind::Stateless,
            cacheable: false,
            differentiable: false,
            deterministic: true,
            stream_mode: StreamMode::FixedState,
            distribution: Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}

fn first(v: &Value) -> f64 {
    match v {
        Value::Tensor { values, .. } => values[0],
        other => panic!("expected a tensor, got {other:?}"),
    }
}

/// The returned output came from `outputs.values().last()` — an arbitrary
/// entry of a `HashMap`, and the map also holds the run's input under
/// `__input_*`. A single-node fit therefore had an even chance of
/// answering with its own input, differently on each run, and
/// `fit_sequence` passes that value straight to the next step.
///
/// A fresh library per iteration because a `HashMap`'s ordering is fixed
/// once the map is built; the randomness shows up across maps.
#[test]
fn fit_answers_with_the_last_node_not_an_arbitrary_one() {
    let cache = MemoryCache::default();
    let bus = Arc::new(EventBus::new(16));
    let plan = ExecutionPlan::Execute {
        node_id: "a".into(),
    };

    for _ in 0..200 {
        let mut lib = NodeCatalog::new();
        lib.register("a", Box::new(Plus(10.0)));
        let ctx = RunContext::new(
            &lib,
            &cache,
            &bus,
            "r",
            somatize_runtime::executor::GraphInfo::new(),
        );
        let (out, _) = LocalRunner
            .fit(&plan, &ctx, &Value::tensor(vec![1.0], vec![1]), None)
            .unwrap();
        assert_eq!(
            first(&out),
            11.0,
            "fit returned something other than `a`'s output"
        );
    }
}
