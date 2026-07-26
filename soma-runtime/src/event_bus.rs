//! Broadcast event bus for runtime observability.
//!
//! Emits [`Event`]s (node started/completed/failed, cache hits, run lifecycle)
//! to all subscribers via a tokio broadcast channel.

use somatize_core::event::Event;
use somatize_core::tracking::EventSink;
use std::sync::{Arc, RwLock};
use tokio::sync::broadcast;

/// Async event bus for broadcasting execution events to multiple subscribers.
///
/// Uses tokio's broadcast channel internally. Subscribers receive all events
/// emitted after they subscribe. Events are cloned for each subscriber.
///
/// Two delivery paths with different guarantees:
/// - **Sinks** ([`add_sink`](Self::add_sink)) are invoked synchronously on
///   the emitting thread before the broadcast — lossless and ordered.
///   Trackers persist events through this path.
/// - **Subscribers** ([`subscribe`](Self::subscribe)) receive via the
///   broadcast channel — live but lossy under lag. Display/relay only.
pub struct EventBus {
    sender: broadcast::Sender<Event>,
    sinks: RwLock<Vec<Arc<dyn EventSink>>>,
}

impl EventBus {
    /// Create a new event bus with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            sinks: RwLock::new(Vec::new()),
        }
    }

    /// Register a lossless sink, called synchronously on every emit.
    pub fn add_sink(&self, sink: Arc<dyn EventSink>) {
        match self.sinks.write() {
            Ok(mut sinks) => sinks.push(sink),
            Err(poisoned) => poisoned.into_inner().push(sink),
        }
    }

    /// Unregister a previously added sink (matched by identity). The
    /// sink is flushed before removal.
    pub fn remove_sink(&self, sink: &Arc<dyn EventSink>) {
        sink.flush();
        let mut sinks = match self.sinks.write() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };
        sinks.retain(|s| !Arc::ptr_eq(s, sink));
    }

    /// Emit an event: sinks first (lossless), then all subscribers.
    /// Returns the number of broadcast receivers that received the event.
    /// If there are no subscribers, the broadcast is silently dropped.
    pub fn emit(&self, event: Event) -> usize {
        {
            let sinks = match self.sinks.read() {
                Ok(s) => s,
                Err(poisoned) => poisoned.into_inner(),
            };
            for sink in sinks.iter() {
                sink.record(&event);
            }
        }
        self.sender.send(event).unwrap_or(0)
    }

    /// Subscribe to receive events.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Flush all registered sinks.
    pub fn flush_sinks(&self) {
        let sinks = match self.sinks.read() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };
        for sink in sinks.iter() {
            sink.flush();
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::event::PlanSummary;
    use std::time::Duration;

    #[tokio::test]
    async fn emit_without_subscribers_succeeds() {
        let bus = EventBus::new(16);
        let count = bus.emit(Event::RunStarted {
            run_id: "r1".into(),
            plan_summary: PlanSummary {
                total_nodes: 1,
                cached_nodes: 0,
                parallel_branches: 0,
            },
        });
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn subscriber_receives_events() {
        let bus = EventBus::new(16);
        let mut rx = bus.subscribe();

        bus.emit(Event::RunStarted {
            run_id: "r1".into(),
            plan_summary: PlanSummary {
                total_nodes: 2,
                cached_nodes: 0,
                parallel_branches: 0,
            },
        });
        bus.emit(Event::RunCompleted {
            run_id: "r1".into(),
            duration: Duration::from_millis(100),
        });

        let e1 = rx.recv().await.unwrap();
        assert!(matches!(e1, Event::RunStarted { .. }));

        let e2 = rx.recv().await.unwrap();
        assert!(matches!(e2, Event::RunCompleted { .. }));
    }

    #[tokio::test]
    async fn multiple_subscribers() {
        let bus = EventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        assert_eq!(bus.subscriber_count(), 2);

        bus.emit(Event::RunCompleted {
            run_id: "r1".into(),
            duration: Duration::from_secs(1),
        });

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, Event::RunCompleted { .. }));
        assert!(matches!(e2, Event::RunCompleted { .. }));
    }

    #[tokio::test]
    async fn subscriber_after_emit_misses_earlier_events() {
        let bus = EventBus::new(16);
        bus.emit(Event::RunCompleted {
            run_id: "r1".into(),
            duration: Duration::from_secs(1),
        });

        let mut rx = bus.subscribe();
        bus.emit(Event::RunCompleted {
            run_id: "r2".into(),
            duration: Duration::from_secs(2),
        });

        let event = rx.recv().await.unwrap();
        if let Event::RunCompleted { run_id, .. } = event {
            assert_eq!(run_id, "r2"); // only sees r2, not r1
        } else {
            panic!("wrong event type");
        }
    }
}
