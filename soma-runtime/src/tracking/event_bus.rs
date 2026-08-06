//! Broadcast event bus for runtime observability.
//!
//! Emits [`Event`]s (node started/completed/failed, cache hits, run lifecycle)
//! to all subscribers via a tokio broadcast channel.

use somatize_core::tracking::EventSink;
use somatize_core::tracking::event::Event;
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
        for sink in self.snapshot_sinks() {
            sink.record(&event);
        }
        self.sender.send(event).unwrap_or(0)
    }

    /// Subscribe to receive events.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    /// Flush all registered sinks.
    pub fn flush_sinks(&self) {
        for sink in self.snapshot_sinks() {
            sink.flush();
        }
    }

    /// Copy the sink list out from under the lock.
    ///
    /// A sink is user code, and calling it with the read guard alive means a
    /// sink that registers or removes another one deadlocks: `RwLock` is not
    /// reentrant. Cloning is an atomic increment per sink, paid once per
    /// event — cheaper than the class of hang it removes.
    fn snapshot_sinks(&self) -> Vec<Arc<dyn EventSink>> {
        match self.sinks.read() {
            Ok(s) => s.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
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
    use somatize_core::tracking::event::PlanSummary;
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

        bus.emit(Event::RunCompleted {
            run_id: "r1".into(),
            duration: Duration::from_secs(1),
        });

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, Event::RunCompleted { .. }));
        assert!(matches!(e2, Event::RunCompleted { .. }));
    }

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Spy sink counting records and flushes.
    #[derive(Default)]
    struct CountingSink {
        records: AtomicUsize,
        flushes: AtomicUsize,
    }

    impl somatize_core::tracking::EventSink for CountingSink {
        fn record(&self, _event: &Event) {
            self.records.fetch_add(1, Ordering::SeqCst);
        }
        fn flush(&self) {
            self.flushes.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn run_completed(id: &str) -> Event {
        Event::RunCompleted {
            run_id: id.into(),
            duration: Duration::from_millis(1),
        }
    }

    /// A sink that touches the bus from inside `record`.
    ///
    /// `RwLock` is not reentrant, so this used to deadlock: `emit` held the
    /// read guard while calling user code, and the user code asked for the
    /// write guard.
    struct ReentrantSink {
        bus: std::sync::Weak<EventBus>,
        added: AtomicUsize,
    }

    impl somatize_core::tracking::EventSink for ReentrantSink {
        fn record(&self, _event: &Event) {
            // Only once, or the bus would grow a sink per event.
            if self.added.fetch_add(1, Ordering::SeqCst) == 0
                && let Some(bus) = self.bus.upgrade()
            {
                bus.add_sink(Arc::new(CountingSink::default()));
            }
        }
        fn flush(&self) {}
    }

    #[test]
    fn a_sink_may_touch_the_bus_from_inside_record() {
        let bus = Arc::new(EventBus::new(16));
        bus.add_sink(Arc::new(ReentrantSink {
            bus: Arc::downgrade(&bus),
            added: AtomicUsize::new(0),
        }));

        // Deadlocking here hangs the test binary rather than failing it,
        // which is the loudest signal available without a watchdog thread.
        bus.emit(run_completed("r1"));
        bus.emit(run_completed("r2"));

        bus.flush_sinks();
    }

    #[test]
    fn sinks_observe_events_synchronously_before_emit_returns() {
        let bus = EventBus::new(16);
        let sink = Arc::new(CountingSink::default());
        bus.add_sink(sink.clone());

        bus.emit(run_completed("r1"));
        // No polling, no await: the sink path is synchronous.
        assert_eq!(sink.records.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn remove_sink_flushes_detaches_and_respects_identity() {
        let bus = EventBus::new(16);
        let sink = Arc::new(CountingSink::default());
        let as_dyn: Arc<dyn somatize_core::tracking::EventSink> = sink.clone();
        bus.add_sink(as_dyn.clone());

        bus.emit(run_completed("r1"));
        assert_eq!(sink.flushes.load(Ordering::SeqCst), 0);

        // A DIFFERENT Arc wrapping an equal-valued sink is not removed.
        let other: Arc<dyn somatize_core::tracking::EventSink> = Arc::new(CountingSink::default());
        bus.remove_sink(&other);
        bus.emit(run_completed("r2"));
        assert_eq!(sink.records.load(Ordering::SeqCst), 2, "still attached");

        // Removing by identity flushes first, then detaches.
        bus.remove_sink(&as_dyn);
        assert_eq!(sink.flushes.load(Ordering::SeqCst), 1, "flushed on removal");
        bus.emit(run_completed("r3"));
        assert_eq!(
            sink.records.load(Ordering::SeqCst),
            2,
            "no events after removal"
        );

        // Removing a never-added sink is a harmless no-op.
        bus.remove_sink(&as_dyn);
    }

    #[test]
    fn remove_sink_drops_every_clone_of_a_doubly_registered_arc() {
        // CONTRACT: registering the same Arc twice means two deliveries
        // per event, and remove_sink detaches BOTH registrations.
        let bus = EventBus::new(16);
        let sink = Arc::new(CountingSink::default());
        let as_dyn: Arc<dyn somatize_core::tracking::EventSink> = sink.clone();
        bus.add_sink(as_dyn.clone());
        bus.add_sink(as_dyn.clone());

        bus.emit(run_completed("r1"));
        assert_eq!(sink.records.load(Ordering::SeqCst), 2);

        bus.remove_sink(&as_dyn);
        bus.emit(run_completed("r2"));
        assert_eq!(sink.records.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn flush_sinks_flushes_all_registered_sinks() {
        let bus = EventBus::new(16);
        let a = Arc::new(CountingSink::default());
        let b = Arc::new(CountingSink::default());
        bus.add_sink(a.clone());
        bus.add_sink(b.clone());
        bus.flush_sinks();
        assert_eq!(a.flushes.load(Ordering::SeqCst), 1);
        assert_eq!(b.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn sinks_stay_lossless_while_subscribers_lag() {
        // The documented contrast between the two delivery paths: a
        // lagging broadcast subscriber drops events; sinks never do.
        let bus = EventBus::new(4); // tiny broadcast capacity
        let sink = Arc::new(CountingSink::default());
        bus.add_sink(sink.clone());
        let mut rx = bus.subscribe();

        for i in 0..100 {
            bus.emit(run_completed(&format!("r{i}")));
        }
        assert_eq!(sink.records.load(Ordering::SeqCst), 100, "sink is lossless");
        match rx.recv().await {
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                assert!(n > 0, "subscriber lost {n} events");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }
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
