//! Where each node runs, and what placing **does not** touch.

use crate::doubles::Add;
use somatize_core::{Device, Host, Placement, compile, node};

#[test]
fn a_map_from_node_to_place() {
    let mut placement = Placement::new();
    assert!(placement.is_empty());

    assert_eq!(placement.place("a", Device::Cuda(0)), None);
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.len(), 1);
}

#[test]
fn placing_again_returns_where_it_was() {
    let mut placement = Placement::new();
    placement.place("a", Device::Cpu);
    assert_eq!(placement.place("a", Device::Cuda(0)), Some(Device::Cpu));
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
}

#[test]
fn unplaced_is_not_the_same_as_on_cpu() {
    // "Wherever it already is" and "move it to the cpu" are different orders,
    // which is why `of` returns an `Option` instead of a default `Device::Cpu`.
    let mut placement = Placement::new();
    placement.place("a", Device::Cpu);
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cpu));
    assert_eq!(placement.of(&"b".into()), None);
}

#[test]
fn on_places_the_whole_piece() {
    let (_, _, placement, _) = ((node("a", Add(1.0)) >> node("b", Add(1.0))).on(Device::Cuda(0)))
        .somatize()
        .unwrap();

    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.of(&"b".into()), Some(&Device::Cuda(0)));
}

#[test]
fn the_innermost_one_wins() {
    let (_, _, placement, _) = ((node("a", Add(1.0)).on(Device::Cuda(0)) >> node("b", Add(1.0)))
        .on(Device::Meta))
    .somatize()
    .unwrap();

    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.of(&"b".into()), Some(&Device::Meta));
}

#[test]
fn each_branch_in_its_own_place() {
    let (_, _, placement, _) = (node("source", Add(1.0))
        >> (node("left", Add(1.0)).on(Device::Cuda(0)) | node("right", Add(1.0)).on(Device::Cpu)))
    .somatize()
    .unwrap();

    assert_eq!(placement.of(&"left".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.of(&"right".into()), Some(&Device::Cpu));
    assert_eq!(placement.of(&"source".into()), None, "nobody placed it");
}

#[test]
fn what_is_not_placed_stays_unplaced() {
    let (_, _, placement, _) = (node("a", Add(1.0)) >> node("b", Add(1.0)))
        .somatize()
        .unwrap();
    assert!(placement.is_empty());
}

#[test]
fn placing_does_not_change_the_plan() {
    // It is true by construction — `compile` does not see the placement — and it
    // is written down so it shows the day someone tries to put it in the plan.
    let expression = || {
        node("source", Add(1.0))
            >> (node("left", Add(1.0)) | node("right", Add(1.0)))
            >> node("join", Add(1.0))
    };

    let (g, c, _, _) = expression().somatize().unwrap();
    let (g_placed, c_placed, placement, _) = expression().on(Device::Cuda(0)).somatize().unwrap();

    assert_eq!(placement.len(), 4, "all four have a place");
    assert_eq!(
        compile(&g, &c).unwrap(),
        compile(&g_placed, &c_placed).unwrap()
    );
}

#[test]
fn placing_does_not_change_the_graph() {
    // `Graph` is still topology only, so two equal graphs placed differently are
    // equal **as graphs**.
    let (g, _, _, _) = (node("a", Add(1.0)) >> node("b", Add(1.0)))
        .somatize()
        .unwrap();
    let (g_placed, _, _, _) = ((node("a", Add(1.0)) >> node("b", Add(1.0))).on(Device::Cuda(0)))
        .somatize()
        .unwrap();

    assert_eq!(g, g_placed);
}

#[test]
fn a_node_can_go_to_a_host() {
    let mut placement = Placement::new();
    assert_eq!(placement.place_at("a", Host::new("w1")), None);
    assert_eq!(placement.host_of(&"a".into()), Some(&Host::new("w1")));
}

#[test]
fn sending_it_again_returns_where_it_was() {
    let mut placement = Placement::new();
    placement.place_at("a", Host::new("w1"));
    assert_eq!(
        placement.place_at("a", Host::new("w2")),
        Some(Host::new("w1"))
    );
}

#[test]
fn the_two_halves_do_not_shadow_each_other() {
    // What lets them be written in any order, and the reason they are two maps
    // and not a pair.
    let mut placement = Placement::new();
    placement.place("a", Device::Cuda(0));
    placement.place_at("a", Host::new("w1"));

    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.host_of(&"a".into()), Some(&Host::new("w1")));
    assert_eq!(placement.len(), 1, "it is one node, not two");
}

#[test]
fn the_hosts_it_names_come_back_once_each() {
    // A host named by three nodes is one rendezvous, not three.
    let mut placement = Placement::new();
    placement.place_at("a", Host::new("w1"));
    placement.place_at("b", Host::new("w1"));
    placement.place_at("c", Host::new("w2"));

    assert_eq!(placement.hosts(), vec![&Host::new("w1"), &Host::new("w2")]);
}

#[test]
fn a_placement_that_sends_nothing_away_names_no_hosts() {
    let mut placement = Placement::new();
    placement.place("a", Device::Cuda(0));

    assert!(placement.hosts().is_empty(), "a device is not a machine");
    assert!(placement.is_local());
}

#[test]
fn the_order_does_not_depend_on_the_order_they_were_placed_in() {
    // Not tidiness: these come out of a `HashMap`, so without sorting the order
    // rendezvous are asked for changes between two runs of the same graph.
    let mut one = Placement::new();
    for (id, host) in [("a", "w3"), ("b", "w1"), ("c", "w2")] {
        one.place_at(id, Host::new(host));
    }
    let mut other = Placement::new();
    for (id, host) in [("c", "w2"), ("a", "w3"), ("b", "w1")] {
        other.place_at(id, Host::new(host));
    }

    assert_eq!(one.hosts(), other.hosts());
    assert_eq!(
        one.hosts(),
        vec![&Host::new("w1"), &Host::new("w2"), &Host::new("w3")]
    );
}

#[test]
fn moving_a_node_elsewhere_leaves_no_ghost_behind() {
    // `place_at` replaces, so the host it left has to stop being named — a
    // client would otherwise ask a broker for a rendezvous nothing needs.
    let mut placement = Placement::new();
    placement.place_at("a", Host::new("w1"));
    placement.place_at("a", Host::new("w2"));

    assert_eq!(placement.hosts(), vec![&Host::new("w2")]);
}

#[test]
fn having_a_host_does_not_require_a_device_or_the_other_way_round() {
    let mut placement = Placement::new();
    placement.place_at("host_only", Host::new("w1"));
    placement.place("device_only", Device::Cpu);

    assert_eq!(placement.of(&"host_only".into()), None);
    assert_eq!(placement.host_of(&"device_only".into()), None);
    assert_eq!(placement.len(), 2);
}

#[test]
fn without_hosts_the_placement_is_local() {
    // It is what allows skipping `distribute` entirely without walking the plan.
    let mut placement = Placement::new();
    assert!(placement.is_local());

    placement.place("a", Device::Cuda(0));
    assert!(placement.is_local(), "a device sends nobody away");

    placement.place_at("a", Host::new("w1"));
    assert!(!placement.is_local());
}

#[test]
fn at_sends_the_whole_piece() {
    let (_, _, placement, _) = ((node("a", Add(1.0)) >> node("b", Add(1.0))).at("w1"))
        .somatize()
        .unwrap();

    assert_eq!(placement.host_of(&"a".into()), Some(&Host::new("w1")));
    assert_eq!(placement.host_of(&"b".into()), Some(&Host::new("w1")));
}

#[test]
fn with_hosts_the_innermost_one_wins_too() {
    let (_, _, placement, _) = ((node("a", Add(1.0)).at("w1") >> node("b", Add(1.0))).at("w2"))
        .somatize()
        .unwrap();

    assert_eq!(placement.host_of(&"a".into()), Some(&Host::new("w1")));
    assert_eq!(placement.host_of(&"b".into()), Some(&Host::new("w2")));
}

#[test]
fn an_inner_device_does_not_stop_the_outer_host_from_arriving() {
    // The counterexample that split the list in two: with one, `.at` would have
    // skipped `a` for already having a place, and it would have ended up
    // without a host for having asked for a GPU.
    let (_, _, placement, _) = ((node("a", Add(1.0)).on(Device::Cuda(0)) >> node("b", Add(1.0)))
        .at("w1"))
    .somatize()
    .unwrap();

    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.host_of(&"a".into()), Some(&Host::new("w1")));
    assert_eq!(placement.host_of(&"b".into()), Some(&Host::new("w1")));
    assert_eq!(placement.of(&"b".into()), None, "nobody gave it a device");
}

#[test]
fn the_order_of_on_and_at_does_not_matter() {
    let first = (node("a", Add(1.0)) >> node("b", Add(1.0)))
        .on(Device::Cuda(0))
        .at("w1")
        .somatize()
        .unwrap()
        .2;
    let afterwards = (node("a", Add(1.0)) >> node("b", Add(1.0)))
        .at("w1")
        .on(Device::Cuda(0))
        .somatize()
        .unwrap()
        .2;

    assert_eq!(first, afterwards);
}

#[test]
fn each_branch_to_its_own_host() {
    let (_, _, placement, _) = (node("source", Add(1.0))
        >> (node("left", Add(1.0)).at("w1") | node("right", Add(1.0)).at("w2")))
    .somatize()
    .unwrap();

    assert_eq!(placement.host_of(&"left".into()), Some(&Host::new("w1")));
    assert_eq!(placement.host_of(&"right".into()), Some(&Host::new("w2")));
    assert_eq!(
        placement.host_of(&"source".into()),
        None,
        "nobody sent the source away"
    );
}

#[test]
fn sending_away_changes_neither_the_graph_nor_what_compile_produces() {
    // As for devices: `compile` sees none of this. What changes is what comes
    // out of `distribute`, tested in `plan.rs`.
    let (g, c, _, _) = (node("a", Add(1.0)) >> node("b", Add(1.0)))
        .somatize()
        .unwrap();
    let (g_away, c_away, placement, _) = ((node("a", Add(1.0)) >> node("b", Add(1.0))).at("w1"))
        .somatize()
        .unwrap();

    assert_eq!(g, g_away);
    assert!(!placement.is_local());
    assert_eq!(compile(&g, &c).unwrap(), compile(&g_away, &c_away).unwrap());
}
