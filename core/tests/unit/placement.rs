//! Dónde corre cada nodo, y lo que la colocación **no** toca.

use crate::dobles::Sumar;
use soma_next_core::{Device, Placement, compile, node};

#[test]
fn un_mapa_de_nodo_a_sitio() {
    let mut placement = Placement::new();
    assert!(placement.is_empty());

    assert_eq!(placement.place("a", Device::Cuda(0)), None);
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.len(), 1);
}

#[test]
fn colocar_otra_vez_devuelve_donde_estaba() {
    let mut placement = Placement::new();
    placement.place("a", Device::Cpu);
    assert_eq!(placement.place("a", Device::Cuda(0)), Some(Device::Cpu));
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
}

#[test]
fn sin_colocar_no_es_lo_mismo_que_en_cpu() {
    // «Donde ya esté» y «muévelo a la cpu» son órdenes distintas, y por eso
    // `of` devuelve `Option` en vez de un `Device::Cpu` por defecto.
    let mut placement = Placement::new();
    placement.place("a", Device::Cpu);
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cpu));
    assert_eq!(placement.of(&"b".into()), None);
}

// ── Lo que el DSL coloca ──

#[test]
fn on_coloca_todo_el_trozo() {
    let (_, _, placement) = ((node("a", Sumar(1.0)) >> node("b", Sumar(1.0))).on(Device::Cuda(0)))
        .somatize()
        .unwrap();

    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.of(&"b".into()), Some(&Device::Cuda(0)));
}

#[test]
fn gana_el_de_dentro() {
    let (_, _, placement) = ((node("a", Sumar(1.0)).on(Device::Cuda(0)) >> node("b", Sumar(1.0)))
        .on(Device::Meta))
    .somatize()
    .unwrap();

    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.of(&"b".into()), Some(&Device::Meta));
}

#[test]
fn cada_rama_en_su_sitio() {
    let (_, _, placement) = (node("fuente", Sumar(1.0))
        >> (node("izq", Sumar(1.0)).on(Device::Cuda(0)) | node("der", Sumar(1.0)).on(Device::Cpu)))
    .somatize()
    .unwrap();

    assert_eq!(placement.of(&"izq".into()), Some(&Device::Cuda(0)));
    assert_eq!(placement.of(&"der".into()), Some(&Device::Cpu));
    assert_eq!(placement.of(&"fuente".into()), None, "nadie la colocó");
}

#[test]
fn lo_que_no_se_coloca_se_queda_sin_colocar() {
    let (_, _, placement) = (node("a", Sumar(1.0)) >> node("b", Sumar(1.0)))
        .somatize()
        .unwrap();
    assert!(placement.is_empty());
}

// ── Y lo que no cambia ──

#[test]
fn colocar_no_cambia_el_plan() {
    // Es cierto por construcción —`compile` no ve la colocación—, y está
    // escrito para que se note el día que alguien intente meterla en el plan.
    let expresion = || {
        node("fuente", Sumar(1.0))
            >> (node("izq", Sumar(1.0)) | node("der", Sumar(1.0)))
            >> node("juntar", Sumar(1.0))
    };

    let (g, c, _) = expresion().somatize().unwrap();
    let (g_colocado, c_colocado, placement) = expresion().on(Device::Cuda(0)).somatize().unwrap();

    assert_eq!(placement.len(), 4, "los cuatro tienen sitio");
    assert_eq!(
        compile(&g, &c).unwrap(),
        compile(&g_colocado, &c_colocado).unwrap()
    );
}

#[test]
fn colocar_no_cambia_el_grafo() {
    // `Graph` sigue siendo solo topología, así que dos grafos iguales
    // colocados distinto son iguales **como grafos**.
    let (g, _, _) = (node("a", Sumar(1.0)) >> node("b", Sumar(1.0)))
        .somatize()
        .unwrap();
    let (g_colocado, _, _) = ((node("a", Sumar(1.0)) >> node("b", Sumar(1.0))).on(Device::Cuda(0)))
        .somatize()
        .unwrap();

    assert_eq!(g, g_colocado);
}
