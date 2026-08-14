//! Compilar: de la estructura a la forma decidida.

use crate::dobles::{Preguntar, Sumar};
use soma_next_core::{Catalog, CompileError, Graph, NodeId, Plan, compile};
use std::sync::Arc;

fn con_filtros(ids: &[&str]) -> (Graph, Catalog) {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ids {
        g.add_node(*id).unwrap();
        c.insert(*id, Arc::new(Sumar(1.0)));
    }
    (g, c)
}

fn ejecuta(node: &str, from: &[&str]) -> Plan {
    Plan::Execute {
        node: node.into(),
        from: from.iter().map(|id| NodeId::from(*id)).collect(),
    }
}

#[test]
fn un_grafo_vacio_compila_a_nada() {
    assert_eq!(
        compile(&Graph::new(), &Catalog::new()).unwrap(),
        Plan::Empty
    );
}

#[test]
fn un_solo_filtro_no_se_envuelve_en_secuencia() {
    let (g, c) = con_filtros(&["a"]);
    assert_eq!(compile(&g, &c).unwrap(), ejecuta("a", &[]));
}

#[test]
fn cada_paso_lleva_escrito_de_donde_sale_su_entrada() {
    let (mut g, c) = con_filtros(&["a", "b", "c"]);
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![
            ejecuta("a", &[]),
            ejecuta("b", &["a"]),
            ejecuta("c", &["b"]),
        ])
    );
}

#[test]
fn el_plan_no_distingue_quien_pide_turnos_de_quien_no() {
    // `Preguntar` pide algo antes de terminar y `Sumar` no, y aun así los dos
    // compilan al mismo paso: eso lo dice su `Transition` al ejecutar, no el plan.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("a").unwrap();
    g.add_node("b").unwrap();
    g.add_edge("a", "b").unwrap();
    c.insert("a", Arc::new(Sumar(1.0)));
    c.insert("b", Arc::new(Preguntar(vec![])));

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![ejecuta("a", &[]), ejecuta("b", &["a"])])
    );
}

// ── Abanicos: sin ninguna variante especial ──

#[test]
fn abrir_en_dos_ramas_es_que_las_dos_leen_del_mismo() {
    let (mut g, c) = con_filtros(&["fuente", "izq", "der"]);
    g.add_edge("fuente", "izq").unwrap();
    g.add_edge("fuente", "der").unwrap();

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![
            ejecuta("fuente", &[]),
            ejecuta("izq", &["fuente"]),
            ejecuta("der", &["fuente"]),
        ])
    );
}

#[test]
fn cerrar_dos_ramas_es_que_uno_lee_de_dos() {
    let (mut g, c) = con_filtros(&["izq", "der", "juntar"]);
    g.add_edge("izq", "juntar").unwrap();
    g.add_edge("der", "juntar").unwrap();

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![
            ejecuta("izq", &[]),
            ejecuta("der", &[]),
            ejecuta("juntar", &["izq", "der"]),
        ])
    );
}

#[test]
fn un_diamante_ejecuta_el_nodo_de_union_una_sola_vez() {
    let (mut g, c) = con_filtros(&["fuente", "izq", "der", "juntar"]);
    g.add_edge("fuente", "izq").unwrap();
    g.add_edge("fuente", "der").unwrap();
    g.add_edge("izq", "juntar").unwrap();
    g.add_edge("der", "juntar").unwrap();

    let Plan::Sequence(pasos) = compile(&g, &c).unwrap() else {
        panic!("un diamante compila a una secuencia");
    };
    assert_eq!(pasos.len(), 4, "cuatro nodos, cuatro pasos");
    assert_eq!(pasos[3], ejecuta("juntar", &["izq", "der"]));
}

#[test]
fn un_nodo_sin_implementacion_no_llega_a_ejecutarse() {
    let mut g = Graph::new();
    g.add_node("huerfano").unwrap();
    assert_eq!(
        compile(&g, &Catalog::new()).unwrap_err(),
        CompileError::NoImplementation("huerfano".into())
    );
}
