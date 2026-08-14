//! Compilar: de la estructura a la forma decidida.

use crate::dobles::{Inmediato, Sumar};
use soma_next_core::{Catalog, CompileError, Graph, Plan, compile};
use std::sync::Arc;

fn con_filtros(ids: &[&str]) -> (Graph, Catalog) {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ids {
        g.add_node(*id).unwrap();
        c.insert_filter(*id, Arc::new(Sumar(1.0)));
    }
    (g, c)
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
    assert_eq!(compile(&g, &c).unwrap(), Plan::Execute("a".into()));
}

#[test]
fn una_cadena_compila_a_una_secuencia_en_orden() {
    let (mut g, c) = con_filtros(&["a", "b", "c"]);
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![
            Plan::Execute("a".into()),
            Plan::Execute("b".into()),
            Plan::Execute("c".into()),
        ])
    );
}

#[test]
fn el_plan_distingue_un_step_de_un_filtro() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("filtro").unwrap();
    g.add_node("step").unwrap();
    g.add_edge("filtro", "step").unwrap();
    c.insert_filter("filtro", Arc::new(Sumar(1.0)));
    c.insert_step("step", Arc::new(Inmediato));

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![
            Plan::Execute("filtro".into()),
            Plan::Step("step".into())
        ])
    );
}

// ── Abanicos ──

#[test]
fn un_nodo_puede_alimentar_a_dos_ramas() {
    let (mut g, c) = con_filtros(&["fuente", "izq", "der"]);
    g.add_edge("fuente", "izq").unwrap();
    g.add_edge("fuente", "der").unwrap();

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![
            Plan::Execute("fuente".into()),
            Plan::Parallel(vec![
                Plan::Execute("izq".into()),
                Plan::Execute("der".into())
            ]),
        ])
    );
}

#[test]
fn cada_rama_sigue_por_su_cuenta() {
    let (mut g, c) = con_filtros(&["fuente", "izq", "izq2", "der"]);
    g.add_edge("fuente", "izq").unwrap();
    g.add_edge("fuente", "der").unwrap();
    g.add_edge("izq", "izq2").unwrap();

    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Sequence(vec![
            Plan::Execute("fuente".into()),
            Plan::Parallel(vec![
                Plan::Sequence(vec![
                    Plan::Execute("izq".into()),
                    Plan::Execute("izq2".into())
                ]),
                Plan::Execute("der".into()),
            ]),
        ])
    );
}

#[test]
fn dos_raices_sueltas_tambien_son_ramas() {
    let (g, c) = con_filtros(&["a", "b"]);
    assert_eq!(
        compile(&g, &c).unwrap(),
        Plan::Parallel(vec![Plan::Execute("a".into()), Plan::Execute("b".into())])
    );
}

// ── Lo estructural falla al compilar, no a mitad del recorrido ──

#[test]
fn juntar_dos_ramas_todavia_no_esta_decidido() {
    let (mut g, c) = con_filtros(&["izq", "der", "juntar"]);
    g.add_edge("izq", "juntar").unwrap();
    g.add_edge("der", "juntar").unwrap();

    let err = compile(&g, &c).unwrap_err();
    assert_eq!(
        err,
        CompileError::Fanin {
            node: "juntar".into(),
            sources: vec!["izq".into(), "der".into()]
        }
    );
    assert!(err.to_string().contains("cómo se combinan"));
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
