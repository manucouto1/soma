//! El cuestionario del CU1, contestado.

use soma_next_core::{Graph, GraphError, NodeId};

/// `a → b → c`, la tubería lineal de la que sale casi todo lo demás.
fn lineal() -> Graph {
    let mut g = Graph::new();
    for id in ["a", "b", "c"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();
    g
}

fn ids(nodes: Vec<&NodeId>) -> Vec<&str> {
    nodes.into_iter().map(NodeId::as_str).collect()
}

// ── Construcción ──

#[test]
fn un_grafo_vacio_es_valido() {
    let g = Graph::new();
    assert_eq!(g.len(), 0);
    assert!(g.is_empty());
    assert!(g.topological_sort().is_empty());
}

#[test]
fn un_grafo_de_un_solo_nodo_es_valido() {
    let mut g = Graph::new();
    g.add_node("solo").unwrap();
    assert_eq!(ids(g.roots()), ["solo"]);
    assert_eq!(ids(g.leaves()), ["solo"]);
}

#[test]
fn los_nodos_conservan_el_orden_de_insercion() {
    assert_eq!(
        lineal()
            .nodes()
            .iter()
            .map(NodeId::as_str)
            .collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
}

#[test]
fn una_tuberia_lineal_tiene_la_estructura_que_dice() {
    let g = lineal();
    assert_eq!(g.len(), 3);
    assert_eq!(g.edges().len(), 2);
    assert_eq!(ids(g.roots()), ["a"]);
    assert_eq!(ids(g.leaves()), ["c"]);
}

#[test]
fn free_id_sufija_hasta_encontrar_hueco() {
    let mut g = Graph::new();
    assert_eq!(g.free_id("limpiar").as_str(), "limpiar");
    g.add_node("limpiar").unwrap();
    assert_eq!(g.free_id("limpiar").as_str(), "limpiar_2");
    g.add_node("limpiar_2").unwrap();
    assert_eq!(g.free_id("limpiar").as_str(), "limpiar_3");
}

// ── Lo que no se puede construir ──

#[test]
fn dos_nodos_no_pueden_llamarse_igual() {
    let mut g = Graph::new();
    g.add_node("a").unwrap();
    assert_eq!(
        g.add_node("a").unwrap_err(),
        GraphError::DuplicateNode("a".into())
    );
    assert_eq!(g.len(), 1);
}

#[test]
fn una_arista_necesita_que_sus_dos_extremos_existan() {
    let mut g = Graph::new();
    g.add_node("a").unwrap();
    assert_eq!(
        g.add_edge("a", "fantasma").unwrap_err(),
        GraphError::UnknownNode("fantasma".into())
    );
    assert_eq!(
        g.add_edge("fantasma", "a").unwrap_err(),
        GraphError::UnknownNode("fantasma".into())
    );
    assert!(g.edges().is_empty());
}

#[test]
fn la_misma_arista_no_se_pone_dos_veces() {
    let mut g = lineal();
    assert_eq!(
        g.add_edge("a", "b").unwrap_err(),
        GraphError::DuplicateEdge {
            from: "a".into(),
            to: "b".into()
        }
    );
    assert_eq!(g.edges().len(), 2);
}

#[test]
fn un_ciclo_se_rechaza_al_ponerlo_no_al_recorrerlo() {
    let mut g = lineal();
    assert_eq!(
        g.add_edge("c", "a").unwrap_err(),
        GraphError::WouldCycle {
            from: "c".into(),
            to: "a".into()
        }
    );
    assert_eq!(g.edges().len(), 2);
}

#[test]
fn un_nodo_no_se_conecta_consigo_mismo() {
    let mut g = lineal();
    assert!(matches!(
        g.add_edge("a", "a").unwrap_err(),
        GraphError::WouldCycle { .. }
    ));
}

// ── Consultas de topología ──

#[test]
fn predecesores_y_sucesores() {
    let g = lineal();
    assert_eq!(ids(g.predecessors(&"b".into())), ["a"]);
    assert_eq!(ids(g.successors(&"b".into())), ["c"]);
    assert!(g.predecessors(&"a".into()).is_empty());
    assert!(g.successors(&"c".into()).is_empty());
}

#[test]
fn raices_y_hojas_con_ramas() {
    let mut g = Graph::new();
    for id in ["fuente_1", "fuente_2", "juntar", "salida"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("fuente_1", "juntar").unwrap();
    g.add_edge("fuente_2", "juntar").unwrap();
    g.add_edge("juntar", "salida").unwrap();

    assert_eq!(ids(g.roots()), ["fuente_1", "fuente_2"]);
    assert_eq!(ids(g.leaves()), ["salida"]);
    assert_eq!(
        ids(g.predecessors(&"juntar".into())),
        ["fuente_1", "fuente_2"]
    );
}

#[test]
fn orden_topologico_de_una_cadena() {
    assert_eq!(ids(lineal().topological_sort()), ["a", "b", "c"]);
}

#[test]
fn orden_topologico_con_ramas_paralelas() {
    let mut g = Graph::new();
    for id in ["entrada", "izquierda", "derecha", "juntar"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("entrada", "izquierda").unwrap();
    g.add_edge("entrada", "derecha").unwrap();
    g.add_edge("izquierda", "juntar").unwrap();
    g.add_edge("derecha", "juntar").unwrap();

    let orden = ids(g.topological_sort());
    assert_eq!(orden.len(), 4);
    assert_eq!(orden[0], "entrada");
    assert_eq!(orden[3], "juntar");
}

#[test]
fn el_orden_topologico_es_determinista() {
    let g = lineal();
    assert_eq!(ids(g.topological_sort()), ids(g.topological_sort()));
}

#[test]
fn nodos_sueltos_tambien_salen_en_el_orden() {
    let mut g = lineal();
    g.add_node("suelto").unwrap();
    assert_eq!(ids(g.topological_sort()), ["a", "b", "c", "suelto"]);
}
