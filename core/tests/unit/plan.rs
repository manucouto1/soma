//! Compilar: de la estructura a la forma decidida.
//!
//! La mitad de este fichero comprueba **formas concretas** —qué árbol sale de
//! qué grafo— y la otra mitad **invariantes** que tienen que valer para
//! cualquier grafo: que ningún nodo se ejecute dos veces, que ninguno se
//! quede fuera, y que el orden que dicta el plan respete las aristas. Los
//! invariantes son los que habrían cazado el bug que mató a `Plan::Parallel`
//! en CU4, y por eso van sobre una batería de topologías y no sobre una.

use crate::dobles::{Preguntar, Sumar};
use soma_next_core::{Catalog, CompileError, Graph, NodeId, Plan, compile};
use std::collections::HashSet;
use std::sync::Arc;

/// Un grafo con estos nodos y estas aristas, todos con implementación.
fn grafo(nodos: &[&str], aristas: &[(&str, &str)]) -> (Graph, Catalog) {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in nodos {
        g.add_node(*id).unwrap();
        c.insert(*id, Arc::new(Sumar(1.0)));
    }
    for (from, to) in aristas {
        g.add_edge(*from, *to).unwrap();
    }
    (g, c)
}

fn con_filtros(ids: &[&str]) -> (Graph, Catalog) {
    grafo(ids, &[])
}

fn ejecuta(node: &str, from: &[&str]) -> Plan {
    Plan::Execute {
        node: node.into(),
        from: from.iter().map(|id| NodeId::from(*id)).collect(),
    }
}

fn plan_de(nodos: &[&str], aristas: &[(&str, &str)]) -> Plan {
    let (g, c) = grafo(nodos, aristas);
    compile(&g, &c).unwrap()
}

// ── Lo de siempre, que no cambia ──

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
    assert_eq!(
        plan_de(&["a", "b", "c"], &[("a", "b"), ("b", "c")]),
        Plan::Sequence(vec![
            ejecuta("a", &[]),
            ejecuta("b", &["a"]),
            ejecuta("c", &["b"]),
        ])
    );
}

#[test]
fn una_cadena_lineal_compila_a_lo_mismo_que_antes_de_las_waves() {
    // La regresión que más importa: todo lo cerrado de CU2 a CU8 son cadenas,
    // y su plan tiene que salir idéntico, sin una sola wave.
    let plan = plan_de(&["a", "b", "c", "d"], &[("a", "b"), ("b", "c"), ("c", "d")]);
    assert_eq!(
        plan,
        Plan::Sequence(vec![
            ejecuta("a", &[]),
            ejecuta("b", &["a"]),
            ejecuta("c", &["b"]),
            ejecuta("d", &["c"]),
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

#[test]
fn un_nodo_sin_implementacion_no_llega_a_ejecutarse() {
    let mut g = Graph::new();
    g.add_node("huerfano").unwrap();
    assert_eq!(
        compile(&g, &Catalog::new()).unwrap_err(),
        CompileError::NoImplementation("huerfano".into())
    );
}

// ── Las formas: qué árbol sale de qué grafo ──

#[test]
fn dos_nodos_sueltos_son_una_wave_de_dos_ramas() {
    // Sin ninguna arista, el grafo son dos componentes: `a | b`.
    assert_eq!(
        plan_de(&["a", "b"], &[]),
        Plan::Wave(vec![ejecuta("a", &[]), ejecuta("b", &[])])
    );
}

#[test]
fn abrir_en_dos_ramas_las_pone_en_una_wave() {
    assert_eq!(
        plan_de(
            &["fuente", "izq", "der"],
            &[("fuente", "izq"), ("fuente", "der")]
        ),
        Plan::Sequence(vec![
            ejecuta("fuente", &[]),
            Plan::Wave(vec![
                ejecuta("izq", &["fuente"]),
                ejecuta("der", &["fuente"]),
            ]),
        ])
    );
}

#[test]
fn cerrar_dos_ramas_es_que_uno_lee_de_dos() {
    assert_eq!(
        plan_de(
            &["izq", "der", "juntar"],
            &[("izq", "juntar"), ("der", "juntar")]
        ),
        Plan::Sequence(vec![
            Plan::Wave(vec![ejecuta("izq", &[]), ejecuta("der", &[])]),
            ejecuta("juntar", &["izq", "der"]),
        ])
    );
}

#[test]
fn un_diamante_ejecuta_el_nodo_de_union_una_sola_vez() {
    // El caso que rompió `Plan::Parallel` en CU4: sus ramas se solapaban y
    // `juntar` acababa en las dos. Aquí las ramas son componentes conexas, así
    // que `juntar` no puede estar en ninguna: se emite fuera de la wave.
    assert_eq!(
        plan_de(
            &["fuente", "izq", "der", "juntar"],
            &[
                ("fuente", "izq"),
                ("fuente", "der"),
                ("izq", "juntar"),
                ("der", "juntar"),
            ]
        ),
        Plan::Sequence(vec![
            ejecuta("fuente", &[]),
            Plan::Wave(vec![
                ejecuta("izq", &["fuente"]),
                ejecuta("der", &["fuente"]),
            ]),
            ejecuta("juntar", &["izq", "der"]),
        ])
    );
}

#[test]
fn una_rama_de_varios_nodos_es_una_sola_rama_de_la_wave() {
    // `a >> (b >> b2 >> b3 | c >> c2) >> d`. Es el caso que descarta agrupar
    // por nivel topológico: así, `b2` no espera a `c`, y la rama entera cabe
    // en un hilo.
    assert_eq!(
        plan_de(
            &["a", "b", "b2", "b3", "c", "c2", "d"],
            &[
                ("a", "b"),
                ("b", "b2"),
                ("b2", "b3"),
                ("a", "c"),
                ("c", "c2"),
                ("b3", "d"),
                ("c2", "d"),
            ]
        ),
        Plan::Sequence(vec![
            ejecuta("a", &[]),
            Plan::Wave(vec![
                Plan::Sequence(vec![
                    ejecuta("b", &["a"]),
                    ejecuta("b2", &["b"]),
                    ejecuta("b3", &["b2"]),
                ]),
                Plan::Sequence(vec![ejecuta("c", &["a"]), ejecuta("c2", &["c"])]),
            ]),
            ejecuta("d", &["b3", "c2"]),
        ])
    );
}

#[test]
fn el_corte_serie_no_parte_una_rama_por_la_mitad() {
    // `(a >> a2 | b) >> (c | d)`. Aquí no hay ningún nodo por el que pase
    // todo, así que buscar un "nodo barrera" fallaría y acabaría metiendo `a`
    // en una wave con `b` y dejando `a2` suelto. El corte serie mira los dos
    // extremos enteros, y por eso recupera la rama `a >> a2` de una pieza.
    assert_eq!(
        plan_de(
            &["a", "a2", "b", "c", "d"],
            &[
                ("a", "a2"),
                ("a2", "c"),
                ("a2", "d"),
                ("b", "c"),
                ("b", "d"),
            ]
        ),
        Plan::Sequence(vec![
            Plan::Wave(vec![
                Plan::Sequence(vec![ejecuta("a", &[]), ejecuta("a2", &["a"])]),
                ejecuta("b", &[]),
            ]),
            Plan::Wave(vec![ejecuta("c", &["a2", "b"]), ejecuta("d", &["a2", "b"]),]),
        ])
    );
}

#[test]
fn dos_waves_seguidas_no_se_funden_en_una() {
    // `(a | b) >> (c | d)`: cuatro nodos independientes dos a dos, pero `c` y
    // `d` dependen de los dos primeros. Son dos waves, no una de cuatro.
    assert_eq!(
        plan_de(
            &["a", "b", "c", "d"],
            &[("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
        ),
        Plan::Sequence(vec![
            Plan::Wave(vec![ejecuta("a", &[]), ejecuta("b", &[])]),
            Plan::Wave(vec![ejecuta("c", &["a", "b"]), ejecuta("d", &["a", "b"])]),
        ])
    );
}

#[test]
fn una_wave_puede_llevar_otra_dentro() {
    // `a >> ((b >> (c | d) >> e) | f) >> g`: la rama larga tiene su propio
    // abanico. El árbol anida tan hondo como la expresión.
    assert_eq!(
        plan_de(
            &["a", "b", "c", "d", "e", "f", "g"],
            &[
                ("a", "b"),
                ("b", "c"),
                ("b", "d"),
                ("c", "e"),
                ("d", "e"),
                ("a", "f"),
                ("e", "g"),
                ("f", "g"),
            ]
        ),
        Plan::Sequence(vec![
            ejecuta("a", &[]),
            Plan::Wave(vec![
                Plan::Sequence(vec![
                    ejecuta("b", &["a"]),
                    Plan::Wave(vec![ejecuta("c", &["b"]), ejecuta("d", &["b"])]),
                    ejecuta("e", &["c", "d"]),
                ]),
                ejecuta("f", &["a"]),
            ]),
            ejecuta("g", &["e", "f"]),
        ])
    );
}

#[test]
fn tres_ramas_dan_una_wave_de_tres() {
    let Plan::Sequence(pasos) = plan_de(
        &["fuente", "x", "y", "z"],
        &[("fuente", "x"), ("fuente", "y"), ("fuente", "z")],
    ) else {
        panic!("un abanico compila a una secuencia");
    };
    let Plan::Wave(ramas) = &pasos[1] else {
        panic!("el segundo paso es la wave");
    };
    assert_eq!(ramas.len(), 3);
}

#[test]
fn dos_grafos_sin_relacion_son_dos_ramas_aunque_cada_uno_sea_largo() {
    assert_eq!(
        plan_de(&["a", "a2", "b", "b2"], &[("a", "a2"), ("b", "b2")]),
        Plan::Wave(vec![
            Plan::Sequence(vec![ejecuta("a", &[]), ejecuta("a2", &["a"])]),
            Plan::Sequence(vec![ejecuta("b", &[]), ejecuta("b2", &["b"])]),
        ])
    );
}

// ── Lo que no es serie-paralelo ──

#[test]
fn la_n_no_tiene_arbol_y_se_recorre_en_secuencia() {
    // `a→c, a→d, b→d` es el patrón mínimo que no es serie-paralelo (Valdes,
    // Tarjan y Lawler, 1982). No hay corte ni componentes, así que se recorre
    // como antes de que existieran las waves: en fila, sin paralelismo.
    //
    // Y no se puede escribir con el DSL: `>>` y `|` solo generan grafos
    // serie-paralelos. Para llegar aquí hay que usar `node()`/`edge()`.
    assert_eq!(
        plan_de(&["a", "b", "c", "d"], &[("a", "c"), ("a", "d"), ("b", "d")]),
        Plan::Sequence(vec![
            ejecuta("a", &[]),
            ejecuta("b", &[]),
            ejecuta("c", &["a"]),
            ejecuta("d", &["a", "b"]),
        ])
    );
}

#[test]
fn una_n_no_estropea_el_paralelismo_de_lo_que_tiene_al_lado() {
    // La N cuelga de `raiz` y no toca a `otra`, que es una componente aparte:
    // la parte sana sigue siendo una rama de la wave.
    let Plan::Wave(ramas) = plan_de(
        &["a", "b", "c", "d", "otra"],
        &[("a", "c"), ("a", "d"), ("b", "d")],
    ) else {
        panic!("la N y el nodo suelto son dos componentes");
    };
    assert_eq!(ramas.len(), 2, "la N entera es una rama, `otra` es la otra");
    assert_eq!(ramas[1], ejecuta("otra", &[]));
}

// ── Invariantes: valen para cualquier grafo ──

/// Una topología de la batería. Lleva nombre para que el fallo diga cuál fue.
struct Topologia {
    nombre: &'static str,
    nodos: Vec<&'static str>,
    aristas: Vec<(&'static str, &'static str)>,
}

fn topologia(
    nombre: &'static str,
    nodos: Vec<&'static str>,
    aristas: Vec<(&'static str, &'static str)>,
) -> Topologia {
    Topologia {
        nombre,
        nodos,
        aristas,
    }
}

/// Todas las topologías interesantes, para pasarles los invariantes a todas.
fn bateria() -> Vec<Topologia> {
    vec![
        topologia("un nodo", vec!["a"], vec![]),
        topologia("cadena", vec!["a", "b", "c"], vec![("a", "b"), ("b", "c")]),
        topologia("sueltos", vec!["a", "b", "c"], vec![]),
        topologia(
            "diamante",
            vec!["f", "i", "d", "j"],
            vec![("f", "i"), ("f", "d"), ("i", "j"), ("d", "j")],
        ),
        topologia(
            "ramas largas",
            vec!["a", "b", "b2", "b3", "c", "c2", "d"],
            vec![
                ("a", "b"),
                ("b", "b2"),
                ("b2", "b3"),
                ("a", "c"),
                ("c", "c2"),
                ("b3", "d"),
                ("c2", "d"),
            ],
        ),
        topologia(
            "ramas desiguales sin junta unica",
            vec!["a", "a2", "b", "c", "d"],
            vec![
                ("a", "a2"),
                ("a2", "c"),
                ("a2", "d"),
                ("b", "c"),
                ("b", "d"),
            ],
        ),
        topologia(
            "wave anidada",
            vec!["a", "b", "c", "d", "e", "f", "g"],
            vec![
                ("a", "b"),
                ("b", "c"),
                ("b", "d"),
                ("c", "e"),
                ("d", "e"),
                ("a", "f"),
                ("e", "g"),
                ("f", "g"),
            ],
        ),
        topologia(
            "la N",
            vec!["a", "b", "c", "d"],
            vec![("a", "c"), ("a", "d"), ("b", "d")],
        ),
        topologia(
            "N con vecino sano",
            vec!["a", "b", "c", "d", "otra", "otra2"],
            vec![("a", "c"), ("a", "d"), ("b", "d"), ("otra", "otra2")],
        ),
        topologia(
            "abanico de tres que se vuelve a juntar",
            vec!["f", "x", "y", "z", "j"],
            vec![
                ("f", "x"),
                ("f", "y"),
                ("f", "z"),
                ("x", "j"),
                ("y", "j"),
                ("z", "j"),
            ],
        ),
    ]
}

/// Los pasos del plan, en el orden en que se ejecutarían.
///
/// Las ramas de una wave se aplanan una detrás de otra: como son
/// independientes, cualquier entrelazado vale, y ése es el más fácil de leer.
fn pasos(plan: &Plan) -> Vec<(NodeId, Vec<NodeId>)> {
    match plan {
        Plan::Empty => Vec::new(),
        Plan::Execute { node, from } => vec![(node.clone(), from.clone())],
        Plan::Sequence(plans) | Plan::Wave(plans) => plans.iter().flat_map(pasos).collect(),
    }
}

#[test]
fn ningun_nodo_se_ejecuta_dos_veces_ni_se_queda_fuera() {
    // El invariante que `Plan::Parallel` incumplía: en un diamante ejecutaba
    // el nodo de unión dos veces porque sus ramas se solapaban.
    for Topologia {
        nombre,
        nodos,
        aristas,
    } in bateria()
    {
        let (g, c) = grafo(&nodos, &aristas);
        let ejecutados: Vec<NodeId> = pasos(&compile(&g, &c).unwrap())
            .into_iter()
            .map(|(node, _)| node)
            .collect();

        let unicos: HashSet<&NodeId> = ejecutados.iter().collect();
        assert_eq!(
            unicos.len(),
            ejecutados.len(),
            "`{nombre}` ejecuta algún nodo dos veces: {ejecutados:?}"
        );
        assert_eq!(
            unicos.len(),
            g.len(),
            "`{nombre}` deja algún nodo del grafo sin ejecutar"
        );
    }
}

#[test]
fn el_orden_que_dicta_el_plan_respeta_las_aristas() {
    // Que un nodo no se ejecute antes que sus predecesores es lo que hace que
    // su entrada exista cuando la va a buscar.
    for Topologia {
        nombre,
        nodos,
        aristas,
    } in bateria()
    {
        let (g, c) = grafo(&nodos, &aristas);
        let orden: Vec<NodeId> = pasos(&compile(&g, &c).unwrap())
            .into_iter()
            .map(|(node, _)| node)
            .collect();

        for (i, node) in orden.iter().enumerate() {
            for pred in g.predecessors(node) {
                let antes = orden
                    .iter()
                    .position(|n| n == pred)
                    .expect("está en el plan");
                assert!(
                    antes < i,
                    "`{nombre}`: {node} se ejecuta antes que su predecesor {pred}"
                );
            }
        }
    }
}

#[test]
fn cada_paso_declara_exactamente_sus_predecesores_del_grafo() {
    for Topologia {
        nombre,
        nodos,
        aristas,
    } in bateria()
    {
        let (g, c) = grafo(&nodos, &aristas);
        for (node, from) in pasos(&compile(&g, &c).unwrap()) {
            let esperado: Vec<NodeId> = g.predecessors(&node).into_iter().cloned().collect();
            assert_eq!(from, esperado, "`{nombre}`: mal el `from` de {node}");
        }
    }
}

#[test]
fn las_ramas_de_una_wave_no_comparten_ningun_nodo() {
    // Es lo que hace que fundir lo que produjo cada rama no pueda pisar nada,
    // y sale gratis de que las ramas sean componentes conexas.
    fn revisa(plan: &Plan, nombre: &str) {
        match plan {
            Plan::Empty | Plan::Execute { .. } => {}
            Plan::Sequence(plans) => plans.iter().for_each(|p| revisa(p, nombre)),
            Plan::Wave(ramas) => {
                let mut vistos: HashSet<NodeId> = HashSet::new();
                for rama in ramas {
                    for (node, _) in pasos(rama) {
                        assert!(
                            vistos.insert(node.clone()),
                            "`{nombre}`: {node} está en dos ramas de la misma wave"
                        );
                    }
                    revisa(rama, nombre);
                }
            }
        }
    }
    for Topologia {
        nombre,
        nodos,
        aristas,
    } in bateria()
    {
        let (g, c) = grafo(&nodos, &aristas);
        revisa(&compile(&g, &c).unwrap(), nombre);
    }
}

#[test]
fn el_mismo_grafo_compila_siempre_igual() {
    // Sin esto, `plan()` no serviría para nada y la caché que venga tampoco.
    for Topologia {
        nombre,
        nodos,
        aristas,
    } in bateria()
    {
        let (g, c) = grafo(&nodos, &aristas);
        let primero = compile(&g, &c).unwrap();
        for _ in 0..5 {
            assert_eq!(
                compile(&g, &c).unwrap(),
                primero,
                "`{nombre}` no es estable"
            );
        }
    }
}
