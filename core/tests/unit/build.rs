//! El DSL: declarar el grafo como una expresión.

use crate::dobles::{Inmediato, Media, Sumar};
use soma_next_core::{Executor, GraphError, NodeId, Plan, Value, compile, node};

fn numero(v: &Value) -> f64 {
    let Value::Number(x) = v else {
        panic!("esperaba un número, había {}", v.type_name());
    };
    *x
}

#[test]
fn una_cadena() {
    let (g, c, _) = (node("a", Sumar(1.0)) >> node("b", Sumar(10.0)))
        .somatize()
        .unwrap();

    assert_eq!(g.len(), 2);
    assert_eq!(g.edges().len(), 1);
    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        numero(&Executor::new(&c).run(&plan, Value::number(0.0)).unwrap()),
        11.0
    );
}

#[test]
fn un_diamante_se_lee_de_un_vistazo() {
    let (g, c, _) = (node("fuente", Sumar(1.0))
        >> (node("izq", Sumar(10.0)) | node("der", Sumar(100.0)))
        >> node("juntar", Media))
    .somatize()
    .unwrap();

    assert_eq!(g.len(), 4);
    assert_eq!(g.edges().len(), 4);
    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        numero(&Executor::new(&c).run(&plan, Value::number(0.0)).unwrap()),
        56.0
    );
}

#[test]
fn las_ramas_pueden_tener_su_propia_longitud() {
    let (g, _, _) = (node("fuente", Sumar(1.0))
        >> ((node("izq", Sumar(1.0)) >> node("izq2", Sumar(1.0))) | node("der", Sumar(1.0))))
    .somatize()
    .unwrap();

    assert_eq!(g.len(), 4);
    // fuente→izq, izq→izq2, fuente→der
    assert_eq!(g.edges().len(), 3);
}

#[test]
fn un_nodo_que_pide_turnos_y_otro_que_no_se_mezclan_igual() {
    let (g, c, _) = (node("sumar", Sumar(1.0)) >> node("eco", Inmediato))
        .somatize()
        .unwrap();

    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        numero(&Executor::new(&c).run(&plan, Value::number(41.0)).unwrap()),
        42.0
    );
}

#[test]
fn un_id_repetido_se_cuenta_al_materializar() {
    let err = (node("a", Sumar(1.0)) >> node("a", Sumar(1.0)))
        .somatize()
        .unwrap_err();
    assert_eq!(err, GraphError::DuplicateNode("a".into()));
}

#[test]
fn el_fallo_sobrevive_a_lo_que_le_pegues_despues() {
    let roto = node("a", Sumar(1.0)) >> node("a", Sumar(1.0));
    let err = (roto >> node("b", Sumar(1.0))).somatize().unwrap_err();
    assert_eq!(err, GraphError::DuplicateNode("a".into()));
}

// ── El oráculo: el plan reproduce el árbol de la expresión ──
//
// `>>` y `|` son composición en serie y en paralelo, así que la expresión que
// escribes **es** un árbol. `compile` no lo recibe —el grafo solo tiene nodos
// y aristas, y tiene que dar lo mismo construido en un bucle con
// `node()`/`edge()`, que fue la decisión 6 de CU5— así que lo **recupera**.
// Estos tests comprueban que lo recupera entero: son el oráculo del que sale
// toda la descomposición.

fn ejecuta(id: &str, from: &[&str]) -> Plan {
    Plan::Execute {
        node: id.into(),
        from: from.iter().map(|f| NodeId::from(*f)).collect(),
    }
}

/// El plan de una expresión, para compararlo con su árbol.
macro_rules! plan_de {
    ($wire:expr) => {{
        let (g, c, _) = ($wire).somatize().unwrap();
        compile(&g, &c).unwrap()
    }};
}

#[test]
fn una_cadena_es_una_secuencia_y_nada_mas() {
    assert_eq!(
        plan_de!(node("a", Sumar(1.0)) >> node("b", Sumar(1.0)) >> node("c", Sumar(1.0))),
        Plan::Sequence(vec![
            ejecuta("a", &[]),
            ejecuta("b", &["a"]),
            ejecuta("c", &["b"]),
        ])
    );
}

#[test]
fn un_or_suelto_es_una_wave() {
    assert_eq!(
        plan_de!(node("a", Sumar(1.0)) | node("b", Sumar(1.0))),
        Plan::Wave(vec![ejecuta("a", &[]), ejecuta("b", &[])])
    );
}

#[test]
fn el_diamante_del_dsl_sale_tal_cual_se_escribe() {
    assert_eq!(
        plan_de!(
            node("f", Sumar(1.0))
                >> (node("i", Sumar(10.0)) | node("d", Sumar(100.0)))
                >> node("j", Media)
        ),
        Plan::Sequence(vec![
            ejecuta("f", &[]),
            Plan::Wave(vec![ejecuta("i", &["f"]), ejecuta("d", &["f"])]),
            ejecuta("j", &["i", "d"]),
        ])
    );
}

#[test]
fn los_parentesis_de_una_rama_larga_sobreviven_al_grafo() {
    // `a >> (b >> b2 >> b3 | c >> c2) >> d`, que es el caso que obliga a que
    // una rama de la wave sea un plan entero y no un paso suelto.
    assert_eq!(
        plan_de!(
            node("a", Sumar(1.0))
                >> ((node("b", Sumar(1.0)) >> node("b2", Sumar(1.0)) >> node("b3", Sumar(1.0)))
                    | (node("c", Sumar(1.0)) >> node("c2", Sumar(1.0))))
                >> node("d", Media)
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
fn dos_ramas_abiertas_seguidas_de_otras_dos() {
    // `(a >> a2 | b) >> (c | d)`: el contraejemplo que descarta cortar por un
    // "nodo barrera". La rama `a >> a2` tiene que salir de una pieza.
    assert_eq!(
        plan_de!(
            ((node("a", Sumar(1.0)) >> node("a2", Sumar(1.0))) | node("b", Sumar(1.0)))
                >> (node("c", Media) | node("d", Media))
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
fn una_wave_dentro_de_una_rama_de_otra_wave() {
    // `a >> ((b >> (c | d) >> e) | f) >> g`: el árbol anida tan hondo como los
    // paréntesis.
    assert_eq!(
        plan_de!(
            node("a", Sumar(1.0))
                >> ((node("b", Sumar(1.0))
                    >> (node("c", Sumar(1.0)) | node("d", Sumar(1.0)))
                    >> node("e", Media))
                    | node("f", Sumar(1.0)))
                >> node("g", Media)
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
fn tres_ramas_de_longitudes_distintas() {
    assert_eq!(
        plan_de!(
            node("f", Sumar(1.0))
                >> (node("x", Sumar(1.0))
                    | (node("y", Sumar(1.0)) >> node("y2", Sumar(1.0)))
                    | (node("z", Sumar(1.0)) >> node("z2", Sumar(1.0)) >> node("z3", Sumar(1.0))))
                >> node("j", Media)
        ),
        Plan::Sequence(vec![
            ejecuta("f", &[]),
            Plan::Wave(vec![
                ejecuta("x", &["f"]),
                Plan::Sequence(vec![ejecuta("y", &["f"]), ejecuta("y2", &["y"])]),
                Plan::Sequence(vec![
                    ejecuta("z", &["f"]),
                    ejecuta("z2", &["z"]),
                    ejecuta("z3", &["z2"]),
                ]),
            ]),
            ejecuta("j", &["x", "y2", "z3"]),
        ])
    );
}

#[test]
fn una_rama_que_se_abre_y_se_cierra_dentro_de_si_misma() {
    // La rama izquierda es un diamante completo; la derecha, un nodo.
    assert_eq!(
        plan_de!(
            ((node("p", Sumar(1.0))
                >> (node("q", Sumar(1.0)) | node("r", Sumar(1.0)))
                >> node("s", Media))
                | node("t", Sumar(1.0)))
        ),
        Plan::Wave(vec![
            Plan::Sequence(vec![
                ejecuta("p", &[]),
                Plan::Wave(vec![ejecuta("q", &["p"]), ejecuta("r", &["p"])]),
                ejecuta("s", &["q", "r"]),
            ]),
            ejecuta("t", &[]),
        ])
    );
}
