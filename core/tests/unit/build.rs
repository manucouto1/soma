//! El DSL: declarar el grafo como una expresión.

use crate::dobles::{Inmediato, Media, Sumar};
use soma_next_core::{Executor, GraphError, Value, build, compile, filter, step};

fn numero(v: &Value) -> f64 {
    let Value::Number(x) = v else {
        panic!("esperaba un número, había {}", v.type_name());
    };
    *x
}

#[test]
fn una_cadena() {
    let (g, c) = build(filter("a", Sumar(1.0)) >> filter("b", Sumar(10.0))).unwrap();

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
    let (g, c) = build(
        filter("fuente", Sumar(1.0))
            >> (filter("izq", Sumar(10.0)) | filter("der", Sumar(100.0)))
            >> filter("juntar", Media),
    )
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
    let (g, _) = build(
        filter("fuente", Sumar(1.0))
            >> ((filter("izq", Sumar(1.0)) >> filter("izq2", Sumar(1.0)))
                | filter("der", Sumar(1.0))),
    )
    .unwrap();

    assert_eq!(g.len(), 4);
    // fuente→izq, izq→izq2, fuente→der
    assert_eq!(g.edges().len(), 3);
}

#[test]
fn un_filtro_y_un_step_se_mezclan_en_la_misma_expresion() {
    let (g, c) = build(filter("sumar", Sumar(1.0)) >> step("eco", Inmediato)).unwrap();

    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        numero(&Executor::new(&c).run(&plan, Value::number(41.0)).unwrap()),
        42.0
    );
}

#[test]
fn un_id_repetido_se_cuenta_al_materializar() {
    let err = build(filter("a", Sumar(1.0)) >> filter("a", Sumar(1.0))).unwrap_err();
    assert_eq!(err, GraphError::DuplicateNode("a".into()));
}

#[test]
fn el_fallo_sobrevive_a_lo_que_le_pegues_despues() {
    let roto = filter("a", Sumar(1.0)) >> filter("a", Sumar(1.0));
    let err = build(roto >> filter("b", Sumar(1.0))).unwrap_err();
    assert_eq!(err, GraphError::DuplicateNode("a".into()));
}
