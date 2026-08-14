//! El DSL: declarar el grafo como una expresión.

use crate::dobles::{Inmediato, Media, Sumar};
use soma_next_core::{Executor, GraphError, Value, compile, node};

fn numero(v: &Value) -> f64 {
    let Value::Number(x) = v else {
        panic!("esperaba un número, había {}", v.type_name());
    };
    *x
}

#[test]
fn una_cadena() {
    let (g, c) = (node("a", Sumar(1.0)) >> node("b", Sumar(10.0)))
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
    let (g, c) = (node("fuente", Sumar(1.0))
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
    let (g, _) = (node("fuente", Sumar(1.0))
        >> ((node("izq", Sumar(1.0)) >> node("izq2", Sumar(1.0))) | node("der", Sumar(1.0))))
    .somatize()
    .unwrap();

    assert_eq!(g.len(), 4);
    // fuente→izq, izq→izq2, fuente→der
    assert_eq!(g.edges().len(), 3);
}

#[test]
fn un_nodo_que_pide_turnos_y_otro_que_no_se_mezclan_igual() {
    let (g, c) = (node("sumar", Sumar(1.0)) >> node("eco", Inmediato))
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
