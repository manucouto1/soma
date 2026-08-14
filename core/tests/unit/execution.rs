//! El motor, contra filtros de Rust: sin Python de por medio.

use soma_next_core::{Catalog, Filter, FilterError, Graph, RunError, Value, run};
use std::sync::Arc;

/// Añade una constante a un escalar.
struct Sumar(f64);

impl Filter for Sumar {
    fn forward(&self, input: &Value) -> Result<Value, FilterError> {
        match input {
            Value::Tensor { values, .. } if values.len() == 1 => {
                Ok(Value::scalar(values[0] + self.0))
            }
            other => Err(FilterError::new(format!(
                "Sumar necesita un escalar, le dieron {}",
                other.type_name()
            ))),
        }
    }
}

/// Falla siempre.
struct Romper;

impl Filter for Romper {
    fn forward(&self, _input: &Value) -> Result<Value, FilterError> {
        Err(FilterError::new("me rompí"))
    }
}

fn escalar(v: &Value) -> f64 {
    let Value::Tensor { values, .. } = v else {
        panic!("esperaba un tensor, había {}", v.type_name());
    };
    values[0]
}

// ── El camino feliz ──

#[test]
fn un_grafo_vacio_devuelve_su_entrada() {
    let g = Graph::new();
    let out = run(&g, &Catalog::new(), Value::text("intacto")).unwrap();
    assert_eq!(out, Value::text("intacto"));
}

#[test]
fn un_solo_nodo_transforma_la_entrada() {
    let mut g = Graph::new();
    g.add_node("sumar").unwrap();
    let mut c = Catalog::new();
    c.insert("sumar", Arc::new(Sumar(1.0)));

    assert_eq!(escalar(&run(&g, &c, Value::scalar(41.0)).unwrap()), 42.0);
}

#[test]
fn una_cadena_encadena_las_salidas() {
    let mut g = Graph::new();
    for id in ["a", "b", "c"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();

    let mut c = Catalog::new();
    c.insert("a", Arc::new(Sumar(1.0)));
    c.insert("b", Arc::new(Sumar(10.0)));
    c.insert("c", Arc::new(Sumar(100.0)));

    assert_eq!(escalar(&run(&g, &c, Value::scalar(0.0)).unwrap()), 111.0);
}

#[test]
fn un_nodo_suelto_recibe_la_entrada_del_grafo() {
    // `suelto` no tiene predecesores, así que le llega la entrada original —
    // pero la hoja del grafo sigue siendo una sola.
    let mut g = Graph::new();
    g.add_node("a").unwrap();
    let mut c = Catalog::new();
    c.insert("a", Arc::new(Sumar(1.0)));

    assert_eq!(escalar(&run(&g, &c, Value::scalar(1.0)).unwrap()), 2.0);
}

// ── Lo que falla, y cómo lo cuenta ──

#[test]
fn un_nodo_sin_implementacion_es_un_error_con_nombre() {
    let mut g = Graph::new();
    g.add_node("huerfano").unwrap();
    let err = run(&g, &Catalog::new(), Value::Null).unwrap_err();

    assert_eq!(err, RunError::NoImplementation("huerfano".into()));
    assert!(err.to_string().contains("`huerfano`"));
}

#[test]
fn el_fallo_de_un_filtro_dice_en_que_nodo_fue() {
    let mut g = Graph::new();
    g.add_node("bomba").unwrap();
    let mut c = Catalog::new();
    c.insert("bomba", Arc::new(Romper));

    let err = run(&g, &c, Value::Null).unwrap_err();
    assert_eq!(
        err,
        RunError::Filter {
            node: "bomba".into(),
            source: FilterError::new("me rompí")
        }
    );
    assert!(err.to_string().contains("`bomba`"));
    assert!(err.to_string().contains("me rompí"));
}

#[test]
fn el_run_para_en_el_primer_fallo() {
    let mut g = Graph::new();
    for id in ["bomba", "despues"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("bomba", "despues").unwrap();
    let mut c = Catalog::new();
    c.insert("bomba", Arc::new(Romper));
    // `despues` no tiene implementación: si el run llegara hasta él, el error
    // sería otro.
    assert!(matches!(
        run(&g, &c, Value::Null).unwrap_err(),
        RunError::Filter { .. }
    ));
}

// ── Las dos decisiones que faltan, dichas en voz alta ──

#[test]
fn juntar_dos_ramas_todavia_no_esta_decidido() {
    let mut g = Graph::new();
    for id in ["izq", "der", "juntar"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("izq", "juntar").unwrap();
    g.add_edge("der", "juntar").unwrap();

    let mut c = Catalog::new();
    for id in ["izq", "der", "juntar"] {
        c.insert(id, Arc::new(Sumar(1.0)));
    }

    let err = run(&g, &c, Value::scalar(0.0)).unwrap_err();
    assert_eq!(
        err,
        RunError::Fanin {
            node: "juntar".into(),
            sources: vec!["izq".into(), "der".into()]
        }
    );
    assert!(err.to_string().contains("cómo se combinan"));
}

#[test]
fn dos_hojas_todavia_no_esta_decidido() {
    let mut g = Graph::new();
    for id in ["a", "b"] {
        g.add_node(id).unwrap();
    }
    let mut c = Catalog::new();
    for id in ["a", "b"] {
        c.insert(id, Arc::new(Sumar(1.0)));
    }

    let err = run(&g, &c, Value::scalar(0.0)).unwrap_err();
    assert_eq!(err, RunError::ManyLeaves(vec!["a".into(), "b".into()]));
    assert!(err.to_string().contains("cuál es la salida"));
}
