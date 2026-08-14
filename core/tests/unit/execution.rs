//! El motor, contra filtros y steps de Rust: sin Python de por medio.

use crate::dobles::{Gritar, Inmediato, Insaciable, Preguntar, Romper, Sumar};
use soma_next_core::{Catalog, Executor, Graph, Plan, RunError, StepError, Value, compile};
use std::sync::Arc;

fn numero(v: &Value) -> f64 {
    let Value::Number(x) = v else {
        panic!("esperaba un número, había {}", v.type_name());
    };
    *x
}

// ── Filtros ──

#[test]
fn un_plan_vacio_devuelve_su_entrada() {
    let out = Executor::new(&Catalog::new())
        .run(&Plan::Empty, Value::text("intacto"))
        .unwrap();
    assert_eq!(out, Value::text("intacto"));
}

#[test]
fn una_cadena_encadena_las_salidas() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("a", 1.0), ("b", 10.0), ("c", 100.0)] {
        g.add_node(id).unwrap();
        c.insert_filter(id, Arc::new(Sumar(cuanto)));
    }
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    assert_eq!(numero(&out), 111.0);
}

#[test]
fn un_abanico_produce_una_lista_con_lo_de_cada_rama() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("fuente", 1.0), ("izq", 10.0), ("der", 100.0)] {
        g.add_node(id).unwrap();
        c.insert_filter(id, Arc::new(Sumar(cuanto)));
    }
    g.add_edge("fuente", "izq").unwrap();
    g.add_edge("fuente", "der").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();

    // fuente deja 1.0, y cada rama parte de ahí.
    assert_eq!(
        out,
        Value::list(vec![Value::number(11.0), Value::number(101.0)])
    );
}

#[test]
fn el_fallo_de_un_filtro_dice_en_que_nodo_fue() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("bomba").unwrap();
    c.insert_filter("bomba", Arc::new(Romper));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert!(matches!(err, RunError::Filter { ref node, .. } if node.as_str() == "bomba"));
    assert!(err.to_string().contains("me rompí"));
}

// ── Steps ──

#[test]
fn un_step_que_termina_a_la_primera_no_necesita_driver() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("ya").unwrap();
    c.insert_step("ya", Arc::new(Inmediato));

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::text("eco")).unwrap();
    assert_eq!(out, Value::text("eco"));
}

#[test]
fn un_step_pide_algo_y_el_driver_se_lo_da() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("pregunta").unwrap();
    c.insert_step("pregunta", Arc::new(Preguntar(vec![Value::text("hola")])));

    let plan = compile(&g, &c).unwrap();
    let gritar = Gritar;
    let out = Executor::new(&c)
        .with_driver(&gritar)
        .run(&plan, Value::Null)
        .unwrap();
    assert_eq!(out, Value::text("HOLA"));
}

#[test]
fn sin_driver_un_step_que_pide_falla_diciendolo() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("pregunta").unwrap();
    c.insert_step("pregunta", Arc::new(Preguntar(vec![Value::text("hola")])));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert_eq!(err, RunError::NoDriver("pregunta".into()));
    assert!(err.to_string().contains("no tiene driver"));
}

#[test]
fn el_fallo_del_driver_se_atribuye_al_step_que_pidio() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("pregunta").unwrap();
    // Gritar solo sabe con texto; se le pide con un número.
    c.insert_step("pregunta", Arc::new(Preguntar(vec![Value::number(1.0)])));

    let plan = compile(&g, &c).unwrap();
    let gritar = Gritar;
    let err = Executor::new(&c)
        .with_driver(&gritar)
        .run(&plan, Value::Null)
        .unwrap_err();
    assert!(matches!(err, RunError::Driver { ref node, .. } if node.as_str() == "pregunta"));
}

#[test]
fn un_step_que_no_sabe_parar_gasta_sus_turnos_y_lo_dice() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("nunca").unwrap();
    c.insert_step("nunca", Arc::new(Insaciable));

    let plan = compile(&g, &c).unwrap();
    let driver = SiempreNull;
    let err = Executor::new(&c)
        .with_driver(&driver)
        .run(&plan, Value::Null)
        .unwrap_err();
    assert!(matches!(err, RunError::TurnLimit { ref node, .. } if node.as_str() == "nunca"));
    assert!(err.to_string().contains("no sabe parar"));
}

struct SiempreNull;

impl soma_next_core::Driver for SiempreNull {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, soma_next_core::DriverError> {
        Ok(vec![Value::Null; requests.len()])
    }
}

// ── Filtros y steps en la misma cadena ──

#[test]
fn un_filtro_y_un_step_se_encadenan_sin_saber_el_uno_del_otro() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("sumar").unwrap();
    g.add_node("eco").unwrap();
    g.add_edge("sumar", "eco").unwrap();
    c.insert_filter("sumar", Arc::new(Sumar(1.0)));
    c.insert_step("eco", Arc::new(Inmediato));

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(41.0)).unwrap();
    assert_eq!(numero(&out), 42.0);
}

#[test]
fn un_step_puede_fallar_por_su_cuenta() {
    struct Rendirse;
    impl soma_next_core::Step for Rendirse {
        fn poll(
            &self,
            _ctx: &soma_next_core::StepCtx<'_>,
        ) -> Result<soma_next_core::Transition, StepError> {
            Err(StepError::new("no puedo"))
        }
    }

    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("rendirse").unwrap();
    c.insert_step("rendirse", Arc::new(Rendirse));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert!(matches!(err, RunError::Step { ref node, .. } if node.as_str() == "rendirse"));
}
