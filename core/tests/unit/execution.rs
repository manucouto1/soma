//! El motor, contra filtros y steps de Rust: sin Python de por medio.

use crate::dobles::{Gritar, Inmediato, Insaciable, Media, Preguntar, Romper, SiempreNull, Sumar};
use soma_next_core::{
    Catalog, Ctx, Executor, Graph, Node, NodeError, Plan, RunError, Transition, Value, compile,
};
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
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    assert_eq!(numero(&out), 111.0);
}

#[test]
fn varias_hojas_salen_como_un_mapa_con_su_nombre() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("fuente", 1.0), ("izq", 10.0), ("der", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    g.add_edge("fuente", "izq").unwrap();
    g.add_edge("fuente", "der").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();

    // fuente deja 1.0, y cada rama parte de ahí.
    assert_eq!(
        out,
        Value::map(vec![
            ("izq".to_string(), Value::number(11.0)),
            ("der".to_string(), Value::number(101.0)),
        ])
    );
    assert_eq!(out.get("der"), Some(&Value::number(101.0)));
}

#[test]
fn a_un_nodo_con_dos_entradas_le_llega_un_mapa() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("izq", 10.0), ("der", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    g.add_node("juntar").unwrap();
    c.insert("juntar", Arc::new(Media));
    g.add_edge("izq", "juntar").unwrap();
    g.add_edge("der", "juntar").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    assert_eq!(numero(&out), 55.0);
}

#[test]
fn un_diamante_da_la_vuelta() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("fuente", 1.0), ("izq", 10.0), ("der", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    g.add_node("juntar").unwrap();
    c.insert("juntar", Arc::new(Media));
    for (a, b) in [
        ("fuente", "izq"),
        ("fuente", "der"),
        ("izq", "juntar"),
        ("der", "juntar"),
    ] {
        g.add_edge(a, b).unwrap();
    }

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // fuente deja 1.0; ramas 11.0 y 101.0; media 56.0.
    assert_eq!(numero(&out), 56.0);
}

#[test]
fn el_fallo_de_un_filtro_dice_en_que_nodo_fue() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("bomba").unwrap();
    c.insert("bomba", Arc::new(Romper));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert!(matches!(err, RunError::Node { ref node, .. } if node.as_str() == "bomba"));
    assert!(err.to_string().contains("me rompí"));
}

// ── Steps ──

#[test]
fn un_step_que_termina_a_la_primera_no_necesita_driver() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("ya").unwrap();
    c.insert("ya", Arc::new(Inmediato));

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::text("eco")).unwrap();
    assert_eq!(out, Value::text("eco"));
}

#[test]
fn un_step_pide_algo_y_el_driver_se_lo_da() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("pregunta").unwrap();
    c.insert("pregunta", Arc::new(Preguntar(vec![Value::text("hola")])));

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
    c.insert("pregunta", Arc::new(Preguntar(vec![Value::text("hola")])));

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
    c.insert("pregunta", Arc::new(Preguntar(vec![Value::number(1.0)])));

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
    c.insert("nunca", Arc::new(Insaciable));

    let plan = compile(&g, &c).unwrap();
    let driver = SiempreNull;
    let err = Executor::new(&c)
        .with_driver(&driver)
        .run(&plan, Value::Null)
        .unwrap_err();
    assert!(matches!(err, RunError::TurnLimit { ref node, .. } if node.as_str() == "nunca"));
    assert!(err.to_string().contains("no sabe parar"));
}

// ── Filtros y steps en la misma cadena ──

#[test]
fn un_filtro_y_un_step_se_encadenan_sin_saber_el_uno_del_otro() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("sumar").unwrap();
    g.add_node("eco").unwrap();
    g.add_edge("sumar", "eco").unwrap();
    c.insert("sumar", Arc::new(Sumar(1.0)));
    c.insert("eco", Arc::new(Inmediato));

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(41.0)).unwrap();
    assert_eq!(numero(&out), 42.0);
}

#[test]
fn un_nodo_puede_fallar_a_mitad_de_sus_turnos() {
    struct Rendirse;
    impl Node for Rendirse {
        fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
            Err(NodeError::new("no puedo"))
        }
    }

    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("rendirse").unwrap();
    c.insert("rendirse", Arc::new(Rendirse));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert!(matches!(err, RunError::Node { ref node, .. } if node.as_str() == "rendirse"));
}

// ── Lo que la fusión hace posible ──

#[test]
fn un_nodo_puede_evolucionar_de_terminar_siempre_a_pedir_un_turno() {
    // Con dos traits esto obligaba a reescribir el tipo (error[E0119] si se
    // intentaba tener los dos). Aquí es una rama más en el mismo cuerpo.
    struct Evoluciona;
    impl Node for Evoluciona {
        fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
            if ctx.turn > 0 {
                // Ya preguntamos: la respuesta es lo que trajo el driver.
                return Ok(Transition::Done(ctx.results[0].clone()));
            }
            match input {
                Value::Number(x) if *x < 0.0 => {
                    Ok(Transition::Await(vec![Value::text("negativo")]))
                }
                otro => Ok(Transition::Done(otro.clone())),
            }
        }
    }

    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("evoluciona").unwrap();
    c.insert("evoluciona", Arc::new(Evoluciona));
    let plan = compile(&g, &c).unwrap();

    // Con entrada positiva no pide nada, así que ni necesita driver.
    let out = Executor::new(&c).run(&plan, Value::number(1.0)).unwrap();
    assert_eq!(out, Value::number(1.0));

    // Con entrada negativa pide un turno, en el mismo nodo.
    let gritar = Gritar;
    let out = Executor::new(&c)
        .with_driver(&gritar)
        .run(&plan, Value::number(-1.0))
        .unwrap();
    assert_eq!(out, Value::text("NEGATIVO"));
}

#[test]
fn pure_envuelve_una_funcion_sin_necesitar_un_segundo_trait() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("doblar").unwrap();
    c.insert(
        "doblar",
        Arc::new(soma_next_core::Pure(|v: &Value| match v {
            Value::Number(x) => Ok(Value::number(x * 2.0)),
            otro => Err(NodeError::new(format!(
                "esperaba un número, había {}",
                otro.type_name()
            ))),
        })),
    );

    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        numero(&Executor::new(&c).run(&plan, Value::number(21.0)).unwrap()),
        42.0
    );
}
