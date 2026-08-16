//! El motor, contra filtros y steps de Rust: sin Python de por medio.

use crate::dobles::{
    Cita, Diario, Gritar, Inmediato, Insaciable, Media, Preguntar, PreguntarEnCita, Punto,
    Reventar, Romper, SiempreNull, Sumar, Testigo,
};
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

// ── Waves: lo que pasa cuando dos ramas se lanzan a la vez ──

/// Monta el grafo, lo compila y lo ejecuta con nodos que se apuntan por dónde
/// pasan. Devuelve el diario y lo que salió.
fn corre_apuntando(
    nodos: &[&'static str],
    aristas: &[(&str, &str)],
) -> (Arc<Diario>, Result<Value, RunError>) {
    let diario = Diario::nuevo();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in nodos {
        g.add_node(*id).unwrap();
        c.insert(*id, Arc::new(Testigo(id, Arc::clone(&diario))));
    }
    for (from, to) in aristas {
        g.add_edge(*from, *to).unwrap();
    }
    let plan = compile(&g, &c).unwrap();
    let salida = Executor::new(&c).run(&plan, Value::Null);
    (diario, salida)
}

#[test]
fn el_orden_de_ejecucion_real_respeta_las_aristas() {
    // El plan dice un orden; esto comprueba el que **de verdad** ocurre, con
    // hilos de por medio. Un nodo no puede haberse ejecutado antes que
    // cualquiera de sus predecesores.
    let aristas = [
        ("a", "b"),
        ("b", "b2"),
        ("b2", "b3"),
        ("a", "c"),
        ("c", "c2"),
        ("b3", "d"),
        ("c2", "d"),
    ];
    let (diario, salida) = corre_apuntando(&["a", "b", "b2", "b3", "c", "c2", "d"], &aristas);
    salida.unwrap();

    let orden = diario.orden();
    assert_eq!(
        orden.len(),
        7,
        "se ejecutaron todos y una sola vez: {orden:?}"
    );
    let cuando = |quien: &str| orden.iter().position(|n| n == quien).unwrap();
    for (from, to) in aristas {
        assert!(
            cuando(from) < cuando(to),
            "{to} se ejecutó antes que {from}: {orden:?}"
        );
    }
}

#[test]
fn una_rama_entera_corre_en_el_mismo_hilo() {
    // Es lo que compra descomponer por ramas en vez de por nivel topológico:
    // la rama se fija a un hilo, y el día que un nodo tenga dispositivo
    // —que en torch es *thread-local*— se fija una vez y no en cada paso.
    let (diario, salida) = corre_apuntando(
        &["a", "b", "b2", "b3", "c", "c2", "d"],
        &[
            ("a", "b"),
            ("b", "b2"),
            ("b2", "b3"),
            ("a", "c"),
            ("c", "c2"),
            ("b3", "d"),
            ("c2", "d"),
        ],
    );
    salida.unwrap();

    assert_eq!(diario.hilo_de("b"), diario.hilo_de("b2"));
    assert_eq!(diario.hilo_de("b2"), diario.hilo_de("b3"));
    assert_eq!(diario.hilo_de("c"), diario.hilo_de("c2"));
    assert_ne!(
        diario.hilo_de("b"),
        diario.hilo_de("c"),
        "las dos ramas no pueden compartir hilo, o no van a la vez"
    );
    assert_eq!(
        diario.hilo_de("a"),
        diario.hilo_de("d"),
        "lo que está fuera de la wave corre en el hilo de quien ejecuta"
    );
}

#[test]
fn las_ramas_de_una_wave_corren_de_verdad_a_la_vez() {
    // Sin dormir ni un milisegundo: los dos nodos quedan en verse, y si se
    // ejecutaran uno detrás de otro el primero se quedaría esperando al
    // segundo hasta agotar el plazo.
    let punto = Punto::nuevo();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ["izq", "der"] {
        g.add_node(id).unwrap();
        c.insert(
            id,
            Arc::new(Cita {
                punto: Arc::clone(&punto),
                cuantos: 2,
                falla: None,
            }),
        );
    }
    let plan = compile(&g, &c).unwrap();
    assert!(
        matches!(plan, Plan::Wave(_)),
        "dos nodos sin relación son una wave"
    );

    Executor::new(&c).run(&plan, Value::Null).unwrap();
}

#[test]
fn tres_ramas_tambien_van_a_la_vez() {
    let punto = Punto::nuevo();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("fuente").unwrap();
    c.insert("fuente", Arc::new(Inmediato));
    for id in ["x", "y", "z"] {
        g.add_node(id).unwrap();
        g.add_edge("fuente", id).unwrap();
        c.insert(
            id,
            Arc::new(Cita {
                punto: Arc::clone(&punto),
                cuantos: 3,
                falla: None,
            }),
        );
    }
    let plan = compile(&g, &c).unwrap();
    Executor::new(&c).run(&plan, Value::Null).unwrap();
}

#[test]
fn el_diamante_da_el_mismo_resultado_con_wave_que_sin_ella() {
    // El resultado no puede depender de que las ramas se repartan o no.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("fuente", 1.0), ("izq", 10.0), ("der", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    g.add_node("juntar").unwrap();
    c.insert("juntar", Arc::new(Media));
    g.add_edge("fuente", "izq").unwrap();
    g.add_edge("fuente", "der").unwrap();
    g.add_edge("izq", "juntar").unwrap();
    g.add_edge("der", "juntar").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // 0 → 1 → {11, 101} → media 56
    assert_eq!(numero(&out), 56.0);
}

#[test]
fn lo_que_produce_cada_rama_llega_a_quien_la_lee() {
    // Las ramas trabajan sobre una copia de lo producido y se funden al
    // juntar; el nodo de unión tiene que ver lo de las dos, y también lo de
    // dentro de cada rama.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [
        ("a", 1.0),
        ("b", 10.0),
        ("b2", 20.0),
        ("c", 100.0),
        ("c2", 200.0),
    ] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    g.add_node("d").unwrap();
    c.insert("d", Arc::new(Media));
    for (from, to) in [
        ("a", "b"),
        ("b", "b2"),
        ("a", "c"),
        ("c", "c2"),
        ("b2", "d"),
        ("c2", "d"),
    ] {
        g.add_edge(from, to).unwrap();
    }

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // 0 → 1 → rama b: 11, 31 · rama c: 101, 301 → media 166
    assert_eq!(numero(&out), 166.0);
}

#[test]
fn dos_ramas_que_fallan_dan_siempre_el_error_de_la_primera() {
    // Las dos fallan de verdad a la vez —quedan en verse antes de romperse—,
    // así que cuál falla antes en el reloj es una carrera. El error que se
    // cuenta no puede depender de ella: es el de la primera rama declarada.
    let punto = Punto::nuevo();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, mensaje) in [("izq", "rompió la izquierda"), ("der", "rompió la derecha")] {
        g.add_node(id).unwrap();
        c.insert(
            id,
            Arc::new(Cita {
                punto: Arc::clone(&punto),
                cuantos: 2,
                falla: Some(mensaje),
            }),
        );
    }
    let plan = compile(&g, &c).unwrap();

    let RunError::Node { node, source } = Executor::new(&c).run(&plan, Value::Null).unwrap_err()
    else {
        panic!("esperaba el fallo de un nodo");
    };
    assert_eq!(node.as_str(), "izq");
    assert_eq!(source.message(), "rompió la izquierda");
}

#[test]
fn el_fallo_de_una_rama_no_impide_ver_el_de_la_otra_si_va_primera() {
    // El mismo grafo con las ramas al revés da el otro error: no es que gane
    // siempre "izq", es que gana el orden de declaración.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("der").unwrap();
    c.insert("der", Arc::new(Romper));
    g.add_node("izq").unwrap();
    c.insert("izq", Arc::new(Romper));

    let plan = compile(&g, &c).unwrap();
    let RunError::Node { node, .. } = Executor::new(&c).run(&plan, Value::Null).unwrap_err() else {
        panic!("esperaba el fallo de un nodo");
    };
    assert_eq!(node.as_str(), "der", "`der` se declaró primero");
}

#[test]
#[should_panic(expected = "reventé")]
fn un_panic_dentro_de_una_rama_no_se_traga() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("sano").unwrap();
    c.insert("sano", Arc::new(Inmediato));
    g.add_node("malo").unwrap();
    c.insert("malo", Arc::new(Reventar));

    let plan = compile(&g, &c).unwrap();
    let _ = Executor::new(&c).run(&plan, Value::Null);
}

#[test]
fn dos_ramas_pueden_tener_al_driver_ocupado_a_la_vez() {
    // Donde una wave gana sin discusión: dos nodos que esperan a algo de
    // fuera. El driver no atiende a la segunda hasta que ha llegado la
    // primera, así que si no fueran concurrentes se agotaría el plazo.
    let punto = Punto::nuevo();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ["uno", "otro"] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Preguntar(vec![Value::text("¿?")])));
    }
    let plan = compile(&g, &c).unwrap();

    let driver = PreguntarEnCita {
        punto: Arc::clone(&punto),
        cuantos: 2,
    };
    let out = Executor::new(&c)
        .with_driver(&driver)
        .run(&plan, Value::Null)
        .unwrap();

    assert_eq!(
        out,
        Value::map(vec![
            ("uno".to_string(), Value::text("atendido")),
            ("otro".to_string(), Value::text("atendido")),
        ])
    );
}

#[test]
fn una_wave_que_es_todo_el_plan_devuelve_el_mapa_de_sus_hojas() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("uno", 1.0), ("otro", 2.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(10.0)).unwrap();

    assert_eq!(
        out,
        Value::map(vec![
            ("uno".to_string(), Value::number(11.0)),
            ("otro".to_string(), Value::number(12.0)),
        ])
    );
}

#[test]
fn un_grafo_que_no_es_serie_paralelo_se_ejecuta_bien_aunque_no_se_reparta() {
    // La N: sin wave ninguna, pero el resultado es el correcto y `d` ve las
    // dos entradas.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, cuanto) in [("a", 1.0), ("b", 2.0), ("c", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Sumar(cuanto)));
    }
    g.add_node("d").unwrap();
    c.insert("d", Arc::new(Media));
    for (from, to) in [("a", "c"), ("a", "d"), ("b", "d")] {
        g.add_edge(from, to).unwrap();
    }

    let plan = compile(&g, &c).unwrap();
    assert!(
        !format!("{plan:?}").contains("Wave"),
        "la N no tiene árbol serie-paralelo"
    );
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // hojas: c = 1+100 = 101, d = media(a=1, b=2) = 1.5
    assert_eq!(
        out,
        Value::map(vec![
            ("c".to_string(), Value::number(101.0)),
            ("d".to_string(), Value::number(1.5)),
        ])
    );
}
