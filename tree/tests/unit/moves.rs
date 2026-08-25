//! El razonamiento, contra un store en un directorio temporal.
//!
//! Un `Local` de verdad y no un doble: lo que se defiende aquí es que dos
//! personas escribiendo a la vez obtienen dos movimientos, y eso es una
//! afirmación sobre cómo se comporta un store bajo dos escritores.

use soma_next_store::Local;
use soma_tree::moves::{Cited, Course, Kind, Move, Moves, Said, Says, Scope, Standing};

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("un directorio temporal");
    let kept = Local::at(at.path()).expect("un store dentro");
    (at, kept)
}

/// Añade un movimiento con alcance de todo y sin citas, que es el caso normal.
fn plain(moves: &Moves, kind: Kind, prose: &str) -> u32 {
    moves
        .add(kind, prose, "yo", Scope::everything(), Vec::new(), None)
        .expect("un movimiento")
}

// ── Las cinco clases y sus verbos ──

#[test]
fn una_pregunta_sin_intentar_es_un_movimiento_como_los_demas() {
    // La única clase que puede existir sin nada debajo. Hoy una pregunta que
    // nadie ha atacado no tiene dónde vivir, y eso es trabajo pendiente que se
    // pierde.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);

    let q = plain(&moves, Kind::Question, "¿es el encoder el cuello?");

    assert_eq!(moves.all().unwrap()[&q].kind, Kind::Question);
    assert_eq!(moves.standing().unwrap()[&q], Standing::Open);
}

#[test]
fn un_valida_apuntando_a_un_intento_no_significa_nada_y_se_rechaza() {
    // Aceptarlo sería guardar una frase que nadie puede leer después.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "el recall no se mueve");
    let a = plain(&moves, Kind::Attempt, "probé con escala 2.0");

    let said = moves.say(Said {
        from: f,
        to: a,
        says: Says::Validates,
        scope: Scope::everything(),
        in_part: false,
    });

    assert!(said.is_err(), "{said:?}");
}

#[test]
fn responder_y_validar_no_son_el_mismo_verbo() {
    // Plegar hipótesis dentro de pregunta borraba justo esto: a una se le
    // responde, a la otra se la valida o se la refuta.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿por qué cae el recall?");
    let h = plain(&moves, Kind::Hypothesis, "el tokenizador parte mal");
    let f = plain(&moves, Kind::Finding, "con otro tokenizador es igual");

    assert!(
        moves
            .say(Said {
                from: f,
                to: h,
                says: Says::Answers,
                scope: Scope::everything(),
                in_part: false
            })
            .is_err(),
        "a una hipótesis no se le responde",
    );
    assert!(
        moves
            .say(Said {
                from: f,
                to: q,
                says: Says::Refutes,
                scope: Scope::everything(),
                in_part: false
            })
            .is_err(),
        "una pregunta no se refuta",
    );
}

// ── Respuestas parciales ──

#[test]
fn tres_respuestas_en_parte_empujan_una_pregunta_sin_cerrarla() {
    // «¿Si aumento la capacidad, mejora?» no se responde de una vez: se generan
    // tres intentos y cada uno responde en parte. Ni abierta ni cerrada.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿si aumento la capacidad, mejora?");

    for said in ["×2 sí", "×4 sí", "×8 se estanca"] {
        let f = plain(&moves, Kind::Finding, said);
        moves
            .say(Said {
                from: f,
                to: q,
                says: Says::Answers,
                scope: Scope::everything(),
                in_part: true,
            })
            .expect("una respuesta parcial");
    }

    assert_eq!(moves.standing().unwrap()[&q], Standing::Partly);
}

#[test]
fn una_que_cierra_basta_para_darla_por_respondida() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿mejora?");
    let a = plain(&moves, Kind::Finding, "en parte");
    let b = plain(&moves, Kind::Finding, "del todo, y por esto");

    for (from, in_part) in [(a, true), (b, false)] {
        moves
            .say(Said {
                from,
                to: q,
                says: Says::Answers,
                scope: Scope::everything(),
                in_part,
            })
            .expect("una respuesta");
    }

    assert_eq!(moves.standing().unwrap()[&q], Standing::Answered);
}

// ── El alcance, y por qué la disputa se mide por solape ──

#[test]
fn validar_y_refutar_sobre_situaciones_distintas_no_es_una_contradiccion() {
    // El caso de la combinación: A sola funcionaba, A+B se anulan. Dos hechos
    // sobre dos situaciones. Contarlos como conflicto sería llamar contradicción
    // a lo que más se aprende de una investigación.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "más capacidad mejora");
    let a = plain(&moves, Kind::Attempt, "variante A");
    let ab = plain(&moves, Kind::Attempt, "variante A + B");
    let sola = plain(&moves, Kind::Finding, "A sola: mejora");
    let juntas = plain(&moves, Kind::Finding, "A+B: se anulan");

    moves
        .say(Said {
            from: sola,
            to: h,
            says: Says::Validates,
            scope: Scope::of([a]),
            in_part: false,
        })
        .unwrap();
    moves
        .say(Said {
            from: juntas,
            to: h,
            says: Says::Refutes,
            scope: Scope::of([ab]),
            in_part: false,
        })
        .unwrap();

    assert_eq!(
        moves.standing().unwrap()[&h],
        Standing::Depends,
        "no es disputa ni media validación: la respuesta depende de dónde se mire",
    );
}

#[test]
fn depende_no_es_lo_mismo_que_en_parte() {
    // Salió al correr el caso entero: los dos usaban la misma palabra y
    // significan cosas distintas. Una pregunta a medio responder está empujada;
    // una hipótesis que vale aquí y no allí tiene una respuesta condicional, que
    // es el desenlace más informativo que da una investigación.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "mejora");
    let a = plain(&moves, Kind::Attempt, "A");
    let f = plain(&moves, Kind::Finding, "en A, y sólo a medias");

    moves
        .say(Said {
            from: f,
            to: h,
            says: Says::Validates,
            scope: Scope::of([a]),
            in_part: true,
        })
        .unwrap();

    assert_eq!(
        moves.standing().unwrap()[&h],
        Standing::PartlyValidated,
        "un solo signo a medias es media validación, no condicional",
    );
}

#[test]
fn y_sobre_la_misma_situacion_si_lo_es() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "más capacidad mejora");
    let a = plain(&moves, Kind::Attempt, "variante A");
    let (si, no) = (
        plain(&moves, Kind::Finding, "mejora"),
        plain(&moves, Kind::Finding, "no mejora"),
    );

    for (from, says) in [(si, Says::Validates), (no, Says::Refutes)] {
        moves
            .say(Said {
                from,
                to: h,
                says,
                scope: Scope::of([a]),
                in_part: false,
            })
            .unwrap();
    }

    assert_eq!(moves.standing().unwrap()[&h], Standing::Disputed);
}

#[test]
fn un_alcance_de_todo_toca_cualquier_otro() {
    // «Esto es falso en general» sí contradice a «esto vale para A».
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "más capacidad mejora");
    let a = plain(&moves, Kind::Attempt, "variante A");
    let (si, no) = (
        plain(&moves, Kind::Finding, "en A mejora"),
        plain(&moves, Kind::Finding, "en ningún sitio mejora"),
    );

    moves
        .say(Said {
            from: si,
            to: h,
            says: Says::Validates,
            scope: Scope::of([a]),
            in_part: false,
        })
        .unwrap();
    moves
        .say(Said {
            from: no,
            to: h,
            says: Says::Refutes,
            scope: Scope::everything(),
            in_part: false,
        })
        .unwrap();

    assert_eq!(moves.standing().unwrap()[&h], Standing::Disputed);
}

#[test]
fn un_alcance_arrastra_lo_que_cuelga_de_su_raiz() {
    // «Toda la rama del encoder» es una raíz, no una enumeración. Es lo que
    // hace pagable la pregunta de si dos alcances se tocan.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let rama = plain(&moves, Kind::Attempt, "la rama del encoder");
    let dentro = plain(&moves, Kind::Attempt, "un paso de esa rama");
    moves.hang(dentro, rama).unwrap();

    let h = plain(&moves, Kind::Hypothesis, "el encoder es el problema");
    let (si, no) = (
        plain(&moves, Kind::Finding, "en la rama entera"),
        plain(&moves, Kind::Finding, "en ese paso concreto"),
    );
    moves
        .say(Said {
            from: si,
            to: h,
            says: Says::Validates,
            scope: Scope::of([rama]),
            in_part: false,
        })
        .unwrap();
    moves
        .say(Said {
            from: no,
            to: h,
            says: Says::Refutes,
            scope: Scope::of([dentro]),
            in_part: false,
        })
        .unwrap();

    assert_eq!(
        moves.standing().unwrap()[&h],
        Standing::Disputed,
        "el paso está dentro de la rama, así que los alcances se tocan",
    );
}

// ── El DAG ──

#[test]
fn un_movimiento_cuelga_de_dos_preguntas_a_la_vez() {
    // El caso que obliga al DAG: la combinación es sobre la interacción de dos
    // respuestas y no cabe bajo ninguna de las dos preguntas.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q1 = plain(&moves, Kind::Question, "¿mejora la interpretabilidad?");
    let q2 = plain(&moves, Kind::Question, "¿mejora el rendimiento?");
    let ab = plain(&moves, Kind::Attempt, "A + B");

    moves.hang(ab, q1).unwrap();
    moves.hang(ab, q2).unwrap();

    let mut of = moves.under().unwrap().parents_of(ab);
    of.sort();
    assert_eq!(of, vec![q1, q2]);
}

#[test]
fn combina_es_una_arista_de_intento_a_intento_y_no_es_colgar() {
    // Dice que este intento **es** la composición de aquellos, que es lo que
    // permite leer «cada una funcionaba sola, juntas se anulan».
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let (a, b) = (
        plain(&moves, Kind::Attempt, "variante A"),
        plain(&moves, Kind::Attempt, "variante B"),
    );
    let ab = plain(&moves, Kind::Attempt, "A + B");

    for one in [a, b] {
        moves
            .say(Said {
                from: ab,
                to: one,
                says: Says::Combines,
                scope: Scope::everything(),
                in_part: false,
            })
            .expect("una combinación");
    }

    let says = moves.says().unwrap();
    assert_eq!(
        says.iter().filter(|one| one.says == Says::Combines).count(),
        2
    );
    assert!(
        moves.under().unwrap().parents_of(ab).is_empty(),
        "combinar no es colgar",
    );
}

#[test]
fn un_ciclo_se_rechaza_al_escribirlo_y_no_al_recorrerlo() {
    // Con `under` multivaluado ya no basta con confiar en la forma, y un ciclo
    // cuelga cualquier recorrido posterior — incluido el que lo dibujaría.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let (a, b, c) = (
        plain(&moves, Kind::Question, "a"),
        plain(&moves, Kind::Question, "b"),
        plain(&moves, Kind::Question, "c"),
    );
    moves.hang(b, a).unwrap();
    moves.hang(c, b).unwrap();

    assert!(moves.hang(a, c).is_err(), "a → b → c → a");
    assert!(moves.hang(a, a).is_err(), "ni consigo mismo");
}

// ── Escribir ──

#[test]
fn un_intento_cita_la_capa_uno() {
    // La única clase que la toca, y lo que ata el razonamiento a algo que se
    // puede volver a ejecutar.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);

    let a = moves
        .add(
            Kind::Attempt,
            "tres escalas",
            "yo",
            Scope::everything(),
            vec![
                Cited {
                    what: "commit".into(),
                    id: "4910005c".into(),
                },
                Cited {
                    what: "trial".into(),
                    id: "exp/t/4910005c/trial/0/0".into(),
                },
            ],
            None,
        )
        .unwrap();

    let body: Move = moves.all().unwrap().remove(&a).unwrap();
    assert_eq!(body.cites.len(), 2);
    assert_eq!(body.cites[0].id, "4910005c");
}

#[test]
fn reescribir_la_prosa_no_borra_lo_anterior() {
    // Como el diario: gana la última, y la de antes sigue ahí.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "no mejora");

    moves
        .reword(f, Some("no mejora entre 1.0 y 3.0"), None, None, "yo")
        .unwrap();

    assert_eq!(moves.all().unwrap()[&f].prose, "no mejora entre 1.0 y 3.0");
}

#[test]
fn nadie_escribiendo_a_la_vez_pierde_su_movimiento() {
    // La propiedad para la que existe `claim`, y la razón de que esto no sea
    // una fila que alguien actualiza: dos máquinas sobre un NFS se oyen las dos.
    let (_at, kept) = somewhere();
    let kept = &kept;

    std::thread::scope(|scope| {
        for which in 0..8 {
            scope.spawn(move || {
                Moves::of("t", kept)
                    .add(
                        Kind::Finding,
                        &format!("vi {which}"),
                        "yo",
                        Scope::everything(),
                        Vec::new(),
                        None,
                    )
                    .unwrap();
            });
        }
    });

    assert_eq!(Moves::of("t", kept).all().unwrap().len(), 8);
}

#[test]
fn dos_investigaciones_en_un_store_no_se_ven() {
    let (_at, kept) = somewhere();
    plain(&Moves::of("una", &kept), Kind::Question, "mía");

    assert!(Moves::of("otra", &kept).all().unwrap().is_empty());
}

#[test]
fn decir_lo_mismo_otra_vez_corrige_el_alcance_en_vez_de_duplicarlo() {
    // Cambiar de opinión sobre **dónde** vale un hallazgo es el caso corriente:
    // se creyó general y resultó ser de una rama. Si las dos aristas
    // sobrevivieran, ampliar un alcance se contaría como decirlo dos veces.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "más capacidad mejora");
    let a = plain(&moves, Kind::Attempt, "variante A");
    let f = plain(&moves, Kind::Finding, "mejora");

    let mut said = Said {
        from: f,
        to: h,
        says: Says::Validates,
        scope: Scope::of(vec![a]),
        in_part: true,
    };
    moves.say(said.clone()).unwrap();
    said.scope = Scope::everything();
    said.in_part = false;
    moves.say(said).unwrap();

    let says = moves.says().unwrap();
    assert_eq!(says.len(), 1, "la de antes sigue guardada, pero no cuenta");
    assert!(says[0].scope.is_everything());
    assert!(!says[0].in_part);
}

#[test]
fn corregir_el_alcance_no_toca_lo_que_se_dijo_con_otro_verbo() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let h = plain(&moves, Kind::Hypothesis, "más capacidad mejora");
    let f = plain(&moves, Kind::Finding, "según dónde");

    for says in [Says::Validates, Says::Refutes, Says::Validates] {
        moves
            .say(Said {
                from: f,
                to: h,
                says,
                scope: Scope::everything(),
                in_part: true,
            })
            .unwrap();
    }

    assert_eq!(
        moves.says().unwrap().len(),
        2,
        "valida y refuta, una de cada"
    );
}

// ── Lo decidido, que se deriva y no se guarda ──

/// Un intento que cita un commit, colgado de donde se le diga.
fn tried(moves: &Moves, prose: &str, commit: &str, under: &[u32]) -> u32 {
    let id = moves
        .add(
            Kind::Attempt,
            prose,
            "yo",
            Scope::everything(),
            vec![Cited {
                what: "commit".into(),
                id: commit.into(),
            }],
            None,
        )
        .unwrap();
    for parent in under {
        moves.hang(id, *parent).unwrap();
    }
    id
}

#[test]
fn abandonar_una_linea_alcanza_los_commits_de_sus_intentos() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿más capacidad?");
    tried(&moves, "x2", "aaa", &[q]);

    let d = moves
        .add(
            Kind::Decision,
            "por aquí no",
            "yo",
            Scope::of(vec![q]),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();
    moves.hang(d, q).unwrap();

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Abandon));
}

#[test]
fn un_intento_colgado_despues_de_la_decision_ya_nace_abandonado() {
    // El caso que justifica derivarlo en vez de guardarlo: nadie vuelve atrás a
    // marcar nada, y aun así la línea entera se lee muerta.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿más capacidad?");
    let d = moves
        .add(
            Kind::Decision,
            "por aquí no",
            "yo",
            Scope::of(vec![q]),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();
    moves.hang(d, q).unwrap();

    tried(&moves, "x4, por probar", "bbb", &[q]);

    assert_eq!(moves.decided().unwrap().get("bbb"), Some(&Course::Abandon));
}

#[test]
fn bifurcar_desde_un_intento_abandonado_empieza_limpio() {
    // Y esto es lo contrario, a propósito. Probar otra cosa **porque** aquello
    // no funcionó es el movimiento que se hace al llegar a un callejón: heredar
    // el abandono por la ascendencia de git lo marcaría como más de lo mismo.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿más capacidad?");
    let a = tried(&moves, "x2", "aaa", &[q]);
    let d = moves
        .add(
            Kind::Decision,
            "x2 no lleva a nada",
            "yo",
            Scope::of(vec![a]),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();
    moves.hang(d, a).unwrap();

    // Como cuelga `placed` una bifurcación: de los padres del que citaba el
    // commit de partida, no del propio intento.
    tried(&moves, "y si en vez de capacidad, profundidad", "bbb", &[q]);

    let decided = moves.decided().unwrap();
    assert_eq!(decided.get("aaa"), Some(&Course::Abandon));
    assert_eq!(decided.get("bbb"), None, "es una hermana, no una hija");
}

#[test]
fn una_decision_sin_alcance_habla_de_donde_cuelga_y_no_del_arbol() {
    // Sin esto, escribir «esta línea está muerta» mirando un intento marcaría
    // toda la investigación, calladamente. Para una pregunta no tener alcance
    // significa hablar de todo; para una decisión sería una trampa.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let una = plain(&moves, Kind::Question, "¿capacidad?");
    let otra = plain(&moves, Kind::Question, "¿profundidad?");
    let a = tried(&moves, "x2", "aaa", &[una]);
    tried(&moves, "más capas", "bbb", &[otra]);

    let d = moves
        .add(
            Kind::Decision,
            "por aquí no",
            "yo",
            Scope::everything(),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();
    moves.hang(d, a).unwrap();

    let decided = moves.decided().unwrap();
    assert_eq!(decided.get("aaa"), Some(&Course::Abandon));
    assert_eq!(
        decided.get("bbb"),
        None,
        "la otra pregunta no era asunto suyo"
    );
}

#[test]
fn una_decision_colgada_de_nada_y_sin_alcance_no_tine_nada() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿capacidad?");
    tried(&moves, "x2", "aaa", &[q]);
    moves
        .add(
            Kind::Decision,
            "hay que dejarlo",
            "yo",
            Scope::everything(),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();

    assert!(moves.decided().unwrap().is_empty());
}

#[test]
fn cambiar_de_opinion_es_decidir_otra_vez_y_gana_la_ultima() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿capacidad?");
    tried(&moves, "x2", "aaa", &[q]);
    for course in [Course::Abandon, Course::Pursue] {
        let d = moves
            .add(
                Kind::Decision,
                "…",
                "yo",
                Scope::of(vec![q]),
                Vec::new(),
                Some(course),
            )
            .unwrap();
        moves.hang(d, q).unwrap();
    }

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Pursue));
}

#[test]
fn un_rumbo_en_algo_que_no_es_una_decision_se_rechaza() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);

    assert!(
        moves
            .add(
                Kind::Finding,
                "vi que no",
                "yo",
                Scope::everything(),
                Vec::new(),
                Some(Course::Abandon),
            )
            .is_err()
    );
}

#[test]
fn corregir_el_alcance_de_una_decision_la_hace_llegar_a_los_commits() {
    // El fallo que esto cierra salió corriéndolo: una decisión escrita mirando
    // un hallazgo se alcanzaba al hallazgo, y un hallazgo no es una línea —no
    // cuelga nada de él ni cita ningún commit—, así que la decisión no llegaba
    // a ninguna parte y la línea seguía leyéndose viva. Sin poder corregir el
    // alcance no habría forma de arreglarlo salvo escribirla otra vez.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿capacidad?");
    let a = tried(&moves, "x2", "aaa", &[q]);
    let f = plain(&moves, Kind::Finding, "sube la latencia");
    moves.hang(f, a).unwrap();
    let d = moves
        .add(
            Kind::Decision,
            "no seguimos",
            "yo",
            Scope::of(vec![f]),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();
    moves.hang(d, f).unwrap();
    assert!(
        moves.decided().unwrap().is_empty(),
        "no llega a ningún sitio"
    );

    moves
        .reword(d, None, Some(Scope::of(vec![a])), None, "yo")
        .unwrap();

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Abandon));
}

#[test]
fn corregir_el_alcance_no_toca_la_prosa_ni_el_rumbo() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿capacidad?");
    let d = moves
        .add(
            Kind::Decision,
            "no seguimos, la latencia no la paga nadie",
            "yo",
            Scope::everything(),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();

    moves
        .reword(d, None, Some(Scope::of(vec![q])), None, "yo")
        .unwrap();

    let body = moves.all().unwrap().remove(&d).unwrap();
    assert_eq!(body.prose, "no seguimos, la latencia no la paga nadie");
    assert_eq!(body.course, Some(Course::Abandon));
    assert_eq!(body.scope, Scope::of(vec![q]));
}

#[test]
fn cambiar_de_rumbo_no_toca_la_prosa_ni_el_alcance() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿capacidad?");
    tried(&moves, "x2", "aaa", &[q]);
    let d = moves
        .add(
            Kind::Decision,
            "no seguimos",
            "yo",
            Scope::of(vec![q]),
            Vec::new(),
            Some(Course::Abandon),
        )
        .unwrap();

    moves
        .reword(d, None, None, Some(Course::Pursue), "yo")
        .unwrap();

    assert_eq!(moves.decided().unwrap().get("aaa"), Some(&Course::Pursue));
    assert_eq!(moves.all().unwrap()[&d].scope, Scope::of(vec![q]));
}

#[test]
fn un_rumbo_en_una_redaccion_de_algo_que_no_decide_se_rechaza() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "vi que no");

    assert!(
        moves
            .reword(f, None, None, Some(Course::Pursue), "yo")
            .is_err()
    );
}

// ── Citar la evidencia, que llega después ──

#[test]
fn un_intento_puede_citar_un_ensayo_despues_de_escrito() {
    // Los ensayos se corren después de anotar el intento, así que la evidencia
    // se junta después. Si sólo pudiera viajar al crearlo, un intento nunca
    // podría apuntar a lo que se corrió con él.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let a = tried(&moves, "x2", "aaa", &[]);

    moves
        .cite(
            a,
            Cited {
                what: "trial".into(),
                id: "exp/t/aaa/trial/3/0".into(),
            },
            "yo",
        )
        .unwrap();

    let body = moves.all().unwrap().remove(&a).unwrap();
    assert_eq!(body.cites.len(), 2, "el commit que ya tenía, y el ensayo");
    assert_eq!(body.cites[1].id, "exp/t/aaa/trial/3/0");
}

#[test]
fn citar_dos_veces_lo_mismo_no_lo_duplica() {
    // Lo pedirían dos personas mirando la misma pantalla, y una lista con el
    // mismo ensayo dos veces no dice nada más que una con él una vez.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let a = tried(&moves, "x2", "aaa", &[]);
    let cited = Cited {
        what: "trial".into(),
        id: "exp/t/aaa/trial/3/0".into(),
    };

    moves.cite(a, cited.clone(), "yo").unwrap();
    moves.cite(a, cited, "otro").unwrap();

    assert_eq!(moves.all().unwrap()[&a].cites.len(), 2);
}

#[test]
fn una_pregunta_no_cita_commits_ni_ensayos() {
    // Habla de movimientos, no de piezas de la capa 1. Dejarla citar sería
    // dejar que apunte a un commit sin que nadie sepa qué significa eso.
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿capacidad?");

    assert!(
        moves
            .cite(
                q,
                Cited {
                    what: "commit".into(),
                    id: "aaa".into()
                },
                "yo"
            )
            .is_err()
    );
}

#[test]
fn un_hallazgo_si_cita_el_ensayo_donde_se_vio() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let f = plain(&moves, Kind::Finding, "la latencia sube");

    moves
        .cite(
            f,
            Cited {
                what: "trial".into(),
                id: "exp/t/aaa/trial/1/0".into(),
            },
            "yo",
        )
        .unwrap();

    assert_eq!(moves.all().unwrap()[&f].cites.len(), 1);
}

#[test]
fn citar_no_toca_la_prosa_ni_el_alcance_ni_el_rumbo() {
    let (_at, kept) = somewhere();
    let moves = Moves::of("t", &kept);
    let q = plain(&moves, Kind::Question, "¿capacidad?");
    let a = tried(&moves, "x2, la buena", "aaa", &[q]);

    moves
        .cite(
            a,
            Cited {
                what: "artifact".into(),
                id: "informe.pdf".into(),
            },
            "yo",
        )
        .unwrap();

    let body = moves.all().unwrap().remove(&a).unwrap();
    assert_eq!(body.prose, "x2, la buena");
    assert_eq!(moves.under().unwrap().parents_of(a), vec![q]);
}
