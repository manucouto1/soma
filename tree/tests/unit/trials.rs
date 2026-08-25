//! Los ensayos de una versión, leídos como los escribe soma-next.
//!
//! Los registros se escriben **a mano y con los nombres de soma-next** en vez
//! de llamar a su `take`/`report`: lo que hay que defender es que este lector
//! entiende ese formato, y llamar al escritor de la otra biblioteca haría que
//! los dos se pusieran de acuerdo en cualquier formato, incluido uno que no sea
//! el que hay en el disco de nadie. El formato está documentado en
//! `study/_run.py`, y esto es una copia de esa documentación que se ejecuta.

use somatize_store::{Local, Store};
use somatize_tree::trials::{Goal, Trials};

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("un directorio temporal");
    let kept = Local::at(at.path()).expect("un store dentro");
    (at, kept)
}

/// Un ensayo como lo ata soma-next: `<study>/trial/<n>/<attempt>`, con el
/// estado y la puntuación en el **registro** y la curva en el blob.
fn ran(
    kept: &Local,
    commit: &str,
    trial: u32,
    attempt: u32,
    state: &str,
    score: Option<f64>,
    reports: &[f64],
) {
    let blob = serde_json::json!({
        "point": "lr=0.001,batch=32",
        "reports": reports,
        "state": state,
        "because": if state == "pruned" { Some("no mejoraba") } else { None },
        "took": 12.5,
    });
    let digest = kept.put(blob.to_string().as_bytes()).expect("el blob");
    let mut meta: Vec<(String, String)> = vec![
        ("state".into(), state.into()),
        ("point".into(), "lr=0.001,batch=32".into()),
        ("who".into(), "maquina-3".into()),
    ];
    if let Some(score) = score {
        // `repr(float(score))`, que es lo que escribe Python.
        meta.push(("score".into(), format!("{score:?}")));
    }
    kept.bind(
        &format!("exp/t/{commit}/trial/{trial}/{attempt}"),
        &digest,
        meta,
    )
    .expect("el registro");
}

#[test]
fn el_estudio_de_un_commit_es_el_prefijo_bajo_el_que_ya_vive_su_diario() {
    // Todo el acoplamiento con soma-next es este nombre, y es la única parte de
    // esto que no se puede cambiar después sin mover directorios de alguien.
    let (_at, kept) = somewhere();

    assert_eq!(Trials::of("t", &kept).study("abc123"), "exp/t/abc123");
}

#[test]
fn los_ensayos_de_una_version_vuelven_en_un_recorrido_y_sin_leer_nada() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.4, 0.7, 0.83]);
    ran(&kept, "abc", 1, 0, "running", None, &[0.3]);

    let seen = Trials::of("t", &kept).of_commit("abc").unwrap();

    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].state.as_deref(), Some("done"));
    assert_eq!(seen[0].score, Some(0.83));
    assert_eq!(seen[0].who.as_deref(), Some("maquina-3"));
    assert_eq!(seen[1].score, None, "todavía corriendo, todavía sin nota");
}

#[test]
fn gana_el_intento_mas_alto_porque_reclamar_es_un_enlace() {
    // Un ensayo cuya máquina murió se queda reclamado para siempre, y
    // rescatarlo escribiendo encima sería una carrera. El rescate es reclamar
    // el intento siguiente, y quien lee se queda con el más alto.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "running", None, &[0.2]);
    ran(&kept, "abc", 0, 1, "done", Some(0.91), &[0.2, 0.6, 0.91]);

    let seen = Trials::of("t", &kept).of_commit("abc").unwrap();

    assert_eq!(seen.len(), 1, "un ensayo, no dos");
    assert_eq!(seen[0].attempt, 1);
    assert_eq!(seen[0].score, Some(0.91));
}

#[test]
fn los_ensayos_de_otra_version_no_son_de_esta() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.8), &[0.8]);
    ran(&kept, "def", 0, 0, "done", Some(0.9), &[0.9]);

    assert_eq!(Trials::of("t", &kept).of_commit("abc").unwrap().len(), 1);
}

#[test]
fn lo_que_hay_en_el_store_y_no_es_un_ensayo_se_pregunta_y_no_se_supone() {
    // Un store guarda lo que le echen: una caché, el diario, el razonamiento.
    let (_at, kept) = somewhere();
    let digest = kept.put(b"algo").unwrap();
    kept.bind("exp/t/abc/said/0", &digest, Vec::new()).unwrap();
    kept.bind("exp/t/move/3/said/0", &digest, Vec::new())
        .unwrap();
    kept.bind("otra-cosa/trial/0/0", &digest, Vec::new())
        .unwrap();
    ran(&kept, "abc", 0, 0, "done", Some(0.8), &[0.8]);

    assert_eq!(Trials::of("t", &kept).of_commit("abc").unwrap().len(), 1);
    assert_eq!(Trials::of("t", &kept).counted().unwrap().len(), 1);
}

#[test]
fn contar_cuarenta_versiones_cuesta_un_recorrido_y_no_cuarenta() {
    // Lo que el raíl necesita. Preguntarlo commit a commit serían cuarenta
    // recorridos del store para dibujar una lista de cuarenta filas.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "failed", None, &[]);
    ran(&kept, "abc", 2, 0, "running", None, &[0.1]);
    ran(&kept, "def", 0, 0, "pruned", Some(0.4), &[0.4]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].trials, 3);
    assert_eq!(counted["abc"].done, 1);
    assert_eq!(counted["abc"].failed, 1);
    assert_eq!(counted["abc"].running, 1);
    assert_eq!(counted["def"].pruned, 1);
}

#[test]
fn un_ensayo_rescatado_se_cuenta_una_vez() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "running", None, &[0.2]);
    ran(&kept, "abc", 0, 1, "done", Some(0.9), &[0.9]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].trials, 1);
    assert_eq!(counted["abc"].done, 1);
    assert_eq!(counted["abc"].running, 0, "el intento muerto no sigue vivo");
}

#[test]
fn sin_direccion_declarada_no_se_dice_cual_es_el_mejor() {
    // El que más importa. Si `0.0837` es bueno o malo depende de si esa métrica
    // se maximiza o se minimiza, y esa dirección no está en ningún registro:
    // vive en el `Goal` que se le pasa al sampler. «El mejor» es la palabra que
    // más se copia a un informe sin comprobar, así que o se sabe o no se dice.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "done", Some(0.21), &[0.21]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].best, None);
    assert_eq!(counted["abc"].lowest, Some(0.21), "el rango sí es cierto");
    assert_eq!(counted["abc"].highest, Some(0.83));
}

#[test]
fn con_la_direccion_declarada_si() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "done", Some(0.21), &[0.21]);

    let maximizando = Trials::of("t", &kept).towards(Some(Goal::Max));
    let minimizando = Trials::of("t", &kept).towards(Some(Goal::Min));

    assert_eq!(maximizando.counted().unwrap()["abc"].best, Some(0.83));
    assert_eq!(minimizando.counted().unwrap()["abc"].best, Some(0.21));
}

#[test]
fn una_puntuacion_podada_no_entra_en_el_rango() {
    // Es real y no es comparable: se midió tras menos épocas. Meterla haría el
    // rango más ancho de lo que nadie llegó a medir, y un rango que exagera es
    // peor que no tenerlo.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.83), &[0.83]);
    ran(&kept, "abc", 1, 0, "pruned", Some(0.05), &[0.05]);

    let counted = Trials::of("t", &kept).counted().unwrap();

    assert_eq!(counted["abc"].lowest, Some(0.83));
    assert_eq!(counted["abc"].pruned, 1, "contado, pero fuera del rango");
}

#[test]
fn la_curva_se_paga_aparte_y_dice_por_que_paro() {
    // El otro lado de la regla de coste: la curva crece, así que vive en el
    // blob y sólo la lee quien pide verla.
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "pruned", Some(0.4), &[0.1, 0.3, 0.4]);
    let trials = Trials::of("t", &kept);
    let seen = trials.of_commit("abc").unwrap();

    let curve = trials.curve(&seen[0]).unwrap().expect("la curva");

    assert_eq!(curve.reports, vec![0.1, 0.3, 0.4]);
    assert_eq!(curve.because.as_deref(), Some("no mejoraba"));
    assert_eq!(curve.took, Some(12.5));
}

#[test]
fn una_version_sin_ensayos_no_es_un_error() {
    let (_at, kept) = somewhere();

    assert!(Trials::of("t", &kept).of_commit("abc").unwrap().is_empty());
    assert!(Trials::of("t", &kept).counted().unwrap().is_empty());
}

#[test]
fn solo_un_done_es_comparable_con_otro_done() {
    let (_at, kept) = somewhere();
    ran(&kept, "abc", 0, 0, "done", Some(0.8), &[0.8]);
    ran(&kept, "abc", 1, 0, "pruned", Some(0.4), &[0.4]);
    ran(&kept, "abc", 2, 0, "running", None, &[0.1]);
    let seen = Trials::of("t", &kept).of_commit("abc").unwrap();

    assert!(seen[0].comparable());
    assert!(!seen[1].comparable(), "se midió tras menos épocas");
    assert!(!seen[2].comparable());
}
