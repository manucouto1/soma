//! De quién es cada dato del store, y qué se dice de lo que no es de nadie.
//!
//! Los registros se escriben **a mano y con los nombres de soma-next**, por lo
//! mismo que en `trials`: lo que hay que defender es que este lector entiende
//! lo que hay en el disco de alguien, y llamar al escritor de la otra
//! biblioteca haría que los dos se pusieran de acuerdo en cualquier formato,
//! incluido uno que nadie tiene guardado.

use soma_next_core::Key;
use soma_next_store::{Local, Store, name_of};
use soma_tree::data::{How, under};
use soma_tree::snapshot::Snapshot;
use std::collections::HashMap;

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("un directorio temporal");
    let kept = Local::at(at.path()).expect("un store dentro");
    (at, kept)
}

/// Un valor guardado, con al lado lo que escribió quien lo produjo.
fn kept_value(kept: &Local, key: &str, node: &str, fingerprint: &str, env: &str) {
    let digest = kept.put(b"lo que produjo").expect("el blob");
    kept.bind(
        &name_of(&Key::new(key)),
        &digest,
        vec![
            ("node".into(), node.into()),
            ("fingerprint".into(), fingerprint.into()),
            ("input".into(), "sha256:la-entrada".into()),
            ("env".into(), env.into()),
        ],
    )
    .expect("se ata");
}

/// Un sondeo como vuelve del probe: sólo los dos campos que se leen de dentro.
fn taken(names: &[(&str, &str)], fingerprints: &[(&str, &str)]) -> Snapshot {
    let of = |pairs: &[(&str, &str)]| {
        pairs
            .iter()
            .map(|(node, told)| (node.to_string(), serde_json::json!(told)))
            .collect::<serde_json::Map<_, _>>()
    };
    Snapshot {
        commit: "aaaa".into(),
        built_from: "experiments.thing:build".into(),
        input: "sentinel".into(),
        environment: Default::default(),
        snapshot: serde_json::json!({
            "names": of(names),
            "fingerprints": of(fingerprints),
        }),
        architecture: serde_json::Value::Null,
        inside: serde_json::Value::Null,
        reaches: serde_json::Value::Null,
        declaring: None,
        code: Default::default(),
        mapped: Vec::new(),
        unneeded: Vec::new(),
    }
}

#[test]
fn un_valor_es_de_la_version_que_lo_va_a_pedir() {
    // Lo más fuerte que se puede decir: no es que lo hiciera un código
    // parecido, es que es el dato que esta versión pediría.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:uno", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([("c1", taken(&[("embed", "sha256:uno")], &[]))]);

    let said = under(&kept, &known).expect("se lee");

    assert_eq!(said.len(), 1);
    assert_eq!(said[0].of.get("c1"), Some(&How::Named));
}

#[test]
fn y_tambien_de_la_que_solo_comparte_el_codigo() {
    // La que aguanta lo que la otra no. Una clave se calcula contra el entorno
    // del que sondea, así que sondear hoy un commit de hace tres meses da otras
    // claves; la huella la escribió quien corrió, entonces, y sigue ahí.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:uno", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([(
        "c1",
        taken(&[("embed", "sha256:otra")], &[("embed", "a1b2c3d4")]),
    )]);

    let said = under(&kept, &known).expect("se lee");

    assert_eq!(said[0].of.get("c1"), Some(&How::Written));
}

#[test]
fn valiendo_las_dos_gana_la_clave() {
    // Decir lo más débil pudiendo decir lo otro es perder información.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:uno", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([(
        "c1",
        taken(&[("embed", "sha256:uno")], &[("embed", "a1b2c3d4")]),
    )]);

    let said = under(&kept, &known).expect("se lee");

    assert_eq!(said[0].of.get("c1"), Some(&How::Named));
}

#[test]
fn un_dato_puede_ser_de_varias_versiones_a_la_vez() {
    // Y no es un empate que haya que resolver: cuatro commits seguidos que no
    // tocan `embed` comparten su respuesta, que es exactamente para lo que hay
    // una caché. Elegir uno sería inventarse una respuesta.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:uno", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([
        ("c1", taken(&[("embed", "sha256:uno")], &[])),
        ("c2", taken(&[("embed", "sha256:uno")], &[])),
    ]);

    let said = under(&kept, &known).expect("se lee");

    assert_eq!(said[0].of.len(), 2, "{:?}", said[0].of);
}

#[test]
fn lo_que_no_es_de_ninguna_version_sale_igualmente_y_dice_lo_que_sabe() {
    // El caso que esto existe para no callar. Un hash mudo se queda en el
    // store para siempre; uno que dice qué nodo y qué código lo hizo sigue
    // siendo una frase verdadera aunque no case con nada de lo que se miró.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:uno", "embed", "de-otra-rama", "9f2c1a");
    let known = HashMap::from([(
        "c1",
        taken(&[("embed", "sha256:otra")], &[("embed", "a1b2c3d4")]),
    )]);

    let said = under(&kept, &known).expect("se lee");

    assert!(said[0].is_nobodys());
    assert_eq!(said[0].node.as_deref(), Some("embed"));
    assert_eq!(said[0].fingerprint.as_deref(), Some("de-otra-rama"));
    assert_eq!(said[0].environment.as_deref(), Some("9f2c1a"));
}

#[test]
fn la_contabilidad_de_quien_mira_no_es_un_dato_de_nadie() {
    // Tres escritores comparten este store y sólo uno deja intermedios. El
    // diario, la caché de sondeos y la lectura de un entorno son lo que
    // **explica** la atribución, no algo que atribuir — y contarlos sería
    // enseñarle a alguien su propio cuaderno como si fuera un intermedio que a
    // lo mejor sobra.
    let (_at, kept) = somewhere();
    let digest = kept.put(b"lo que sea").expect("el blob");
    for name in [
        "exp/una-investigacion/aaaa/trial/1/0",
        "snapshot:aaaa:sha256:receta",
        "env/9f2c1a",
    ] {
        kept.bind(name, &digest, Vec::new()).expect("se ata");
    }
    kept_value(&kept, "sha256:uno", "embed", "a1b2c3d4", "9f2c1a");

    let said = under(&kept, &HashMap::new()).expect("se lee");

    assert_eq!(
        said.len(),
        1,
        "{:?}",
        said.iter().map(|one| &one.name).collect::<Vec<_>>()
    );
    assert_eq!(said[0].node.as_deref(), Some("embed"));
}

#[test]
fn un_sondeo_de_antes_de_que_esto_existiera_se_lee_igual() {
    // Un snapshot guardado antes de que el modelo publicara estos campos sigue
    // siendo una respuesta buena a todo lo demás. Caerse por leer algo que
    // entonces no se le pidió sería tirar el registro de una investigación por
    // una función que se añadió después.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:uno", "embed", "a1b2c3d4", "9f2c1a");
    let mut viejo = taken(&[], &[]);
    viejo.snapshot = serde_json::json!({"shape": {}});
    let known = HashMap::from([("c1", viejo)]);

    let said = under(&kept, &known).expect("se lee");

    assert!(said[0].is_nobodys(), "no se sabe, y eso no es caerse");
}
