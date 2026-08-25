//! Qué se pliega al dibujar, que es lo único de un recorrido que es una regla y
//! no un recorrido.

use somatize_tree::journal::Verdict;
use somatize_tree::moves::Course;
use somatize_tree::walk::folds;

#[test]
fn se_pliega_lo_que_alguien_decidio_no_seguir() {
    assert!(folds(Some(Course::Abandon), None, false));
    assert!(folds(Some(Course::Superseded), None, false));
}

#[test]
fn no_se_pliega_lo_que_nadie_ha_decidido() {
    // Por defecto se dibuja todo. Plegar es la respuesta a que un árbol de
    // cuarenta variantes no se lee, no a que sobre nada.
    assert!(!folds(None, None, false));
    assert!(!folds(Some(Course::Pursue), None, false));
}

#[test]
fn no_se_pliega_lo_que_alguien_ha_marcado_mal() {
    // El que más importa. Un `invalid` es lo que pone en duda la medida en la
    // que se apoyó la decisión de abandonar la línea: esconderlo sería esconder
    // justo la razón para volver a mirarla.
    assert!(!folds(Some(Course::Abandon), Some(Verdict::Invalid), false));
}

#[test]
fn ni_lo_que_hereda_esa_duda() {
    // La misma razón un nivel más abajo, y la que hace que esto no se pueda
    // decidir mirando sólo lo que se escribió de este commit.
    assert!(!folds(Some(Course::Abandon), None, true));
}

#[test]
fn haber_mirado_y_no_encontrar_nada_no_la_despliega() {
    // `sound` dice que se miró y no había nada malo, así que no hay ninguna
    // razón nueva para volver: la decisión de abandonarla sigue en pie.
    assert!(folds(Some(Course::Abandon), Some(Verdict::Sound), false));
}
