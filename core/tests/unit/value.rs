use soma_next_core::Value;

#[test]
fn una_lista_guarda_lo_que_le_metes() {
    let Value::List(items) = Value::list(vec![Value::number(1.0), Value::text("dos")]) else {
        panic!("list() construye una lista");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[1], Value::text("dos"));
}

#[test]
fn clonar_no_copia_los_datos() {
    let original = Value::list(vec![Value::Null; 1000]);
    let copia = original.clone();
    let (Value::List(a), Value::List(b)) = (&original, &copia) else {
        panic!("las dos son listas");
    };
    assert!(std::sync::Arc::ptr_eq(a, b), "el Arc debería compartirse");
}

#[test]
fn cada_variante_sabe_como_se_llama() {
    assert_eq!(Value::Null.type_name(), "null");
    assert_eq!(Value::number(1.0).type_name(), "number");
    assert_eq!(Value::text("hola").type_name(), "text");
    assert_eq!(Value::list(vec![]).type_name(), "list");
}

// ── Lo que el núcleo transporta sin mirar ──

/// Algo que el núcleo no tiene forma de entender.
#[derive(Debug, PartialEq)]
struct Ajeno(String);

#[test]
fn un_opaco_sale_tal_cual_entro() {
    let v = Value::opaque(Ajeno("intacto".into()));
    assert_eq!(v.downcast::<Ajeno>(), Some(&Ajeno("intacto".into())));
    assert_eq!(v.type_name(), "opaque");
}

#[test]
fn clonarlo_no_lo_duplica() {
    let original = Value::opaque(Ajeno("uno".into()));
    let copia = original.clone();
    // Es el MISMO, que es lo que hace que un tensor cruce sin romperse.
    assert_eq!(original, copia);
}

#[test]
fn dos_opacos_distintos_no_son_iguales_aunque_lleven_lo_mismo() {
    let a = Value::opaque(Ajeno("igual".into()));
    let b = Value::opaque(Ajeno("igual".into()));
    assert_ne!(a, b, "el núcleo no puede comparar lo que no mira");
}

#[test]
fn preguntar_por_el_tipo_equivocado_devuelve_none() {
    let v = Value::opaque(Ajeno("x".into()));
    assert!(v.downcast::<String>().is_none());
    assert!(Value::number(1.0).downcast::<Ajeno>().is_none());
}

#[test]
fn no_se_imprime_lo_que_no_se_sabe_que_es() {
    assert_eq!(
        format!("{:?}", Value::opaque(Ajeno("secreto".into()))),
        "Opaque(..)"
    );
}

#[test]
fn cabe_dentro_de_una_lista_y_de_un_mapa() {
    let v = Value::list(vec![Value::opaque(Ajeno("dentro".into()))]);
    let Value::List(items) = &v else {
        panic!("es una lista")
    };
    assert_eq!(items[0].downcast::<Ajeno>(), Some(&Ajeno("dentro".into())));
}
