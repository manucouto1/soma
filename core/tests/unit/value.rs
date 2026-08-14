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
