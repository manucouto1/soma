use soma_next_core::Value;

#[test]
fn un_vector_lleva_su_forma() {
    let Value::Tensor { values, shape } = Value::vector(vec![1.0, 2.0, 3.0]) else {
        panic!("vector() construye un tensor");
    };
    assert_eq!(*values, vec![1.0, 2.0, 3.0]);
    assert_eq!(shape, vec![3]);
}

#[test]
fn un_escalar_no_tiene_forma() {
    let Value::Tensor { values, shape } = Value::scalar(7.0) else {
        panic!("scalar() construye un tensor");
    };
    assert_eq!(*values, vec![7.0]);
    assert!(shape.is_empty());
}

#[test]
fn clonar_no_copia_los_datos() {
    let original = Value::vector(vec![1.0; 1000]);
    let copia = original.clone();
    let (Value::Tensor { values: a, .. }, Value::Tensor { values: b, .. }) = (&original, &copia)
    else {
        panic!("los dos son tensores");
    };
    assert!(std::sync::Arc::ptr_eq(a, b), "el Arc debería compartirse");
}

#[test]
fn cada_variante_sabe_como_se_llama() {
    assert_eq!(Value::Null.type_name(), "null");
    assert_eq!(Value::text("hola").type_name(), "text");
    assert_eq!(Value::scalar(1.0).type_name(), "tensor");
}
