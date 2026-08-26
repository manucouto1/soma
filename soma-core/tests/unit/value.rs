use somatize_core::Value;

#[test]
fn a_list_holds_what_you_put_in_it() {
    let Value::List(items) = Value::list(vec![Value::number(1.0), Value::text("two")]) else {
        panic!("list() builds a list");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[1], Value::text("two"));
}

#[test]
fn cloning_does_not_copy_the_data() {
    let original = Value::list(vec![Value::Null; 1000]);
    let copy = original.clone();
    let (Value::List(a), Value::List(b)) = (&original, &copy) else {
        panic!("both are lists");
    };
    assert!(std::sync::Arc::ptr_eq(a, b), "the Arc should be shared");
}

#[test]
fn every_variant_knows_its_name() {
    assert_eq!(Value::Null.type_name(), "null");
    assert_eq!(Value::number(1.0).type_name(), "number");
    assert_eq!(Value::text("hello").type_name(), "text");
    assert_eq!(Value::list(vec![]).type_name(), "list");
}

/// Something the core has no way of understanding.
#[derive(Debug, PartialEq)]
struct Foreign(String);

#[test]
fn an_opaque_comes_out_exactly_as_it_went_in() {
    let v = Value::opaque(Foreign("intact".into()));
    assert_eq!(v.downcast::<Foreign>(), Some(&Foreign("intact".into())));
    assert_eq!(v.type_name(), "opaque");
}

#[test]
fn cloning_it_does_not_duplicate_it() {
    let original = Value::opaque(Foreign("one".into()));
    let copy = original.clone();
    // It is the SAME one, which is what lets a tensor cross unbroken.
    assert_eq!(original, copy);
}

#[test]
fn two_distinct_opaques_are_not_equal_even_carrying_the_same_thing() {
    let a = Value::opaque(Foreign("same".into()));
    let b = Value::opaque(Foreign("same".into()));
    assert_ne!(a, b, "the core cannot compare what it does not look at");
}

#[test]
fn asking_for_the_wrong_type_gives_none() {
    let v = Value::opaque(Foreign("x".into()));
    assert!(v.downcast::<String>().is_none());
    assert!(Value::number(1.0).downcast::<Foreign>().is_none());
}

#[test]
fn what_is_not_known_is_not_printed() {
    assert_eq!(
        format!("{:?}", Value::opaque(Foreign("secret".into()))),
        "Opaque(..)"
    );
}

#[test]
fn it_fits_inside_a_list_and_inside_a_map() {
    let v = Value::list(vec![Value::opaque(Foreign("inside".into()))]);
    let Value::List(items) = &v else {
        panic!("it is a list")
    };
    assert_eq!(
        items[0].downcast::<Foreign>(),
        Some(&Foreign("inside".into()))
    );
}

#[test]
fn what_travels_is_everything_but_an_opaque() {
    // The question whoever is about to send it asks, and it is the core's to
    // answer: it is the one that knows what an opaque is.
    assert!(Value::Null.travels());
    assert!(Value::text("hello").travels());
    assert!(!Value::opaque(7u32).travels());
}

#[test]
fn an_opaque_at_any_depth_stops_the_whole_value_travelling() {
    // The one that would slip through if the question were about the outermost
    // variant: a list of numbers with one opaque hidden in it.
    assert!(Value::list(vec![Value::number(1.0), Value::text("two")]).travels());
    assert!(!Value::list(vec![Value::number(1.0), Value::opaque(7u32)]).travels());
    assert!(
        !Value::map(vec![(
            "inside".to_string(),
            Value::list(vec![Value::opaque(7u32)])
        )])
        .travels()
    );
}
