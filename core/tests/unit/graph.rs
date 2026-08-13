use soma_next_core::Graph;

#[test]
fn un_grafo_recien_creado_esta_vacio() {
    let g = Graph::new();
    assert_eq!(g.len(), 0);
    assert!(g.is_empty());
}
