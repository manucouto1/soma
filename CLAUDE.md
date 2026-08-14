# soma-next

Re-derivación de [Soma](/mnt/cluster/projects/soma) escrita a mano, un caso de
uso cada vez. El objetivo no es un diseño mejor: es **autoría**. Un sistema que
diseñaste tú lo sostienes en la cabeza aunque tenga trescientos tipos; uno que
no, no — y por eso el original, que funciona y está publicado, dejó de ser
mantenible por su autor.

Objetivo secundario, no negociable: aprender Rust escribiéndolo.

## Reglas

**Rebanadas verticales, no capas de crate.** Nada se escribe sin un consumidor
real *hoy*. Construir un crate entero antes de que algo lo use es cómo el
original acabó con 14 traits de un solo implementor y 2 de ninguno.

**Taxonomía de tipos** — la regla que evita la sopa de `dyn Trait`:

- **enum** cuando el conjunto es cerrado y lo conoces tú. Es el lenguaje del
  dominio, y el compilador lleva la cuenta al añadir variantes.
- **trait** solo cuando la implementación la pone *otro* (el usuario). Si no
  puedes nombrar dos implementors reales hoy, es una struct.
- **struct con typestate** para los invariantes: que un estado imposible no se
  pueda escribir, en vez de validarlo.

**Un fichero por tipo.** El tipo, sus `impl` inherentes y los errores que
producen sus operaciones, juntos. Un `impl` inherente **nunca** se parte entre
ficheros: si te apetece partirlo, la operación probablemente no era un método
de ese tipo (fue el caso de `run`, que necesitaba grafo *y* catálogo *y*
entrada, y acabó siendo una función de `execution.rs`).

Lo que en Rust no se puede sostener, y conviene saberlo si vienes de Java: el
comportamiento no pertenece a un tipo, pertenece al par **(tipo, trait)**.
`impl Filter for PyFilter` vive en otro crate. Los `impl` de trait se dispersan
por necesidad y se buscan por el trait, no por el tipo.

**Los tests viven fuera de `src/`.** Son otro crate: solo ven la API pública,
así que no pueden pasar apoyándose en lo privado. Un binario de test por tipo
(`tests/unit/main.rs` con un `mod` por módulo), no uno por fichero.

**El núcleo no sabe de Python.** `#[pyclass]` no entra en `core/`. En cuanto un
tipo del núcleo lo lleva, deja de poder usarse sin un intérprete cargado.
`python/` traduce; si una regla del dominio acaba escrita ahí, está mal puesta.

## El original como oráculo, no como plantilla

`/mnt/cluster/projects/soma` está **congelado**: consulta y bugfixes, cero
features. Sus 31.994 líneas de test son la especificación ejecutable de lo que
tiene que ser verdad.

Se leen como **cuestionario, no como código**. Un test viejo mezcla dos cosas:
qué debe ser verdad (conocimiento, irremplazable) y cómo se invoca (diseño, que
se decide aquí). Copiarlos tal cual recrearía la API vieja, que es justo lo que
no queremos. De cada fichero se extrae la lista de garantías y se contesta a
cada una con la forma de llamada que decidamos.

## Comandos

```bash
conda activate mos                  # PyO3 no compila fuera de este env
cargo test --workspace
cargo clippy --workspace -- -D warnings && cargo fmt --all -- --check
cd python && maturin develop && python -m pytest tests/ -q
```

## Estado

Esqueleto andante: `Graph()` cruza Rust→PyO3→Python y no hace nada más. Qué es
un nodo, y por tanto `node()` y `edge()`, es el caso de uso 1 — sin diseñar a
propósito. Ver `docs/casos-de-uso.md`.
