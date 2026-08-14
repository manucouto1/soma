"""Un pipeline real de extremo a extremo: texto → encoder → cuello → LSTM.

Es el caso que motivó `Opaque`, y está aquí entero por dos razones: comprueba
que los gradientes atraviesan varios nodos, y **documenta el patrón** — cómo se
escribe un nodo con parámetros, dónde va el bucle de entrenamiento y qué pasa
cuando un nodo del pipeline no tiene gradientes.

El "LLM" es un `Embedding` + `TransformerEncoderLayer`, no un modelo real: lo
que se prueba es la costura, no el modelo. Con un `AutoModel` de HuggingFace el
patrón es idéntico.
"""

import pytest

from soma_next import Done, Graph, Node, Opaque

torch = pytest.importorskip("torch")
nn = torch.nn

VOCAB, DIM, CUELLO, CLASES, LARGO = 64, 32, 8, 3, 6


def _ids(textos):
    """Tokenizador de juguete, determinista a propósito.

    `hash()` de un str va salado por proceso en Python, así que un test que lo
    usara daría una tokenización distinta en cada ejecución.
    """
    filas = []
    for t in textos:
        palabras = [sum(map(ord, p)) % VOCAB for p in t.split()][:LARGO]
        filas.append(palabras + [0] * (LARGO - len(palabras)))
    return torch.tensor(filas)


# ── Un nodo sin gradientes: texto → texto, y NO se envuelve ──


class Lematizador(Node):
    def forward(self, textos, ctx):
        return Done([t.strip().lower().replace("corriendo", "correr") for t in textos])


# ── Los tres con parámetros. Ojo: tienen los módulos, no heredan de nn.Module ──
#
# Heredar de `nn.Module` registra los parámetros solo, pero rompe llamar al nodo
# como un módulo: nuestro `forward` lleva `ctx` y torch lo llama sin él.


class Encoder(Node):
    """Donde empieza la gráfica de gradientes."""

    def __init__(self):
        self.emb = nn.Embedding(VOCAB, DIM)
        self.enc = nn.TransformerEncoderLayer(DIM, 4, dim_feedforward=64, batch_first=True)
        self.ultima_salida = None

    def forward(self, textos, ctx):
        self.ultima_salida = self.enc(self.emb(_ids(textos)))
        return Done(Opaque(self.ultima_salida))

    def parameters(self):
        return list(self.emb.parameters()) + list(self.enc.parameters())


class Cuello(Node):
    def __init__(self):
        self.proj = nn.Linear(DIM, CUELLO)
        self.ultima_entrada = None

    def forward(self, h, ctx):
        self.ultima_entrada = h
        return Done(Opaque(self.proj(h)))

    def parameters(self):
        return list(self.proj.parameters())


class Clasificador(Node):
    def __init__(self):
        self.lstm = nn.LSTM(CUELLO, 16, batch_first=True)
        self.cabeza = nn.Linear(16, CLASES)

    def forward(self, h, ctx):
        salida, _ = self.lstm(h)
        return Done(Opaque(self.cabeza(salida[:, -1, :])))

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.cabeza.parameters())


TEXTOS = ["  El perro Corriendo rápido  ", "gato duerme mucho", "  Pájaro vuela alto  "]
ETIQUETAS = torch.tensor([0, 1, 2])


@pytest.fixture
def pipeline():
    torch.manual_seed(0)
    nodos = (Lematizador(), Encoder(), Cuello(), Clasificador())
    return Graph.somatize(nodos[0] >> nodos[1] >> nodos[2] >> nodos[3]), nodos


def _parametros(g):
    """Lo que un `soma_next.torch.parameters(g)` haría, si existiera."""
    return [p for nid in g.nodes() for p in getattr(g.implementation(nid), "parameters", list)()]


# ── La topología ──


def test_cada_parte_es_un_nodo(pipeline):
    g, _ = pipeline
    assert g.nodes() == ["lematizador", "encoder", "cuello", "clasificador"]
    assert g.plan().count("Execute") == 4


# ── Los gradientes ──


def test_el_backward_atraviesa_los_tres_nodos_con_parametros(pipeline):
    g, (_, encoder, cuello, clasificador) = pipeline

    logits = g.forward(TEXTOS)
    assert logits.shape == (len(TEXTOS), CLASES)
    assert logits.grad_fn is not None, "la salida sigue enganchada a la gráfica"

    torch.nn.functional.cross_entropy(logits, ETIQUETAS).backward()

    for nombre, nodo in (("encoder", encoder), ("cuello", cuello), ("clf", clasificador)):
        faltan = [p for p in nodo.parameters() if p.grad is None]
        assert not faltan, f"{nombre} tiene {len(faltan)} parámetros sin gradiente"


def test_el_tensor_cruza_la_arista_siendo_el_mismo_objeto(pipeline):
    g, (_, encoder, cuello, _) = pipeline
    g.forward(TEXTOS)
    assert cuello.ultima_entrada is encoder.ultima_salida


def test_el_nodo_sin_gradientes_convive_sin_romper_nada(pipeline):
    g, (lematizador, _, _, _) = pipeline
    # No tiene parámetros, y su salida cruza convertida (texto), no opaca.
    assert not hasattr(lematizador, "parameters")
    assert lematizador.forward(["  Corriendo  "], None).value == ["correr"]

    g.forward(TEXTOS)  # y el pipeline entero sigue funcionando


# ── El bucle de entrenamiento, que va FUERA del grafo ──


def test_el_pipeline_entrena(pipeline):
    g, _ = pipeline
    opt = torch.optim.Adam(_parametros(g), lr=0.01)

    primera = ultima = None
    for paso in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(g.forward(TEXTOS), ETIQUETAS)
        loss.backward()
        opt.step()
        if paso == 0:
            primera = loss.item()
        ultima = loss.item()

    assert ultima < primera / 2, f"la pérdida bajó de {primera:.4f} a {ultima:.4f}"


def test_los_pesos_que_actualiza_el_optimizador_son_los_que_usa_el_grafo(pipeline):
    g, (_, _, cuello, _) = pipeline
    antes = cuello.proj.weight.detach().clone()

    opt = torch.optim.Adam(_parametros(g), lr=0.1)
    opt.zero_grad()
    torch.nn.functional.cross_entropy(g.forward(TEXTOS), ETIQUETAS).backward()
    opt.step()

    assert not torch.allclose(antes, cuello.proj.weight), "el paso no cambió nada"
    assert g.implementation("cuello") is cuello, "el grafo guarda el mismo objeto"
