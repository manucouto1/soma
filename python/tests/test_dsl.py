"""El DSL: el grafo como una expresión."""

import pytest

import soma_next
from soma_next import Filter, Graph, Step


class Sumar(Filter):
    def __init__(self, cuanto):
        self.cuanto = cuanto

    def forward(self, x):
        return x + self.cuanto


class Media(Filter):
    """Un agregador es un filtro que lee un mapa."""

    def forward(self, entradas):
        return sum(entradas.values()) / len(entradas)


class Eco(Step):
    def poll(self, ctx):
        return {"done": ctx["input"]}


# ── Encadenar ──


def test_un_solo_filtro_ya_es_un_grafo():
    g = Graph.somatize(Sumar(1))
    assert g.nodes() == ["sumar"]
    assert g.forward(41) == 42.0


def test_una_cadena():
    g = Graph.somatize(Sumar(1) >> Sumar(10) >> Sumar(100))
    assert g.nodes() == ["sumar", "sumar_2", "sumar_3"]
    assert g.forward(0) == 111.0


def test_named_pone_el_id():
    g = Graph.somatize(Sumar(1).named("primero") >> Sumar(10).named("segundo"))
    assert g.nodes() == ["primero", "segundo"]
    assert g.edges() == [("primero", "segundo")]


# ── Abrir y cerrar ramas ──


def test_un_diamante_se_lee_de_un_vistazo():
    g = Graph.somatize(Sumar(1) >> (Sumar(10).named("izq") | Sumar(100).named("der")) >> Media())
    assert g.edges() == [
        ("sumar", "izq"),
        ("sumar", "der"),
        ("izq", "media"),
        ("der", "media"),
    ]
    assert g.forward(0) == 56.0


def test_ramas_abiertas_salen_como_mapa():
    g = Graph.somatize(Sumar(1) >> (Sumar(10).named("izq") | Sumar(100).named("der")))
    assert g.forward(0) == {"izq": 11.0, "der": 101.0}


def test_una_rama_puede_ser_mas_larga_que_la_otra():
    g = Graph.somatize(
        Sumar(1).named("fuente")
        >> ((Sumar(1).named("izq") >> Sumar(1).named("izq2")) | Sumar(1).named("der"))
    )
    assert g.edges() == [
        ("fuente", "izq"),
        ("izq", "izq2"),
        ("fuente", "der"),
    ]


def test_tres_ramas():
    g = Graph.somatize(Sumar(0) >> (Sumar(1).named("a") | Sumar(2).named("b") | Sumar(3).named("c")))
    assert g.forward(0) == {"a": 1.0, "b": 2.0, "c": 3.0}


# ── Filtros y steps en la misma expresión ──


def test_un_step_encaja_donde_encajaria_un_filtro():
    g = Graph.somatize(Sumar(1) >> Eco())
    assert g.forward(41) == 42.0


# ── La herencia es lo que decide ──


def test_la_clase_obliga_a_implementar_su_metodo():
    class SinForward(Filter):
        pass

    class SinPoll(Step):
        pass

    with pytest.raises(TypeError, match="abstract method 'forward'"):
        SinForward()
    with pytest.raises(TypeError, match="abstract method 'poll'"):
        SinPoll()


def test_manda_la_herencia_no_los_metodos_que_tenga():
    class Confusa(Step):
        def poll(self, ctx):
            return {"done": ctx["input"]}

        def forward(self, x):  # existe, y da igual
            return x

    assert "Step {" in Graph.somatize(Confusa()).plan()


def test_heredar_de_las_dos_no_vale():
    class Ambas(Filter, Step):
        def forward(self, x):
            return x

        def poll(self, ctx):
            return {"done": ctx["input"]}

    with pytest.raises(TypeError, match="hereda de Filter y de Step a la vez"):
        Graph.somatize(Ambas())


def test_en_el_dsl_hay_que_heredar():
    class Suelto:
        def forward(self, x):
            return x * 2

    with pytest.raises(TypeError, match="tiene que heredar de soma_next.Filter"):
        Graph.somatize(Suelto() >> Sumar(1))


def test_lo_que_no_puede_ser_nodo_lo_dice():
    with pytest.raises(TypeError, match="tiene que heredar de"):
        Graph.somatize(Sumar(1) >> "esto no es un filtro")


# ── La puerta de abajo: node() y step() ──


def test_node_con_algo_que_hereda_de_step_es_una_contradiccion():
    class UnStep(Step):
        def poll(self, ctx):
            return {"done": ctx["input"]}

    g = Graph()
    with pytest.raises(TypeError, match="hereda de Step, así que se añade con step"):
        g.node("x", UnStep())


def test_step_con_algo_que_hereda_de_filter_es_una_contradiccion():
    g = Graph()
    with pytest.raises(TypeError, match="hereda de Filter, así que se añade con node"):
        g.step("x", Sumar(1))


def test_un_objeto_de_fuera_sigue_entrando_por_la_puerta_de_abajo():
    class Ajeno:  # no hereda de nada nuestro
        def forward(self, x):
            return x * 2

    g = Graph()
    g.node("ajeno", Ajeno())
    assert g.forward(21) == 42.0


# ── El DSL y las llamadas construyen lo mismo ──


def test_el_dsl_no_es_otra_cosa_que_node_y_edge():
    dsl = Graph.somatize(Sumar(1).named("a") >> Sumar(10).named("b"))

    a_mano = Graph()
    a_mano.node("a", Sumar(1))
    a_mano.node("b", Sumar(10))
    a_mano.edge("a", "b")

    assert dsl.nodes() == a_mano.nodes()
    assert dsl.edges() == a_mano.edges()
    assert dsl.plan() == a_mano.plan()


def test_el_numero_de_argumentos_lo_sigue_contando_rust():
    g = Graph()
    with pytest.raises(ValueError, match="toma \\(objeto\\)"):
        g.node()
