"""Ramas que se lanzan a la vez.

`|` abre ramas en el DSL, y desde CU9 eso no es solo topología: las ramas de
una wave **corren a la vez**, cada una entera en su hilo. Aquí se comprueban
las tres cosas que pueden salir mal cruzando la frontera: que el plan tenga la
forma que dice, que los hilos existan de verdad, y que el GIL no lo estropee.
"""

import subprocess
import sys
import textwrap
import threading

import pytest

from soma_next import Done, Graph, Node
from conftest import Media, Sumar

PLAZO = 10


# ── Nodos que miran cómo se les ejecuta ──


class Apuntador(Node):
    """Se apunta en una lista compartida: quién, cuándo y en qué hilo."""

    def __init__(self, nombre, diario, cerrojo):
        self.nombre = nombre
        self.diario = diario
        self.cerrojo = cerrojo

    def forward(self, x, ctx):
        with self.cerrojo:
            self.diario.append((self.nombre, threading.get_ident()))
        return Done(x)


class Cita(Node):
    """No termina hasta que ha llegado la otra rama.

    Si las ramas se ejecutaran una detrás de otra, la primera esperaría a una
    segunda que todavía no ha empezado y la barrera reventaría al agotar el
    plazo. `Barrier.wait()` suelta el GIL mientras espera, que es lo que hace
    que esto pueda funcionar con nodos de Python.
    """

    def __init__(self, barrera, falla=None):
        self.barrera = barrera
        self.falla = falla

    def forward(self, x, ctx):
        self.barrera.wait()
        if self.falla:
            raise ValueError(self.falla)
        return Done(x)


def en_otro_proceso(fuente):
    """Corre un programa que usa waves, con plazo, en un intérprete aparte.

    Tiene que ser otro **proceso** y no otro hilo: si el motor no soltara el
    GIL, los hilos de la wave se quedarían esperándolo y con ellos el
    intérprete entero —hasta un `join(timeout=…)` del hilo principal necesita
    el GIL para volver—. Un cuelgue así no lo puede cazar nada de dentro.
    """
    try:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(fuente)],
            capture_output=True,
            text=True,
            timeout=PLAZO,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"el programa no volvió en {PLAZO}s. El motor no soltó el GIL: las ramas "
            "de la wave lo están esperando y quien lo tiene está bloqueado dentro de "
            "Rust. Falta `py.allow_threads` alrededor de `executor.run` en "
            "`python/src/lib.rs`."
        )


# ── La forma del plan ──


def test_una_cadena_no_tiene_waves():
    g = Graph.somatize(Sumar(1).named("a") >> Sumar(10).named("b"))
    assert "Wave" not in g.plan()


def test_dos_ramas_salen_como_una_wave_en_el_plan():
    g = Graph.somatize(
        Sumar(1).named("f") >> (Sumar(10).named("i") | Sumar(100).named("d")) >> Media().named("j")
    )
    plan = g.plan()
    assert "Wave" in plan
    assert plan.count("Execute") == 4


def test_una_rama_larga_es_una_sola_rama_de_la_wave():
    # `a >> (b >> b2 | c) >> d`: la wave lleva dentro una secuencia.
    g = Graph.somatize(
        Sumar(1).named("a")
        >> ((Sumar(10).named("b") >> Sumar(20).named("b2")) | Sumar(100).named("c"))
        >> Media().named("d")
    )
    plan = g.plan()
    assert "Wave" in plan
    # La secuencia de la rama larga está dentro de la wave, no fuera.
    assert plan.index("Wave") < plan.index('NodeId("b2")')


def test_el_dsl_con_ramas_da_el_mismo_plan_que_node_y_edge():
    # La decisión 6 de CU5, ahora con waves de por medio: el plan sale del
    # grafo, no de la expresión, así que las dos puertas dan lo mismo.
    dsl = Graph.somatize(
        Sumar(1).named("f") >> (Sumar(10).named("i") | Sumar(100).named("d")) >> Media().named("j")
    )

    a_mano = Graph()
    for nombre, nodo in [("f", Sumar(1)), ("i", Sumar(10)), ("d", Sumar(100)), ("j", Media())]:
        a_mano.node(nombre, nodo)
    for origen, destino in [("f", "i"), ("f", "d"), ("i", "j"), ("d", "j")]:
        a_mano.edge(origen, destino)

    assert dsl.plan() == a_mano.plan()


# ── Que los hilos existan de verdad ──


def test_las_ramas_corren_a_la_vez():
    barrera = threading.Barrier(2, timeout=PLAZO)
    g = Graph.somatize(Cita(barrera).named("izq") | Cita(barrera).named("der"))

    salida = g.forward("x")
    assert salida == {"izq": "x", "der": "x"}


def test_tres_ramas_tambien():
    barrera = threading.Barrier(3, timeout=PLAZO)
    g = Graph.somatize(
        Sumar(1).named("f")
        >> (Cita(barrera).named("x") | Cita(barrera).named("y") | Cita(barrera).named("z"))
    )
    g.forward(0)


def test_una_rama_entera_corre_en_el_mismo_hilo():
    # Lo que compra descomponer por ramas: el día que un nodo tenga
    # dispositivo, torch lo fija por hilo y la rama no salta de uno a otro.
    diario, cerrojo = [], threading.Lock()

    def testigo(nombre):
        return Apuntador(nombre, diario, cerrojo).named(nombre)

    g = Graph.somatize(
        testigo("a")
        >> ((testigo("b") >> testigo("b2") >> testigo("b3")) | (testigo("c") >> testigo("c2")))
        >> testigo("d")
    )
    g.forward("x")

    hilos = dict(diario)
    assert hilos["b"] == hilos["b2"] == hilos["b3"]
    assert hilos["c"] == hilos["c2"]
    assert hilos["b"] != hilos["c"], "las dos ramas comparten hilo: no van a la vez"
    assert hilos["a"] == hilos["d"], "lo de fuera de la wave corre en el hilo de quien ejecuta"


def test_el_orden_real_de_ejecucion_respeta_las_aristas():
    # El plan dice un orden; esto mira el que de verdad ocurrió, con hilos.
    diario, cerrojo = [], threading.Lock()

    def testigo(nombre):
        return Apuntador(nombre, diario, cerrojo).named(nombre)

    aristas = [("a", "b"), ("b", "b2"), ("a", "c"), ("b2", "d"), ("c", "d")]
    g = Graph.somatize(
        testigo("a") >> ((testigo("b") >> testigo("b2")) | testigo("c")) >> testigo("d")
    )
    g.forward("x")

    orden = [nombre for nombre, _ in diario]
    assert sorted(orden) == ["a", "b", "b2", "c", "d"], f"alguno sobra o falta: {orden}"
    for origen, destino in aristas:
        assert orden.index(origen) < orden.index(destino), (
            f"{destino} se ejecutó antes que {origen}: {orden}"
        )


# ── Que el resultado no dependa de nada de lo anterior ──


def test_el_diamante_da_lo_mismo_repartido_que_en_fila():
    g = Graph.somatize(
        Sumar(1).named("f") >> (Sumar(10).named("i") | Sumar(100).named("d")) >> Media().named("j")
    )
    assert g.forward(0) == 56.0


def test_lo_que_produce_cada_rama_por_dentro_llega_al_final():
    g = Graph.somatize(
        Sumar(1).named("a")
        >> (
            (Sumar(10).named("b") >> Sumar(20).named("b2"))
            | (Sumar(100).named("c") >> Sumar(200).named("c2"))
        )
        >> Media().named("d")
    )
    # 0 → 1 → rama b: 11, 31 · rama c: 101, 301 → media 166
    assert g.forward(0) == 166.0


def test_ejecutar_dos_veces_da_lo_mismo():
    g = Graph.somatize(
        Sumar(1).named("f") >> (Sumar(10).named("i") | Sumar(100).named("d")) >> Media().named("j")
    )
    assert [g.forward(0) for _ in range(5)] == [56.0] * 5


# ── Fallos ──


def test_si_fallan_dos_ramas_se_cuenta_siempre_la_primera():
    # Las dos fallan a la vez de verdad —quedan en verse antes de romperse—,
    # así que cuál llega antes es una carrera; el error que se cuenta no.
    barrera = threading.Barrier(2, timeout=PLAZO)
    g = Graph.somatize(
        Cita(barrera, falla="rompió la izquierda").named("izq")
        | Cita(barrera, falla="rompió la derecha").named("der")
    )

    with pytest.raises(ValueError, match="rompió la izquierda"):
        g.forward("x")


def test_el_error_dice_en_que_rama_fue():
    class Romper(Node):
        def forward(self, x, ctx):
            raise ValueError("me rompí")

    g = Graph.somatize(Sumar(1).named("sano") | Romper().named("malo"))
    with pytest.raises(ValueError, match="malo"):
        g.forward(0)


# ── Lo que el DSL no puede escribir ──


def test_un_grafo_que_no_es_serie_paralelo_se_ejecuta_aunque_no_se_reparta():
    # La «N»: `a→c, a→d, b→d`. No tiene árbol serie-paralelo —es un teorema—,
    # así que no hay wave; y no se puede escribir con `>>` y `|`, hay que
    # construirla con node()/edge(). Sigue ejecutándose como siempre.
    g = Graph()
    for nombre, nodo in [("a", Sumar(1)), ("b", Sumar(2)), ("c", Sumar(100)), ("d", Media())]:
        g.node(nombre, nodo)
    for origen, destino in [("a", "c"), ("a", "d"), ("b", "d")]:
        g.edge(origen, destino)

    assert "Wave" not in g.plan(), "la N no tiene árbol que recuperar"
    assert g.forward(0) == {"c": 101.0, "d": 1.5}


# ── El GIL ──


def test_el_motor_suelta_el_gil_mientras_corre():
    """La guarda de todo lo anterior, y la única que no puede vivir dentro.

    Una wave lanza hilos que llaman al `forward` de objetos Python. Si el hilo
    que entró por `Graph.forward` se quedara con el GIL cogido, esos hilos se
    bloquearían al pedirlo y el proceso entero se congelaría — no un error, un
    cuelgue. Por eso este test vive en otro proceso: aquí el plazo sí funciona.
    """
    hecho = en_otro_proceso("""
        import threading
        from soma_next import Done, Graph, Node

        barrera = threading.Barrier(2, timeout=5)

        class Cita(Node):
            def forward(self, x, ctx):
                barrera.wait()
                return Done(x)

        g = Graph.somatize(Cita().named("izq") | Cita().named("der"))
        assert g.forward("x") == {"izq": "x", "der": "x"}
        print("volvió")
    """)

    assert hecho.returncode == 0, hecho.stderr
    assert "volvió" in hecho.stdout


def test_dos_nodos_python_en_la_misma_wave_dan_el_resultado_correcto():
    """El GIL los serializa entre ellos y eso no lo arregla nada.

    Lo que sí tiene que valer es el resultado: que el intérprete reparta los
    turnos como quiera no puede cambiar lo que sale. Es el precio dicho claro
    —dos nodos Python puros no se solapan— y el límite de lo que una wave
    compra en este lado de la frontera.
    """
    contador, cerrojo = [], threading.Lock()

    class Cuenta(Node):
        def forward(self, x, ctx):
            for _ in range(2000):
                with cerrojo:
                    contador.append(1)
            return Done(len(contador))

    g = Graph.somatize(Cuenta().named("uno") | Cuenta().named("otro"))
    salida = g.forward(0)

    assert len(contador) == 4000, "se perdieron incrementos"
    assert set(salida) == {"uno", "otro"}
