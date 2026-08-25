"""Writes down what a graph is, at one commit, without running any of it.

This runs *inside* a worktree, against whatever soma-next that checkout imports.
It holds a graph; nothing else in `soma-tree` ever does. What it writes is the
contract — the part meant to outlive this file being rewritten in another
language — so a field is added here before it is read anywhere else.

What is decided here: nothing. `soma_next.foreseen` owns the model, and the
declaration went into the key with CU13, so this walks a checkout, builds the
graph, and writes down what the model asks for. One thing is added that the
model does not carry: the **environment**, which no commit pins — a checkout
names its nodes identically against torch 2.3 and 2.6.
"""

import argparse
import contextlib
import importlib
import io
import json
import os
import pathlib
import sys
import tempfile


def fingerprint(implementation):
    """The digest of the code that implements it, or `None`.

    soma-next computes this only for what is `.cached()`, because parsing an
    AST is paid where it is used. Here every node is worth one: a node nobody
    caches is still a node somebody edited.
    """
    from soma_next import _fingerprint

    try:
        return _fingerprint.digest(type(implementation))
    except Exception:
        # `CannotVersion` for anything with no source to read — a builtin, a
        # class typed into a REPL. Not knowing is a fact, not a failure.
        return None


def reaches(implementation):
    """Los ficheros de los que está hecho un nodo, y dónde para la cuenta.

    Un nodo **es** su clase, y por eso `written_where` enseña una sola: es lo
    que quiere quien pincha un nodo. Pero una red se escribe muchas veces en
    cuatro módulos que se juntan en un `__init__`, y de esos cuatro
    `inspect.getsourcefile` sólo sabe uno. Los otros tres no estaban en ninguna
    parte de la respuesta.

    Estaban calculados, eso sí. La huella de soma-next recorre el cierre
    transitivo —las bases, los globales que el código nombra, las clases que
    compone— y hasta ahora lo tiraba y devolvía ocho caracteres de sha256.
    `_fingerprint.bill` es ese mismo recorrido dicho en voz alta, así que esto
    no es un segundo modelo de qué depende de qué: es el primero, leído.

    Van dos listas y no una porque son dos cosas distintas:

    - `files`, lo que se puede abrir: agrupado **por fichero**, porque un
      fichero suele llevar cuatro clases y lo que se edita es el fichero.
    - `stops`, donde el recorrido se para a propósito: una distribución
      instalada, un módulo entero. No tienen fichero, y ofrecer uno hacia el
      `site-packages` de alguien sería decir que no para ahí.

    Sin fuente ninguna de las dos: un árbol de ficheros de cuarenta commits con
    el contenido dentro es casi toda la respuesta y nada de ella leída. Se pide
    por su ruta cuando alguien abre uno.
    """
    from soma_next import _fingerprint

    try:
        billed = _fingerprint.bill(type(implementation))
    except Exception:
        # La misma ausencia que `fingerprint` dice arriba: una clase sin fuente
        # que leer no tiene recorrido del que hablar.
        return None

    files, stops = {}, []
    for one in billed:
        if one["kind"] != "yours" or not one["file"]:
            if one["kind"] in ("installed", "module"):
                stops.append(
                    {
                        "called": one["called"],
                        "module": one["module"],
                        "kind": one["kind"],
                        "version": one["version"],
                    }
                )
            continue
        # Relativa al checkout, por lo mismo que `written_where`: la absoluta es
        # el worktree temporal en el que corrió esto y no le dice nada a nadie
        # media hora después.
        where = os.path.relpath(one["file"], os.getcwd())
        files.setdefault(where, []).append(
            {"called": one["called"], "line": one["line"], "lines": one["lines"]}
        )

    return {
        "files": [
            {"file": where, "defs": sorted(defs, key=lambda one: one["line"])}
            for where, defs in sorted(files.items())
        ],
        "stops": stops,
    }


#: Past this, a class is not something anybody reads in a panel — it is a file
#: to open. The snapshot keeps a name and a line number instead.
MOST = 8000


def written_where(implementation):
    """The source of the class behind a node, and where it lives.

    A node **is** its class, so this is the thing somebody actually wants when
    they click one. It is the class and not the module: the file may hold four
    of them and the other three are somebody else's business.

    Read here because here is the only place the object exists — a snapshot is
    read back long after the process that made it is gone, and a checkout of
    that commit may no longer be on disk.
    """
    import inspect

    try:
        at = inspect.getsourcefile(type(implementation))
        source, line = inspect.getsourcelines(type(implementation))
    except (OSError, TypeError):
        # A class defined in a cell or by `exec` has no source to read. Said as
        # an absence, which is what `UNVERSIONED` says about the same class.
        return None

    text = "".join(source)
    return {
        # Relative to the checkout, because an absolute path is the temporary
        # worktree this ran in and means nothing to anybody afterwards.
        "file": os.path.relpath(at, os.getcwd()) if at else None,
        "line": line,
        "source": text if len(text) <= MOST else None,
        "lines": len(source),
    }


#: Cuánto se baja dentro de un nodo. Dos niveles llegan a `router.gru` y paran
#: ahí, que es donde deja de decir algo: por debajo son las piezas de torch, y
#: leerlas como catorce cosas convierte el dibujo en uno que nadie mira dos
#: veces. Es la misma razón por la que la figura de soma-next no abre un bloque
#: que todo el mundo reconoce.
DEEP = 2


def parts_of(implementation, deep=DEEP):
    """De qué está hecho un nodo, leído sin correr nada.

    `soma_next.torch.architecture` responde mejor a esto —ve hasta las
    operaciones que no son módulos, una conexión residual entre ellas— y para
    conseguirlo **corre el grafo** con una entrada de ejemplo. Aquí no se corre
    nada nunca, así que lo que se puede leer es lo que `__init__` construyó: la
    composición **declarada**. Es la misma categoría que todo lo demás de este
    lado, y por eso encaja: imprimir una declaración no es observarla.

    Lo que se saca de aquí es lo que un nodo esconde. `Pure` es una caja con un
    enrutador dentro, y ese enrutador es la pieza de la que va el experimento;
    dibujar la caja y callarse lo de dentro es dibujar el envoltorio.
    """
    try:
        import torch
    except ImportError:
        # Un grafo sin torch no tiene módulos dentro que mirar, y eso no es un
        # fallo: es un grafo de otra clase.
        return []

    said = []

    def walk(holder, prefix, left):
        if left <= 0:
            return
        children = (
            holder.named_children()
            if isinstance(holder, torch.nn.Module)
            else [
                (who, what)
                for who, what in vars(holder).items()
                if isinstance(what, torch.nn.Module)
            ]
        )
        for who, what in children:
            path = f"{prefix}{who}"
            kind = type(what)
            where = None
            # La fuente sólo de lo que escribió alguien de este lado. `GRU` y
            # `Linear` son de torch, su fuente son cientos de líneas que nadie
            # va a leer aquí, y el nombre ya dice todo lo que hace falta.
            if not kind.__module__.startswith("torch."):
                where = written_where(what)
            said.append(
                {
                    "path": path,
                    "kind": kind.__name__,
                    "module": kind.__module__,
                    **(where or {"file": None, "line": 0, "source": None, "lines": 0}),
                }
            )
            walk(what, f"{path}.", left - 1)

    walk(implementation, "", deep)
    return said


def environment():
    """The versions this graph was actually built against.

    Not the whole `pip list`: what `sys.modules` holds once `build()` has run,
    which is what the graph reached for and nothing else. Installing something
    unrelated does not move it.

    This is the axis git does not cover. The probe imports the **checkout's**
    code, but its dependencies come from the interpreter outside it, so two
    commits with identical code name their nodes identically whether they ran
    against torch 2.3 or 2.6. Recorded here, and deliberately **not** in the
    key a snapshot is remembered under: that is what lets a probe from last
    month and one from today disagree out loud rather than quietly.
    """
    import importlib.metadata as about

    said = {"python": ".".join(str(n) for n in sys.version_info[:3])}
    known = about.packages_distributions()
    wanted = {top.split(".")[0] for top in sys.modules}
    for module in wanted:
        for distribution in known.get(module, ()):
            try:
                said[distribution] = about.version(distribution)
            except about.PackageNotFoundError:
                pass
    # The engine itself, which an editable install leaves out of that map.
    try:
        said["soma-next"] = about.version("soma-next")
    except about.PackageNotFoundError:
        pass
    return dict(sorted(said.items()))


def built_by(where):
    """The `module:function` the config named, imported and called."""
    return declares(where)()


def declares(where):
    """The function itself, before calling it.

    Aparte de `built_by` porque hace falta **la función** y no sólo el grafo: lo
    que declara la topología —los `>>` y los `|`— está en su cuerpo, y es lo
    único de un grafo que no se puede leer nodo a nodo. Cada nodo dice qué hace
    y ninguno dice cómo se conectan.
    """
    module, _, name = where.partition(":")
    if not module or not name:
        raise SystemExit(f"`build` should read `module:function`, and read `{where}`")
    return getattr(importlib.import_module(module), name)


def declaring(where):
    """El código que declara el grafo, con su fichero y su línea.

    Un fallo aquí no es un fallo del sondeo: una función definida en una celda,
    o compuesta en tiempo de ejecución, no tiene fuente que leer, y eso es lo
    mismo que `UNVERSIONED` nombra un nivel más abajo. Se dice que no está en
    vez de romper lo demás.
    """
    import inspect

    try:
        made = declares(where)
        source, line = inspect.getsourcelines(made)
        return {
            "file": os.path.relpath(inspect.getsourcefile(made)),
            "line": line,
            "lines": len(source),
            "source": "".join(source),
        }
    except (OSError, TypeError, ValueError, SystemExit):
        return None


def architecture(graph):
    """Los cinco hechos ortogonales de un grafo, aparte de qué calcula cada nodo.

    El cuaderno de soma-next lo dice en una tabla: `Graph` dice **qué** existe,
    el catálogo **quién** lo ejecuta, la colocación **dónde**, el plan
    **cuándo** y la memoria **qué se recuerda** de cada nodo. Confundirlos es el
    error fácil, y un dibujo que sólo enseña el primero deja fuera cuatro.

    Van en claves separadas y no fundidos en un campo por nodo, por lo que el
    mismo cuaderno dice del relleno de su figura: tres hechos no caben en un
    color, e inventarles una precedencia esconde dos de los tres.
    """
    return {
        "identities": graph.identities(),
        "hosts": graph.hosts(),
        "devices": graph.devices(),
        "cached": sorted(graph.cached()),
        "frozen": sorted(graph.frozen()),
        "roots": sorted(graph.roots()),
        "leaves": sorted(graph.leaves()),
        # El orden en que correrían, que es el «cuándo» y no el «qué alimenta a
        # qué». Para un grafo serie-paralelo los dos coinciden; para uno que no
        # lo es, dejan de coincidir, y ahí es donde hace falta tener los dos.
        "order": list(graph.topological_sort()),
    }


def seen(graph, store, given):
    """This graph, as data: soma-next's own snapshot plus what only we look at.

    The names, shape, state, salt, fingerprints and the readable declarations
    all come from `foreseen.snapshot`, which is the model's and not ours — if
    what a name is made of changes again, it changes there and this keeps
    working. What is added is `environment`, the axis git does not have.

    A fingerprint is written for **every** node first. soma-next computes them
    only for what is `.cached()`, because parsing an AST is paid where it is
    used; here a node nobody caches is still a node somebody edited.
    """
    from soma_next import foreseen

    code = {}
    inside = {}
    reached = {}
    for node_id in graph.nodes():
        implementation = graph.implementation(node_id)
        if said := fingerprint(implementation):
            graph.written_as(node_id, said)
        if where := written_where(implementation):
            code[node_id] = where
        if parts := parts_of(implementation):
            inside[node_id] = parts
        if made := reaches(implementation):
            reached[node_id] = made

    return {
        "snapshot": foreseen.snapshot(graph, given, store=store),
        "code": code,
        "inside": inside,
        "reaches": reached,
        "architecture": architecture(graph),
        "mapped": sorted(graph.mapped_nodes()),
        # Empty against a scratch directory, and honestly so: with no real
        # store there is nothing kept to not have to compute.
        "unneeded": sorted(foreseen.unneeded(graph, given, store=store)),
    }


def compared(before, after):
    """What the edit did, as `{node: [finding, ...]}`.

    `foreseen.changes` decides all of it — the declaration included, since it
    went into the key. This used to fold the declaration into the fingerprint
    so that `STALE` would reach it; doing that now would be a lie in the
    loudest place there is, because a name that moved is a cache that **misses**
    and nothing stale is served.

    What is still added is only the reading: the before and after of a
    declaration, in the words somebody typed.
    """
    from soma_next import foreseen

    found = foreseen.changes(before["snapshot"], after["snapshot"])
    # Both sides or nothing. A node whose declaration cannot be written down is
    # **absent** from `declared`, and that absence is not "it takes no
    # arguments" — `shape` already said so, and saying it again here in words
    # would be inventing an answer out of a silence.
    said = (before["snapshot"].get("declared", {}), after["snapshot"].get("declared", {}))
    moved = {
        node: [was, is_]
        for node, was in said[0].items()
        if (is_ := said[1].get(node)) is not None and was != is_
    }
    return {"findings": found, "declared": moved}


def checked(build, node, given, store):
    """Whether an edit survives four questions, from the cheapest to the dearest.

    A fork is a commit, and a commit that does not import is a variant nobody
    can measure. Each of these can only be asked once the one before passed, so
    the first failure is the answer and the rest say why they were not run.

    Three things come back, not one: a line per question for somebody to read,
    **diagnostics** with a line and a column so an editor can underline them
    where they happened, and everything the code **printed**. That last one is
    not a nicety — the way anybody finds out what a `forward` is really doing is
    to put a `print` in it, and a check that swallowed those would be asking
    people to debug blind.
    """
    said, marks = [], []
    printed = io.StringIO()

    module, _, _ = build.partition(":")
    at = pathlib.Path(module.replace(".", "/") + ".py")

    # 1. It parses. Free, and catches a typo without importing anything.
    try:
        compile(at.read_text(), str(at), "exec")
        said.append({"what": "sintaxis", "ok": True, "said": "compila"})
    except SyntaxError as why:
        marks.append(
            {
                "line": why.lineno or 1,
                "col": why.offset or 1,
                "severity": "error",
                "message": f"{type(why).__name__}: {why.msg}",
                "from": "sintaxis",
            }
        )
        said.append({"what": "sintaxis", "ok": False, "said": f"{type(why).__name__}: {why.msg}"})
        return {"checks": said, "diagnostics": marks, "output": ""}

    # 2. Whatever linter is on the machine. Found, never installed.
    lint, complaints = linted(at)
    said.append(lint)
    marks.extend(complaints)

    # 3. The graph still builds. This is what catches the edit that renames a
    #    class and leaves `build()` calling the old name — invisible to a
    #    linter, and a commit nobody can run.
    graph = None
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        try:
            graph = built_by(build)
            said.append(
                {
                    "what": "el grafo construye",
                    "ok": True,
                    "said": f"{len(graph.nodes())} nodos: {' · '.join(graph.nodes())}",
                }
            )
        except Exception as why:
            said.append(
                {
                    "what": "el grafo construye",
                    "ok": False,
                    "said": f"{type(why).__name__}: {why}",
                }
            )
            marks.extend(marked(inside(why, at.name), f"{type(why).__name__}: {why}", "construir"))

    if graph is None:
        return {"checks": said, "diagnostics": marks, "output": printed.getvalue()}

    # 4. The node against the real value its predecessors left in the store.
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        ran_it = ran(graph, node, given, store)
    said.append(ran_it)
    if not ran_it.get("ok", True):
        marks.extend(marked(ran_it.get("at"), ran_it["said"], "ejecución"))

    return {"checks": said, "diagnostics": marks, "output": printed.getvalue()}


def inside(why, only):
    """The line the traceback last passed through in the file being checked.

    A traceback runs through soma-next, through the engine and through Python's
    own machinery, and none of that is anybody's edit. Only a frame in the file
    under the cursor can be underlined; with none, the message is still said in
    the summary and simply has nowhere to point.
    """
    import traceback

    # Only frames in that one file. Without the filter the last frame of any
    # traceback wins, and the last frame is nearly always somebody else's
    # library — which is how this first reported line 151 of a file with 25.
    if only is None:
        return None
    line = None
    for frame in traceback.extract_tb(why.__traceback__):
        if pathlib.Path(frame.filename).name == only:
            line = frame.lineno
    return line


def marked(line, message, whence):
    """One underline, or nothing when there is nowhere to put it."""
    return (
        [{"line": line, "col": 1, "severity": "error", "message": message, "from": whence}]
        if line
        else []
    )


def beside_the_interpreter(name):
    """A tool installed into this environment, or on the path, or nowhere.

    Beside the interpreter **first**: a probe runs under a project's own
    `.venv/bin/python`, and `.venv/bin` is not on the `PATH` of a subprocess
    somebody else spawned. Looking only at `PATH` finds the system's linter or
    none at all, and both are the wrong answer about this project.
    """
    import shutil

    mine = pathlib.Path(sys.executable).parent / name
    return str(mine) if mine.exists() else shutil.which(name)


def linted(at):
    """`ruff`, then `pyflakes`, then nothing — whatever is already here.

    Never installed, only found: a check that reaches for the network to answer
    fails on a machine without one and blames the code for it.
    """
    import subprocess

    found = beside_the_interpreter("ruff")
    if found:
        done = subprocess.run(
            [found, "check", "--no-cache", "--output-format", "json", str(at)],
            capture_output=True,
            text=True,
        )
        try:
            complaints = json.loads(done.stdout or "[]")
        except json.JSONDecodeError:
            complaints = []
        marks = [
            {
                "line": one["location"]["row"],
                "col": one["location"]["column"],
                "severity": "warning",
                "message": f"{one.get('code') or ''} {one['message']}".strip(),
                "from": "ruff",
            }
            for one in complaints
        ]
        return {
            "what": "ruff",
            "ok": not marks,
            "said": f"{len(marks)} queja(s)" if marks else "sin quejas",
        }, marks

    found = beside_the_interpreter("pyflakes")
    if found:
        done = subprocess.run([found, str(at)], capture_output=True, text=True)
        return {
            "what": "pyflakes",
            "ok": done.returncode == 0,
            "said": shortened((done.stdout + done.stderr).strip() or "sin quejas", 400),
        }, []

    return {
        "what": "lint",
        "ok": True,
        "skipped": True,
        "said": "ni ruff ni pyflakes en esta máquina",
    }, []


def prettified(source):
    """`ruff format` over one class, or the same text back.

    Refused rather than half-done when there is no formatter: handing back
    something that looks formatted and is not is worse than a button that says
    it cannot.
    """
    import subprocess

    found = beside_the_interpreter("ruff")
    if not found:
        return {"ok": False, "said": "no hay ruff en esta máquina", "source": source}
    done = subprocess.run(
        [found, "format", "--no-cache", "-"], input=source, capture_output=True, text=True
    )
    if done.returncode != 0:
        return {"ok": False, "said": done.stderr.strip(), "source": source}
    return {"ok": True, "said": "formateado", "source": done.stdout}


def ran(graph, node, given, store):
    """This node, alone, on what its predecessors actually left in the store.

    Not invented data: the engine keeps every `.cached()` node's answer under
    the name of its recipe, so the value the node **would** receive is already
    on disk — if somebody ran the graph, with a store, on this input. When it is
    not there this says so rather than making something up: a green light from a
    fabricated tensor is worse than no light.
    """
    from soma_next import Store, foreseen

    what = {"what": "corre con datos reales", "ok": True}
    if store is None:
        return {**what, "skipped": True, "said": "sin --store no hay nada guardado que darle"}

    shape = foreseen.snapshot(graph, given, store=store)["shape"]
    if node not in shape:
        return {**what, "ok": False, "said": f"`{node}` no está en el grafo que acaba de construir"}
    feeding = shape[node][2]

    kept = Store(store)
    names = foreseen.names(graph, given, store=store)
    values = {}
    for one in feeding:
        if one not in names:
            return {**what, "skipped": True, "said": f"`{one}` no se puede nombrar por adelantado"}
        if (had := kept.recall(f"value:{names[one]}")) is None:
            return {
                **what,
                "skipped": True,
                "said": f"nada guardado para `{one}`: corre el grafo con --store una vez",
            }
        values[one] = had

    # The engine's own fan-in rule: one thing arrives as itself, several as a
    # map keyed by whoever produced each.
    if not feeding:
        arriving = given
    elif len(feeding) == 1:
        arriving = values[feeding[0]]
    else:
        arriving = values

    # Called **directly**, and not through a graph of one, for the traceback.
    # The engine catches what a node raises and raises its own in its place, so
    # the frame somebody actually wrote is gone by the time it comes back — and
    # a check that cannot point at the line is half a check. `Ctx()` is the
    # honest stand-in: no device, which is exactly "wherever it lands".
    from soma_next import Ctx

    try:
        out = graph.implementation(node).forward(arriving, Ctx())
    except Exception as why:
        # Resolved here, where the traceback still exists. An exception object
        # does not survive a JSON dump.
        return {
            **what,
            "ok": False,
            "said": f"{type(why).__name__}: {why}",
            "at": inside(why, module_of(graph, node)),
        }
    return {**what, "said": f"devolvió {type(out).__name__}: {shortened(repr(out))}"}


def module_of(graph, node):
    """The file the node's class lives in, so only its frames are underlined."""
    import inspect

    try:
        return pathlib.Path(inspect.getsourcefile(type(graph.implementation(node)))).name
    except (OSError, TypeError):
        return None


def shortened(text, most=160):
    return text if len(text) <= most else text[:most] + "…"


def main():
    parsing = argparse.ArgumentParser(description=__doc__)
    parsing.add_argument("--compare", help="a file: snapshots by commit, and pairs")
    parsing.add_argument("--check", help="a node id: does this checkout survive it")
    parsing.add_argument("--format", action="store_true", help="prettify stdin and stop")
    parsing.add_argument("--build", help="module:function")
    parsing.add_argument("--commit", default="", help="what this checkout is")
    parsing.add_argument("--store", default=None, help="a real store, if there is one")
    parsing.add_argument("--input", default=None, help="a JSON file: the real input")
    said = parsing.parse_args()

    # Comparing needs no checkout and no graph: two files and the model. Every
    # consecutive pair in one call, because a walk of ten commits is nine
    # comparisons and nine interpreters would be the slowest part of it.
    if said.compare:
        # Snapshots by commit and the pairs to compare, rather than a list read
        # two at a time. A step is an **edge**: with three branches off one
        # commit, adjacent entries in a list are three different lines of
        # exploration, and comparing them would answer about an edit nobody made.
        asked = json.loads(pathlib.Path(said.compare).read_text())
        taken = asked["snapshots"]
        json.dump(
            [compared(taken[older], taken[newer]) for older, newer in asked["pairs"]],
            sys.stdout,
        )
        return
    if not said.build:
        raise SystemExit("one of --build or --compare")

    # Never a `.pyc`, and it is not tidiness. Python validates cached bytecode
    # by mtime and size, and `git checkout` stamps mtime with *now*: two
    # checkouts a second apart of a file that changed without changing length —
    # `Classify(1.0)` to `Classify(2.0)` — leave the old bytecode looking
    # valid. The probe would then read a graph nobody has committed. It also
    # keeps a `__pycache__` out of somebody's worktree.
    sys.dont_write_bytecode = True
    sys.path.insert(0, "")

    if said.format:
        json.dump(prettified(sys.stdin.read()), sys.stdout)
        return

    if said.check:
        given = None
        if said.input is not None:
            given = json.loads(pathlib.Path(said.input).read_text())
        json.dump(checked(said.build, said.check, given, said.store), sys.stdout)
        return

    graph = built_by(said.build)

    # The root is the one node named by **content**, so what is handed over
    # here decides whether these are the real names. Without an input it hashes
    # `Null`: the same sentinel on both sides of a comparison, so a difference
    # under it is still attributable to the recipe — but no name matches one a
    # run produced, and so nothing can be looked up in a store.
    given = None
    if said.input is not None:
        with open(said.input) as file:
            given = json.load(file)

    # A store is needed to name anything at all — the core has no algorithm to
    # hash with and takes one from the keeper. An empty directory names exactly
    # what a real one would; what it cannot say is which of those names is
    # already there, so `unneeded` comes back empty and honestly so.
    with tempfile.TemporaryDirectory() as scratch:
        about = seen(graph, said.store or scratch, given)

    json.dump(
        {
            "commit": said.commit,
            "built_from": said.build,
            "declaring": declaring(said.build),
            "environment": environment(),
            "input": "sentinel" if said.input is None else said.input,
            **about,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
