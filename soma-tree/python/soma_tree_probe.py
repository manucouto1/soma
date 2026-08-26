"""Writes down what a graph is, at one commit, without running any of it.

This runs *inside* a worktree, against whatever soma that checkout imports.
It holds a graph; nothing else in `soma-tree` ever does. What it writes is the
contract — the part meant to outlive this file being rewritten in another
language — so a field is added here before it is read anywhere else.

What is decided here: nothing. `somatize.foreseen` owns the model, and the
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

    soma computes this only for what is `.cached()`, because parsing an
    AST is paid where it is used. Here every node is worth one: a node nobody
    caches is still a node somebody edited.
    """
    from somatize import _fingerprint

    try:
        return _fingerprint.digest(type(implementation))
    except Exception:
        # `CannotVersion` for anything with no source to read — a builtin, a
        # class typed into a REPL. Not knowing is a fact, not a failure.
        return None


def reaches(implementation):
    """The files a node is made of, and where the count stops.

    A node **is** its class, which is why `written_where` shows just one: it is
    what somebody clicking a node wants. But a network is often written across
    four modules joined in an `__init__`, and `inspect.getsourcefile` knows
    only one of the four.

    They were computed already. soma's fingerprint walks the transitive
    closure — the bases, the globals the code names, the classes it composes —
    and until now threw it away and returned eight characters of sha256.
    `_fingerprint.bill` is that same walk said out loud, so this is not a second
    model of what depends on what: it is the first one, read.

    Two lists and not one, because they are two things: `files`, what can be
    opened, grouped **by file** since a file usually holds four classes and
    what gets edited is the file; and `stops`, where the walk deliberately
    halts — an installed distribution, a whole module — which have no file, and
    offering one into somebody's `site-packages` would say it does not stop
    there.

    No source in either: a file tree of forty commits with the content inside
    is nearly all of the answer and none of it read.
    """
    from somatize import _fingerprint

    try:
        billed = _fingerprint.bill(type(implementation))
    except Exception:
        # The same absence `fingerprint` says above: a class with no source to
        # read has no walk to talk about.
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
        # Relative to the checkout, for the same reason as `written_where`: the
        # absolute one is the temporary worktree this ran in, and says nothing
        # to anybody half an hour later.
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


#: How far down inside a node to go. Two levels reach `router.gru` and stop
#: there, which is where it stops saying anything: below that are torch's own
#: pieces, and reading them as fourteen things makes a drawing nobody looks at
#: twice. The same reason soma's figure does not open a block everybody knows.
DEEP = 2


def parts_of(implementation, deep=DEEP):
    """What a node is made of, read without running anything.

    `somatize.torch.architecture` answers this better — it sees even the
    operations that are not modules, a residual connection among them — and to
    do it **runs the graph** with a sample input. Nothing is ever run here, so
    what can be read is what `__init__` built: the **declared** composition.
    Printing a declaration is not observing it, which is what puts it in the
    same category as everything else on this side.

    What comes out is what a node hides. `Pure` is a box with a router inside,
    and that router is what the experiment is about; drawing the box and
    keeping quiet about the inside draws the wrapper.
    """
    try:
        import torch
    except ImportError:
        # A graph without torch has no modules inside to look at, and that is
        # not a failure: it is another kind of graph.
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
            # Source only for what somebody on this side wrote. `GRU` and
            # `Linear` are torch's, their source is hundreds of lines nobody
            # will read here, and the name already says what is needed.
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
        said["somatize"] = about.version("somatize")
    except about.PackageNotFoundError:
        pass
    return dict(sorted(said.items()))


def built_by(where):
    """The `module:function` the config named, imported and called."""
    return declares(where)()


def declares(where):
    """The function itself, before calling it.

    Apart from `built_by` because **the function** is needed and not just the
    graph: what declares the topology — the `>>` and the `|` — is in its body,
    and it is the one part of a graph that cannot be read node by node.
    """
    module, _, name = where.partition(":")
    if not module or not name:
        raise SystemExit(f"`build` should read `module:function`, and read `{where}`")
    return getattr(importlib.import_module(module), name)


def declaring(where):
    """The code that declares the graph, with its file and its line.

    A failure here is not a failure of the probe: a function defined in a cell,
    or composed at runtime, has no source to read, which is the same absence
    `UNVERSIONED` names a level below. It says so rather than breaking the rest.
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
    """The orthogonal facts of a graph, apart from what each node computes.

    soma's notebook puts it in a table: `Graph` says **what** exists, the
    catalog **who** executes it, the placement **where**, the plan **when** and
    the memory **what is remembered** of each node. Confusing them is the easy
    mistake, and a drawing that shows only the first leaves four out.

    In separate keys and not merged into one field per node, for what that same
    notebook says about the fill of its figure: three facts do not fit in one
    colour, and inventing a precedence among them hides two of the three.
    """
    return {
        "identities": graph.identities(),
        "hosts": graph.hosts(),
        "devices": graph.devices(),
        "cached": sorted(graph.cached()),
        "frozen": sorted(graph.frozen()),
        "roots": sorted(graph.roots()),
        "leaves": sorted(graph.leaves()),
        # The order they would run in, which is the *when* and not the *what
        # feeds what*. For a series-parallel graph the two coincide; for one
        # that is not, they stop coinciding, and that is where both are needed.
        "order": list(graph.topological_sort()),
    }


def seen(graph, store, given):
    """This graph, as data: soma's own snapshot plus what only we look at.

    The names, shape, state, salt, fingerprints and the readable declarations
    all come from `foreseen.snapshot`, which is the model's and not ours — if
    what a name is made of changes again, it changes there and this keeps
    working. What is added is `environment`, the axis git does not have.

    A fingerprint is written for **every** node first. soma computes them
    only for what is `.cached()`, because parsing an AST is paid where it is
    used; here a node nobody caches is still a node somebody edited.
    """
    from somatize import foreseen

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
    from somatize import foreseen

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
        said.append({"what": "syntax", "ok": True, "said": "it compiles"})
    except SyntaxError as why:
        marks.append(
            {
                "line": why.lineno or 1,
                "col": why.offset or 1,
                "severity": "error",
                "message": f"{type(why).__name__}: {why.msg}",
                "from": "syntax",
            }
        )
        said.append({"what": "syntax", "ok": False, "said": f"{type(why).__name__}: {why.msg}"})
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
                    "what": "the graph builds",
                    "ok": True,
                    "said": f"{len(graph.nodes())} nodes: {' · '.join(graph.nodes())}",
                }
            )
        except Exception as why:
            said.append(
                {
                    "what": "the graph builds",
                    "ok": False,
                    "said": f"{type(why).__name__}: {why}",
                }
            )
            marks.extend(marked(inside(why, at.name), f"{type(why).__name__}: {why}", "build"))

    if graph is None:
        return {"checks": said, "diagnostics": marks, "output": printed.getvalue()}

    # 4. The node against the real value its predecessors left in the store.
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        ran_it = ran(graph, node, given, store)
    said.append(ran_it)
    if not ran_it.get("ok", True):
        marks.extend(marked(ran_it.get("at"), ran_it["said"], "execution"))

    return {"checks": said, "diagnostics": marks, "output": printed.getvalue()}


def inside(why, only):
    """The line the traceback last passed through in the file being checked.

    A traceback runs through soma, through the engine and through Python's
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
            "said": f"{len(marks)} complaint(s)" if marks else "no complaints",
        }, marks

    found = beside_the_interpreter("pyflakes")
    if found:
        done = subprocess.run([found, str(at)], capture_output=True, text=True)
        return {
            "what": "pyflakes",
            "ok": done.returncode == 0,
            "said": shortened((done.stdout + done.stderr).strip() or "no complaints", 400),
        }, []

    return {
        "what": "lint",
        "ok": True,
        "skipped": True,
        "said": "neither ruff nor pyflakes on this machine",
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
        return {"ok": False, "said": "there is no ruff on this machine", "source": source}
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
    from somatize import Store, foreseen

    what = {"what": "it runs on real data", "ok": True}
    if store is None:
        return {**what, "skipped": True, "said": "without --store there is nothing kept to hand it"}

    shape = foreseen.snapshot(graph, given, store=store)["shape"]
    if node not in shape:
        return {**what, "ok": False, "said": f"`{node}` is not in the graph it just built"}
    feeding = shape[node][2]

    kept = Store(store)
    names = foreseen.names(graph, given, store=store)
    values = {}
    for one in feeding:
        if one not in names:
            return {**what, "skipped": True, "said": f"`{one}` cannot be named in advance"}
        if (had := kept.recall(f"value:{names[one]}")) is None:
            return {
                **what,
                "skipped": True,
                "said": f"nothing kept for `{one}`: run the graph with --store once",
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
    from somatize import Ctx

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
    return {**what, "said": f"returned {type(out).__name__}: {shortened(repr(out))}"}


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
