"""What the Rust extension offers, said in Python's types.

Hand-written, because there is no stub generator that reads `#[pymethods]` and
tells the truth about a `PyResult<Option<PyObject>>`. A hand-written stub can
lie, so `tests/test_stubs.py` checks it against the module that was **built**:
same names, same methods, same parameters, same defaults. What no test can check
is whether a type is *right* — that part is read from `python/src/*.rs`, where
`PyResult<T>` is `T` and `Option<T>` is `T | None`.

Two PyO3 facts worth knowing before editing this: a `#[new]`'s signature lands on
the **type** (`cls.__text_signature__`) and not on `__new__`, so the test
compares constructors there; and a `#[staticmethod]` arrives as a plain
`builtin_function_or_method` with no `self`, which is why the ones below carry
`@staticmethod` and the instance methods do not.
"""

import builtins
from typing import Any, Callable, Iterator

__version__: str

# ── The graph ────────────────────────────────────────────────────────────────

class Ctx:
    """What a node knows beyond its input."""

    def __init__(self, device: str | None = None) -> None: ...
    @property
    def device(self) -> str | None:
        """Where this node was said to run, written the way torch writes it."""

    def __repr__(self) -> str: ...

class Graph:
    """The core's topology plus the implementations."""

    def __init__(self) -> None: ...

    # Building it
    def node(self, *args: Any) -> str:
        """`node(obj)` names it for you, `node("id", obj)` you name it."""

    def edge(self, source: str, target: str) -> None: ...
    def place(self, node_id: str, device: str) -> None: ...
    def place_at(self, node_id: str, host: str) -> None: ...

    # What is remembered
    def freeze(self, node_id: str, state: str | None = None) -> None: ...
    def mapped(self, node_id: str) -> None: ...
    def cache(self, node_id: str, salt: str | None = None) -> None: ...
    def declared_as(self, node_id: str, declaration: str) -> None: ...
    def written_as(self, node_id: str, fingerprint: str) -> None: ...

    # Reading it back. `frozen` and `cached` carry something beside the id —
    # a state digest, a salt — and either may legitimately be absent, which is
    # why the values are optional and the two below them are not.
    def frozen(self) -> dict[str, str | None]: ...
    def cached(self) -> dict[str, str | None]: ...
    def identities(self) -> dict[str, str]: ...
    def declarations(self) -> dict[str, str]: ...
    def fingerprints(self) -> dict[str, str]: ...
    def mapped_nodes(self) -> list[str]: ...
    def hosts(self) -> dict[str, str]: ...
    def devices(self) -> dict[str, str]: ...

    # Topology
    def nodes(self) -> list[str]: ...
    def edges(self) -> list[tuple[str, str]]: ...
    def roots(self) -> list[str]: ...
    def leaves(self) -> list[str]: ...
    def predecessors(self, node_id: str) -> list[str]: ...
    def successors(self, node_id: str) -> list[str]: ...
    def topological_sort(self) -> list[str]: ...
    def implementation(self, node_id: str) -> Any:
        """The object registered under that id — `None` if there is none.

        `Any` rather than `Any | None`: what comes back is the caller's own
        object and unchecked either way, and the miss is a wrong id, which is a
        bug at the call site."""

    # The shape of an execution
    def plan(self) -> str:
        """The decided shape as `Debug` text, for a person."""

    def plan_json(self) -> str:
        """The same shape as data, for whoever draws it."""

    def foreseen_json(self, input: Any | None = None, *, store: Any | None = None) -> str:
        """`{"keys": {node: name}, "unneeded": [node, ...]}`, nothing executed."""

    # Running it
    def forward(
        self,
        input: Any | None = None,
        *,
        broker: Broker | None = None,
        store: Store | str | None = None,
        watching: Recorder | Callable[[dict[str, Any]], None]
        | list[Recorder | Callable[[dict[str, Any]], None]]
        | None = None,
        stamping: dict[str, str] | None = None,
    ) -> Any: ...
    def __len__(self) -> int: ...
    def __contains__(self, node_id: str) -> bool: ...
    def __repr__(self) -> str: ...

class Opaque:
    """Marks a value so it crosses the graph untouched."""

    def __init__(self, value: Any) -> None: ...
    @property
    def value(self) -> Any: ...
    def __repr__(self) -> str: ...

# ── Keeping things ───────────────────────────────────────────────────────────

class Bound:
    """A name, and what it points at. Frozen: the store is what writes these."""

    @property
    def name(self) -> str: ...
    @property
    def digest(self) -> str: ...
    @property
    def meta(self) -> list[tuple[str, str]]:
        """What was said beside it, in the order it was said."""

    @property
    def when(self) -> int:
        """Seconds since the epoch."""

    def __repr__(self) -> str: ...

class Store:
    """Something that keeps bytes by their content, and names that point at them."""

    def __init__(self, where_: str) -> None: ...
    @staticmethod
    def on_bucket(
        endpoint: str,
        bucket: str,
        *,
        region: str = "us-east-1",
        key: str | None = None,
        secret: str | None = None,
        hosted: bool = False,
    ) -> Store: ...

    # Bytes, by their content
    def put(self, bytes: bytes) -> str:
        """Writes them and answers with their digest."""

    def get(self, digest: str) -> bytes | None: ...

    # Names, pointing at bytes
    def bind(self, name: str, digest: str, meta: dict[str, str] | None = None) -> None: ...
    def claim(self, name: str, digest: str, meta: dict[str, str] | None = None) -> bool:
        """Binds it only if nobody has: `True` means this caller got it."""

    def resolve(self, name: str) -> Bound | None: ...
    def bound(self) -> list[Bound]: ...

    # The two together, through the codecs
    def keep(self, name: str, what: Any, meta: dict[str, str] | None = None) -> str: ...
    def recall(self, name: str) -> Any | None: ...
    def __repr__(self) -> str: ...

    # `Store(where)` for a person to read, `Store()` for a key: a declaration
    # says what, and where is the machine. See `somatize._declaration.DECLARED`.
    def __soma_declared__(self) -> str: ...

def codec(
    kind: str,
    of_type: type,
    *,
    dump: Callable[[Any], bytes],
    load: Callable[[bytes], Any],
) -> None:
    """Says how objects of a type are written down and read back."""

def codecs_registered() -> list[str]:
    """What has a codec registered today, in the order they were registered."""

def reasoning(store: Store, tree: str) -> str:
    """The reasoning of that investigation as JSON: its moves, what was said,
    and what folds. Read `somatize.reasoning` instead of this."""

def reasoning_covers(store: Store, tree: str, by: list[str]) -> list[str]:
    """What a scope with those roots reaches, in the order they were made."""

# ── The record ───────────────────────────────────────────────────────────────

class Recorder:
    """Writes down what happened, one record per `forward`."""

    def __init__(
        self,
        store: Store,
        *,
        run: str | None = None,
        summarising: list[str] | None = None,
    ) -> None: ...
    @property
    def run(self) -> str: ...
    def __call__(self, fact: dict[str, Any]) -> None: ...
    def __repr__(self) -> str: ...

# ── Data ─────────────────────────────────────────────────────────────────────

class Frame:
    """The batch of columns a source answered with, as Python sees it."""

    @property
    def rows(self) -> int: ...
    @property
    def columns(self) -> list[str]: ...
    def column(self, name: str) -> list[Any]:
        """One column as a list of Python values; a missing value is `None`."""

    def ipc(self) -> bytes:
        """The Arrow IPC bytes, which is what every dataframe library reads."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class Source:
    """A parquet file in a store, answering spans of rows."""

    def __init__(self, store: Store, name: str) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    def forward(self, input: Any) -> Frame: ...
    def __repr__(self) -> str: ...

# ── Health ───────────────────────────────────────────────────────────────────

class Thresholds:
    """The bounds a verdict is taken at."""

    def __init__(self, **over: float) -> None: ...
    def as_dict(self) -> dict[str, float]: ...
    def __repr__(self) -> str: ...

def verdict(seen: dict[str, Any], thresholds: Thresholds | None = None) -> list[str]:
    """What is wrong with these numbers, as a list of flag names."""

def leaning(
    drops: dict[str, float],
    thresholds: Thresholds | None = None,
) -> tuple[list[tuple[str, float, float]], list[str]]:
    """`(shares, flags)` — what each input is worth, and how that goes wrong.

    A share is `(name, drop, fraction)`.
    """

def family(flag: str) -> str:
    """Which family of trouble a flag belongs to, by name."""

def about(flag: str) -> str:
    """What a flag means and what to do about it, by name."""

# ── Searching ────────────────────────────────────────────────────────────────

class Space:
    """What is being searched over. Every method answers a new `Space`.

    `int` below is a method named after the builtin, which shadows it for the
    rest of the class body — so everything after it spells `builtins.int`. Not a
    style choice: without it a checker reads `-> int` as *this method*.
    """

    def __init__(self) -> None: ...
    def real(self, name: str, low: float, high: float, *, log: bool = False) -> Space: ...
    def int(self, name: str, low: builtins.int, high: builtins.int) -> Space: ...
    def choice(self, name: str, options: list[str]) -> Space: ...
    def read(self, said: str) -> Point:
        """The point that text was written down as."""

    def names(self) -> list[str]: ...
    def __len__(self) -> builtins.int: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Point:
    """One configuration. Reads like a mapping and is not one: it is frozen."""

    def keys(self) -> list[str]: ...
    def values(self) -> list[Any]: ...
    def items(self) -> list[tuple[str, Any]]: ...
    def __getitem__(self, name: str) -> Any: ...
    def __contains__(self, name: str) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Sampler:
    """Where to look for the next configuration."""

    @staticmethod
    def grid(steps: int = 5) -> Sampler: ...
    @staticmethod
    def random(*, seed: int = 0) -> Sampler: ...
    @staticmethod
    def halton(*, seed: int = 0) -> Sampler: ...
    @staticmethod
    def sobol(*, seed: int = 0) -> Sampler: ...
    @staticmethod
    def tpe(
        *,
        goal: str = "min",
        startup: int = 10,
        candidates: int = 24,
        quantile: float = 0.25,
        seed: int = 0,
    ) -> Sampler: ...
    def ask(
        self,
        space: Space,
        trial: int,
        seen: list[tuple[Point, float | None]] | None = None,
    ) -> Point | None:
        """The configuration for that trial index. A score of `None` means it is
        still running, which is kept away from without being voted on."""

    def total(self, space: Space) -> int | None:
        """How many points there are, when that is a finite number."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Pruner:
    """Whether a trial that is going badly is worth going on with."""

    @staticmethod
    def median(*, goal: str = "min", warmup: int = 0, startup: int = 1) -> Pruner: ...
    @staticmethod
    def percentile(
        p: float, *, goal: str = "min", warmup: int = 0, startup: int = 1
    ) -> Pruner: ...
    @staticmethod
    def threshold(*, lower: float | None = None, upper: float | None = None) -> Pruner: ...
    @staticmethod
    def diverged() -> Pruner: ...
    @staticmethod
    def patience(steps: int, *, min_delta: float = 0.0, goal: str = "min") -> Pruner: ...
    def verdict(self, mine: list[float], others: list[list[float]] | None = None) -> str | None:
        """Why to stop, or `None` to go on."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Partition:
    """How the samples are cut into folds."""

    @staticmethod
    def kfold(k: int, *, shuffle: int | None = None) -> Partition: ...
    @staticmethod
    def stratified(k: int, *, shuffle: int | None = None) -> Partition: ...
    @staticmethod
    def grouped(k: int) -> Partition: ...
    @staticmethod
    def stratified_grouped(k: int) -> Partition: ...
    @staticmethod
    def time_series(k: int, *, gap: int = 0) -> Partition: ...
    def folds(
        self,
        n: int,
        *,
        classes: list[int] | None = None,
        groups: list[int] | None = None,
    ) -> list[tuple[list[int], list[int]]]:
        """`(train, test)` index lists, one pair per fold."""

    @property
    def k(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ── Workers ──────────────────────────────────────────────────────────────────

class Broker:
    """Where the hosts of a graph are, and how to reach them."""

    def __init__(self, listing: dict[str, Any]) -> None: ...
    def wire_token(self, host: str) -> bytes: ...
    def provision(
        self, host: str, kind: str, id: str, blob: bytes, runtime: str
    ) -> None: ...
    def __repr__(self) -> str: ...

def serve(nodes: dict[str, Any]) -> None:
    """Serves slices with the catalog you pass it, until the client hangs up."""

def serve_provisioned(
    provision: Any,
    store: Store | str | None = None,
    reporting: float | None = None,
) -> None:
    """Serves slices with what the client sends it: the generic worker."""

def listen(
    addr: str,
    nodes: dict[str, Any],
    opened: Callable[[str], None] | None = None,
) -> None:
    """The same as `serve`, standing on an address. It does not return."""

def listen_provisioned(
    addr: str,
    provision: Any,
    opened: Callable[[str], None] | None = None,
    store: Store | str | None = None,
    reporting: float | None = None,
) -> None:
    """The same as `serve_provisioned`, standing on an address. It does not return."""
