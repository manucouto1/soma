"""Base Filter class for user-defined pipeline nodes."""

from soma.search import SearchDescriptor


class FilterMeta(type):
    """Metaclass that collects search() descriptors into _soma_search_space."""

    def __new__(mcs, name, bases, namespace):
        search_space = []

        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, SearchDescriptor):
                attr_value.field_name = attr_name
                search_space.append(attr_value.to_dict())

        namespace["_soma_search_space"] = search_space
        cls = super().__new__(mcs, name, bases, namespace)
        return cls


class Filter(metaclass=FilterMeta):
    """Base class for pipeline filters.

    A filter has three types of data:

    1. **Parameters** (constructor args / search spaces):
       - Passed via constructor: ``KNN(n_neighbors=5)``
       - Or defined as search spaces: ``n_neighbors = search(1, 15)``
       - Live on ``self`` — used in both ``fit()`` and ``forward()``
       - Cached by config_hash (change params → new cache key)

    2. **State** (returned by ``fit()``):
       - Learned from training data, returned as a dict
       - Passed to ``forward(x, state)`` as argument, NOT stored on self
       - Cached independently — same params + same data → same state

    3. **Internal attributes** (private, not parameters):
       - Prefix with ``_`` to mark as internal
       - Not included in config_hash or search space

    Example::

        from soma import Filter, search

        class KNN(Filter):
            # Search space parameters (optimizable)
            n_neighbors = search(1, 15)
            weights = search(choices=["uniform", "distance"])

            # Fixed parameter (not in search space)
            metric: str = "euclidean"

            def __init__(self, metric="euclidean", **kwargs):
                super().__init__(**kwargs)
                self.metric = metric

            def fit(self, x, y=None):
                # State: learned from data, returned as dict
                return {"train_x": x.tolist(), "train_y": y.tolist()}

            def forward(self, x, state):
                # self.n_neighbors → parameter (from constructor or search)
                # state["train_x"] → state (from fit)
                ...

    Metadata class attributes (optional)::

        _cacheable = True        # whether outputs can be cached (default True)
        _differentiable = False  # whether gradients can flow through (default False)
        _kind = "trainable"      # "trainable" (default) or "stateless"
        _stream_mode = "fixed"   # "fixed" (default), "evolving", or "barrier"
    """

    #: Schema version of this filter class. Bump in subclasses when
    #: constructor kwargs or saved-state layout changes; ``Graph.load``
    #: reads it from the manifest and warns on mismatch so users can
    #: plug a migration hook.
    class_version: int = 1

    def __init__(self, **kwargs):
        # Remember the constructor kwargs verbatim so checkpoints can
        # rebuild the filter via ``cls(**self.kwargs())``. Stored as a
        # plain dict on the instance — subclasses that need to keep a
        # different surface should override :meth:`kwargs`.
        self._init_kwargs: dict = dict(kwargs)

        # Set all kwargs as attributes (parameters)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Materialize search-descriptor defaults into __dict__ so they
        # behave like explicitly-passed kwargs everywhere — including in
        # the cache identity (an unset searchable param still shapes
        # behavior through its default).
        for klass in type(self).__mro__:
            for name, attr in vars(klass).items():
                if (
                    isinstance(attr, SearchDescriptor)
                    and name not in self.__dict__
                    and attr.default is not None
                ):
                    self.__dict__[name] = attr.default

    # ── Introspection (used by Graph.save / Graph.load) ──

    def kwargs(self) -> dict:
        """Return the constructor kwargs needed to rebuild this filter.

        Default: the dict captured by ``__init__``. Subclasses with
        non-serialisable runtime objects should override and return only
        the JSON-serialisable subset (everything else must be
        reconstructed inside the new instance, e.g. in ``build_module``).
        """
        return dict(getattr(self, "_init_kwargs", {}))

    @classmethod
    def class_path(cls) -> str:
        """Fully-qualified import path: ``"module.submodule.Class"``.

        ``Graph.load`` uses it to rebuild filter instances without the
        original Python objects in scope. The class must therefore be
        importable by name from a stable module path.
        """
        return f"{cls.__module__}.{cls.__qualname__}"

    def fit(self, x, y=None):
        """Learn state from training data.

        Override this to implement training logic. Return a dict
        with the learned state. This state will be passed to forward().

        Args:
            x: Input data (numpy array, list, etc.)
            y: Labels (optional, for supervised learning)

        Returns:
            dict: Learned state (e.g., {"mean": ..., "std": ...})
        """
        return {}

    def forward(self, x, state):
        """Transform data using learned state.

        Override this to implement the transformation. Use parameters
        from self (e.g., self.n_neighbors) and state from fit().

        Args:
            x: Input data to transform
            state: Dict returned by fit()

        Returns:
            Transformed data
        """
        return x

    # ── Fluent graph building ──

    def to(self, other):
        """Chain this filter to another. Returns a Chain.

        Usage::

            Scaler().to(PCA()).to(Model())
            Scaler().to([HeadA(), HeadB()]).collect(Ensemble())
        """
        from soma.chain import Chain, Fork
        if isinstance(other, list):
            return Chain([self, Fork.from_list(other)])
        return Chain([self, other])

    def __rshift__(self, other):
        """filter >> other → Chain"""
        from soma.chain import Chain, Fork
        if isinstance(other, (Chain, Fork)):
            return Chain([self]) >> other
        if isinstance(other, list):
            return Chain([self, Fork.from_list(other)])
        return Chain([self, other])

    def __or__(self, other):
        """filter | other → Fork (parallel branches)"""
        from soma.chain import Chain, Fork
        self_chain = Chain([self])
        other_chain = other if isinstance(other, Chain) else Chain([other])
        return Fork([self_chain, other_chain])
