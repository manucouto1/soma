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

    Subclass this and implement fit() and forward().
    Use search() descriptors to define hyperparameter search spaces.

    Optional class attributes for metadata (read by the Soma runtime):

        _cacheable = True        # whether outputs can be cached (default True)
        _differentiable = False  # whether gradients can flow through (default False)
        _kind = "trainable"      # "trainable" (default) or "stateless"
        _stream_mode = "fixed"   # "fixed" (default), "evolving", or "barrier"

    Example::

        class MyScaler(Filter):
            _differentiable = True
            scale: float = search(0.1, 10.0, scale="log")

            def fit(self, x, y=None):
                return {"mean": x.mean(0), "std": x.std(0)}

            def forward(self, x, state):
                return (x - state["mean"]) / state["std"] * self.scale

        class DecisionTree(Filter):
            _differentiable = False
            _cacheable = True

            def fit(self, x, y=None):
                # train tree...
                return {"tree": tree}
    """

    def __init__(self, **kwargs):
        # Set all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set defaults from search descriptors
        for dim in self.__class__._soma_search_space:
            name = dim["name"]
            if not hasattr(self, name) and "default" in dim:
                setattr(self, name, dim["default"])

    def fit(self, x, y=None):
        """Learn state from training data. Override in subclass."""
        return {}

    def forward(self, x, state):
        """Transform data using learned state. Override in subclass."""
        return x
