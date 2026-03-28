"""Search space descriptors for hyperparameter optimization."""


class SearchDescriptor:
    """Descriptor that defines a searchable hyperparameter.

    Used as a class variable in Filter subclasses::

        class MyFilter(Filter):
            lr: float = search(0.001, 0.1, scale="log")
            kernel: str = search(choices=["rbf", "linear"])
            use_bias: bool = search()  # auto-detect: Categorical[True, False]
    """

    def __init__(self, low=None, high=None, scale="linear", choices=None, default=None):
        self.low = low
        self.high = high
        self.scale = scale
        self.choices = choices
        self.default = default
        self.field_name = None  # set by FilterMeta

    def to_dict(self):
        """Convert to a dict for passing to Rust."""
        if self.choices is not None:
            return {
                "type": "categorical",
                "name": self.field_name,
                "choices": self.choices,
            }
        elif self.low is not None and self.high is not None:
            # Detect int vs float from bounds
            if isinstance(self.low, int) and isinstance(self.high, int):
                return {
                    "type": "int",
                    "name": self.field_name,
                    "low": self.low,
                    "high": self.high,
                }
            return {
                "type": "float",
                "name": self.field_name,
                "low": float(self.low),
                "high": float(self.high),
                "scale": self.scale,
            }
        else:
            # Auto: bool → categorical
            return {
                "type": "categorical",
                "name": self.field_name,
                "choices": [True, False],
            }

    def __set_name__(self, owner, name):
        self.field_name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.field_name, self.default)

    def __set__(self, obj, value):
        obj.__dict__[self.field_name] = value


def search(low=None, high=None, scale="linear", choices=None, default=None):
    """Define a searchable hyperparameter.

    Args:
        low: Lower bound (for float/int ranges).
        high: Upper bound (for float/int ranges).
        scale: "linear", "log", or "reverse_log".
        choices: List of discrete choices (for categorical).
        default: Default value when not in a Study.

    Returns:
        SearchDescriptor instance.

    Examples::

        scale: float = search(0.1, 10.0, scale="log")
        method: str = search(choices=["standard", "robust"])
        enabled: bool = search()  # auto: Categorical[True, False]
    """
    return SearchDescriptor(low=low, high=high, scale=scale, choices=choices, default=default)
