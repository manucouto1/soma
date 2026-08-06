"""The AST behind ``>>`` and ``|``.

``Chain`` and ``Fork`` describe topology without materializing it, and
``Graph.somatize`` walks the tree into nodes and edges. They are internal:
a user writes the operators and never the constructors, which is why this
module is private and neither type is exported.

The operators are sugar for the linear case and nothing more. They cannot
express loops, branches, steps, optional edges or ``target=`` — two of the
five node kinds — so the graph methods stay the model and this stays a
shorthand.

Examples::

    from soma import Filter, Graph

    # Linear chain with >>
    g = Graph.somatize(Scaler() >> PCA() >> Model())

    # Fork with |, merged by the next step
    g = Graph.somatize(
        Scaler() >> (HeadA() | HeadB()) >> Ensemble()
    )

    # Nested branches
    g = Graph.somatize(
        (LoadA() >> NormA() | LoadB() >> NormB())
        >> Aggregate()
        >> Backbone()
        >> (ClassA() | ClassB())
    )
"""


class Chain:
    """A lazy linear sequence of steps (filters, forks, or nested chains).

    Created by ``filter >> other``.
    """

    __slots__ = ("steps",)

    def __init__(self, steps=None):
        self.steps = list(steps) if steps else []

    def __rshift__(self, other):
        """chain >> other"""
        if isinstance(other, Fork):
            return Chain([*self.steps, other])
        if isinstance(other, Chain):
            return Chain([*self.steps, *other.steps])
        if isinstance(other, list):
            return Chain([*self.steps, Fork.from_list(other)])
        # Filter
        return Chain([*self.steps, other])

    def __rrshift__(self, other):
        """other >> chain (when other is a Filter)"""
        return Chain([other, *self.steps])

    def __or__(self, other):
        """chain | other → Fork"""
        other_chain = other if isinstance(other, Chain) else Chain([other])
        return Fork([self, other_chain])

    def __repr__(self):
        return f"Chain({self.steps!r})"


class Fork:
    """Parallel branches, merged by whatever step follows.

    Created by ``chain | chain`` or ``filter | filter``.
    """

    __slots__ = ("branches",)

    def __init__(self, branches=None):
        self.branches = list(branches) if branches else []

    @classmethod
    def from_list(cls, items):
        """Create a Fork from a list of filters/chains."""
        branches = []
        for item in items:
            if isinstance(item, Chain):
                branches.append(item)
            elif isinstance(item, Fork):
                branches.extend(item.branches)
            else:
                branches.append(Chain([item]))
        return cls(branches)

    def __rshift__(self, other):
        """fork >> filter = auto-collect"""
        if isinstance(other, Fork):
            return Chain([self, other])
        if isinstance(other, Chain):
            return Chain([self, *other.steps])
        # Filter → collect
        return Chain([self, other])

    def __or__(self, other):
        """fork | other → add branch"""
        other_chain = other if isinstance(other, Chain) else Chain([other])
        return Fork([*self.branches, other_chain])

    def __repr__(self):
        return f"Fork({self.branches!r})"
