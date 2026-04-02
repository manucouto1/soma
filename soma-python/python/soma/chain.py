"""Lazy graph builders: Chain, Fork, and operator overloads.

These types describe graph topology without materializing it.
Use ``Graph.somatize(chain)`` to turn them into an executable graph.

Examples::

    from soma import Filter, Graph

    # Linear chain with >>
    g = Graph.somatize(Scaler() >> PCA() >> Model())

    # Fork with |, auto-collect with >>
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

    # Method syntax
    g = Graph.somatize(
        Scaler().to(PCA()).to([
            PCA() >> ClassA(),
            UMAP() >> ClassB(),
        ]).collect(Ensemble())
    )
"""


class Chain:
    """A lazy linear sequence of steps (filters, forks, or nested chains).

    Created by ``filter >> other`` or ``filter.to(other)``.
    """

    __slots__ = ("steps",)

    def __init__(self, steps=None):
        self.steps = list(steps) if steps else []

    def to(self, other):
        """Append a step. Accepts a Filter, list (fork), Chain, or Fork."""
        if isinstance(other, list):
            return Chain([*self.steps, Fork.from_list(other)])
        if isinstance(other, (Chain, Fork)):
            return Chain([*self.steps, other])
        # Assume it's a Filter
        return Chain([*self.steps, other])

    def collect(self, filter):
        """After a fork, merge all branches into this filter."""
        # Find the last Fork in steps and wrap it
        return Chain([*self.steps, filter])

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
    """Parallel branches that will be merged by a subsequent collect/>> step.

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

    def collect(self, filter):
        """Merge all branches into this filter."""
        return Chain([self, filter])

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
