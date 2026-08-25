"""Whether the model is learning what you think it is learning.

The question that is not about the network at all, and the one a real project
spent months not asking: symptom channels for a mental-health condition, where
interpretability and performance came one at a time. The signal was in the
self-disclosure. Nothing in the architecture was ever going to say so.
"""

import pytest

torch = pytest.importorskip("torch")

import somatize.torch  # noqa: E402, F401
from somatize import Graph, Node, Opaque  # noqa: E402
from somatize.data import contribution, leaning, shares, shuffled  # noqa: E402
from somatize.torch import Trainer, parameters  # noqa: E402

MSE = torch.nn.functional.mse_loss


class Reads(Node):
    """One branch, reading its own key out of the input."""

    def __init__(self, key, width=4, out=8):
        self.key = key
        self.net = torch.nn.Linear(width, out)

    def forward(self, said, ctx):
        return Opaque(self.net(said[self.key]))

    def parameters(self):
        return list(self.net.parameters())


class Fuse(Node):
    def __init__(self, width=16):
        self.out = torch.nn.Linear(width, 1)

    def forward(self, said, ctx):
        return Opaque(self.out(torch.cat(list(said.values()), dim=1)))

    def parameters(self):
        return list(self.out.parameters())


def two_branches():
    return Graph.somatize(
        (Reads("symptoms").named("symptoms") | Reads("disclosure").named("disclosure"))
        >> Fuse().named("fuse")
    )


def batches(how_many=4, rows=128, answer="disclosure", seed=0):
    """Data where the answer is in **one** of the two channels and nowhere else."""
    torch.manual_seed(seed)
    weights = torch.tensor([[1.0], [-1.0], [0.5], [0.0]])
    made = []
    for _ in range(how_many):
        said = {one: torch.randn(rows, 4) for one in ("symptoms", "disclosure")}
        target = said[answer] @ weights + 0.05 * torch.randn(rows, 1)
        made.append(({one: Opaque(what) for one, what in said.items()}, target))
    return made


def trained(g, data, steps=300):
    t = Trainer(g, objective=MSE, optimizer=torch.optim.Adam(parameters(g), lr=0.02))
    for which in range(steps):
        t.step(data[which % len(data)])
    return g


# ── The case this exists for ──


def test_an_input_the_model_is_not_using_is_found_in_one_afternoon():
    g = two_branches()
    data = batches()
    trained(g, data)

    said = contribution(g, data, objective=MSE, over=("symptoms", "disclosure"))

    assert leaning(said) == {
        "symptoms": ["IGNORED_INPUT(symptoms)"],
        "disclosure": ["SOLE_RELIANCE(disclosure)"],
    }


def test_and_the_shares_say_how_lopsided_it_is():
    g = two_branches()
    data = batches()
    trained(g, data)

    said = shares(contribution(g, data, objective=MSE))

    assert said["disclosure"] > 0.95
    assert said["symptoms"] < 0.05


def test_two_channels_that_both_carry_it_say_nothing():
    # The detector may not cry wolf: a model using both of its inputs is the
    # ordinary case and has to come back quiet.
    torch.manual_seed(0)
    g = two_branches()
    data = []
    for _ in range(4):
        said = {one: torch.randn(128, 4) for one in ("symptoms", "disclosure")}
        target = (said["symptoms"].sum(1) + said["disclosure"].sum(1)).unsqueeze(1)
        data.append(({one: Opaque(what) for one, what in said.items()}, target))
    trained(g, data)

    assert leaning(contribution(g, data, objective=MSE)) == {}


# ── What it measures, and how ──


def test_nothing_is_trained_and_nothing_is_changed():
    g = two_branches()
    data = batches()
    trained(g, data, steps=50)
    before = torch.cat([p.detach().clone().reshape(-1) for p in parameters(g)])

    contribution(g, data, objective=MSE)

    assert torch.equal(before, torch.cat([p.detach().reshape(-1) for p in parameters(g)]))


def test_shuffling_keeps_the_channel_and_breaks_only_what_it_lines_up_with():
    # A zero is a value, and often an unusually informative one. What is being
    # asked about is the **correspondence** with the answer, and shuffling is
    # what destroys exactly that and nothing else.
    what = torch.arange(12.0).reshape(4, 3)

    moved = shuffled(what, [3, 2, 1, 0])

    assert torch.equal(moved.sort(dim=0).values, what.sort(dim=0).values)
    assert not torch.equal(moved, what)


def test_an_opaque_is_unwrapped_and_wrapped_again():
    # Shuffling the wrapper rather than the tensor is a very quiet way of
    # measuring nothing.
    what = Opaque(torch.arange(8.0).reshape(4, 2))

    moved = shuffled(what, [1, 0, 3, 2])

    assert isinstance(moved, Opaque)
    assert not torch.equal(moved.value, what.value)


def test_something_that_is_not_a_batch_is_left_alone():
    assert shuffled("a string", [1, 0]) == "a string"
    assert shuffled(7, [1, 0]) == 7


def test_only_the_inputs_asked_for_are_tried():
    g = two_branches()
    data = batches()
    trained(g, data, steps=50)

    assert list(contribution(g, data, objective=MSE, over=("symptoms",))) == ["symptoms"]


def test_no_data_is_nothing_and_not_a_failure():
    assert contribution(two_branches(), [], objective=MSE) == {}


def test_one_input_alone_says_nothing_because_there_is_nothing_to_compare():
    assert leaning({"only": 5.0}) == {}
