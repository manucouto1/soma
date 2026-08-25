"""Scoring a candidate without training it — and what that is **not**.

A proxy does not diagnose a network. `synflow` of one network is a number with
no meaning; it only means something next to another network's. So none of these
ever produces a `Flag`, and the tests that matter here are the ones that say so:
a proxy is a ranking, it belongs at level 3 where a study is a `for` loop, and
the one question worth asking of it is whether it beats counting parameters.

The measurement that answers that question is `health/tests/proxies.py`, which
needs an afternoon and does not belong in a suite. What is here is the contract:
that they can be taken, that they cost nothing they should not, and that taking
one leaves the candidate exactly as it was.
"""

import pytest

torch = pytest.importorskip("torch")

import soma_next.torch  # noqa: E402, F401
from soma_next import Graph, Node, Opaque  # noqa: E402
from soma_next.torch import parameters, proxies, proxy  # noqa: E402
from soma_next.torch._proxies import EVERY, FREE  # noqa: E402


class Block(Node):
    def __init__(self, width=16, out=None):
        self.net = torch.nn.Linear(width, out or width)
        self.after = torch.nn.ReLU()

    def forward(self, x, ctx):
        return Opaque(self.after(self.net(x)))

    def parameters(self):
        return list(self.net.parameters())


def chain(blocks):
    named = [block.named(f"b{i}") for i, block in enumerate(blocks)]
    wired = named[0]
    for one in named[1:]:
        wired = wired >> one
    return Graph.somatize(wired)


@pytest.fixture
def small():
    torch.manual_seed(0)
    return chain([Block() for _ in range(3)])


# ── They can be taken ──


def test_three_of_them_never_see_a_label(small):
    # The point of the family: a candidate can be scored before there is
    # anything to train it on.
    said = proxies(small, torch.randn(8, 16))

    assert set(said) == set(FREE)
    assert all(isinstance(one, float) for one in said.values())


def test_and_with_a_loss_every_one_of_them_answers(small):
    said = proxies(small, torch.randn(8, 16), target=torch.randn(8, 16),
                   objective=torch.nn.functional.mse_loss)

    assert set(said) == set(EVERY)


def test_one_that_reads_a_loss_and_was_given_none_says_which(small):
    # By name and refused, the same rule a threshold nobody has follows: a
    # measurement quietly skipped is a comparison somebody thinks they made.
    with pytest.raises(ValueError, match="snip"):
        proxy(small, torch.randn(8, 16), "snip")


def test_something_that_is_not_a_proxy_is_refused_by_name(small):
    with pytest.raises(ValueError, match="fisher"):
        proxy(small, torch.randn(8, 16), "fisher")


# ── And they leave nothing behind ──


def test_nothing_is_trained_and_no_weight_moves(small):
    before = [p.detach().clone() for p in parameters(small)]

    proxies(small, torch.randn(8, 16), target=torch.randn(8, 16),
            objective=torch.nn.functional.mse_loss)

    assert all(torch.equal(a, b) for a, b in zip(before, parameters(small)))


def test_synflow_puts_the_signs_back(small):
    # It makes every weight positive to read the topology with the values taken
    # out. A proxy that left them that way would have scored the next candidate
    # too, and nothing would have said so.
    before = [p.detach().clone() for p in parameters(small)]

    proxy(small, torch.randn(8, 16), "synflow")

    assert any((one < 0).any() for one in before), "the fixture proves nothing"
    assert all(torch.equal(a, b) for a, b in zip(before, parameters(small)))


def test_no_gradient_is_left_hanging_off_the_candidate(small):
    # Two of them run a backward. A gradient left on the parameters is a first
    # optimizer step that came from a proxy nobody asked to train with.
    proxies(small, torch.randn(8, 16), target=torch.randn(8, 16),
            objective=torch.nn.functional.mse_loss)

    assert all(p.grad is None for p in parameters(small))


# ── What they are, and what they are not ──


def test_a_proxy_is_a_ranking_and_says_nothing_about_one_network(small):
    # There is no bound in this module and there is no `Flag` coming out of it.
    # The number is only ever compared with another candidate's, which is what
    # puts the whole family at level 3.
    one = proxies(small, torch.randn(8, 16))

    assert all(isinstance(what, float) for what in one.values())
    assert not any(isinstance(what, str) for what in one.values())


def test_the_bigger_network_scores_higher_on_synflow():
    # The caveat, written as a test rather than as a footnote: `synflow`
    # correlates 0.76 with parameter count in the literature, which is close to
    # saying it measures size. Here it is, measuring size.
    torch.manual_seed(0)
    narrow = chain([Block(width=16) for _ in range(3)])
    torch.manual_seed(0)
    wide = chain([Block(width=16, out=64), Block(width=64), Block(width=64, out=16)])
    x = torch.randn(8, 16)

    assert proxy(wide, x, "synflow") > proxy(narrow, x, "synflow")


def test_a_batch_of_one_has_nothing_to_tell_apart(small):
    # `naswot` asks how differently two inputs switch the units. One input is
    # not two, and `nan` says so rather than a number that reads like an answer.
    import math

    assert math.isnan(proxy(small, torch.randn(1, 16), "naswot"))
