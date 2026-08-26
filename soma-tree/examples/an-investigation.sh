#!/usr/bin/env bash
# A small investigation, from nothing to a judged line of exploration.
#
# Four commits, each a real kind of edit, and one of them wrong on purpose.
# Run it and read what `somatize-tree` says at each step; it takes about a minute
# and it is the fastest way to see what the tool is for.
#
#     examples/an-investigation.sh
#
# With `--only-build <dir>` it lays down the repository and stops, which is
# what the end-to-end test uses: the example and the test share one fixture, so
# the example cannot rot without the test going red.
set -euo pipefail

ONLY_BUILD=""
if [ "${1:-}" = "--only-build" ]; then ONLY_BUILD="$2"; fi
WHERE="${ONLY_BUILD:-$(mktemp -d)}"
PYTHON="${SOMA_TREE_PYTHON:-$(cd "$(dirname "$0")/../../soma" 2>/dev/null && pwd)/.venv/bin/python}"

mkdir -p "$WHERE/experiments"
cd "$WHERE"
git init -q . && git config user.email you@example.com && git config user.name "You"

cat > soma-tree.toml <<TOML
build = "experiments.encoder:build"
python = "$PYTHON"
tree = "an-investigation"
TOML

# The graph. Four nodes, and every knob through a constructor.
cat > experiments/encoder.py <<'PY'
from somatize import Graph, Node


class Tokenize(Node):
    def forward(self, x, ctx):
        return x.split()


class Embed(Node):
    def __init__(self, scale=0.5):
        self.scale = scale

    def forward(self, x, ctx):
        return [len(t) * self.scale for t in x]


class Classify(Node):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def forward(self, x, ctx):
        return sum(1 for v in x if v > self.threshold)


class Vote(Node):
    def forward(self, x, ctx):
        return max(x.values()) if isinstance(x, dict) else x


def build():
    g = Graph.somatize(
        Tokenize().named("tokenize").frozen()
        >> Embed().named("embed").frozen().cached()
        >> (Classify(1.0).named("strict") | Classify(0.2).named("loose"))
        >> Vote().named("vote")
    )
    g.freeze("embed", "weights-v1")
    return g
PY
git add -A && git commit -qm "base: tokenize, embed, two classifiers, a vote"

# 1. A constructor argument. In the key, so the cache misses.
sed -i 's/Classify(1.0).named("strict")/Classify(2.0).named("strict")/' experiments/encoder.py
git commit -qam "the strict threshold goes up to 2.0"

# 2. The body of a forward. NOT in the key: the cache will hit.
sed -i 's/return \[len(t) \* self.scale for t in x\]/return [(len(t) ** 2) * self.scale for t in x]/' experiments/encoder.py
git commit -qam "the embedding becomes quadratic"

# 3. Only the weights. Another trial, not another variant.
sed -i 's/g.freeze("embed", "weights-v1")/g.freeze("embed", "weights-v2")/' experiments/encoder.py
git commit -qam "encoder retrained, same code"

if [ -n "$ONLY_BUILD" ]; then echo "$WHERE"; exit 0; fi

SOMA_TREE="${SOMA_TREE_BIN:-somatize-tree}"
say() { printf '\n\033[1m── %s\033[0m\n\n' "$1"; }

say "Step 1: a constructor argument"
"$SOMA_TREE" diff HEAD~3 HEAD~2 || true
echo "   The key moves, so the cache misses and recomputes. Correct."

say "Step 2: the body of a forward"
"$SOMA_TREE" diff HEAD~2 HEAD~1 || true
echo "   STALE. The key does NOT move, so the cache hits and hands back what"
echo "   the old code produced. Nothing else says this."

say "Step 3: only the weights"
"$SOMA_TREE" diff HEAD~1 HEAD || true
echo "   No edit. Retraining is another trial of the same variant."

say "The whole line"
"$SOMA_TREE" log HEAD~3..HEAD || true

say "And now it is judged"
"$SOMA_TREE" verdict invalid HEAD~1 -m "The dataloader duplicated the last batch: nothing measured below here holds."
"$SOMA_TREE" note HEAD~2 -m "Recall 0.61 at threshold 2.0. Check whether it is the split."
"$SOMA_TREE" log HEAD~3..HEAD || true
echo "   Nobody judged the commit above: it inherits the doubt through git, and"
echo "   nobody had to write that down anywhere."

say "Everything said about one commit"
"$SOMA_TREE" show HEAD~2

printf '\nThe repository is at %s\n' "$WHERE"
