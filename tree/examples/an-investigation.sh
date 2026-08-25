#!/usr/bin/env bash
# A small investigation, from nothing to a judged line of exploration.
#
# Four commits, each a real kind of edit, and one of them wrong on purpose.
# Run it and read what `soma-tree` says at each step; it takes about a minute
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
PYTHON="${SOMA_TREE_PYTHON:-$(cd "$(dirname "$0")/../../soma-next" 2>/dev/null && pwd)/.venv/bin/python}"

mkdir -p "$WHERE/experiments"
cd "$WHERE"
git init -q . && git config user.email you@example.com && git config user.name "You"

cat > soma-tree.toml <<TOML
build = "experiments.encoder:build"
python = "$PYTHON"
tree = "an-investigation"
TOML

# ── The graph. Four nodes, and every knob through a constructor. ──
cat > experiments/encoder.py <<'PY'
from soma_next import Graph, Node


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
    g.freeze("embed", "pesos-v1")
    return g
PY
git add -A && git commit -qm "base: tokenize, embed, dos clasificadores, voto"

# ── 1. A constructor argument. In the key, so the cache misses. ──
sed -i 's/Classify(1.0).named("strict")/Classify(2.0).named("strict")/' experiments/encoder.py
git commit -qam "el umbral estricto sube a 2.0"

# ── 2. The body of a forward. NOT in the key: the cache will hit. ──
sed -i 's/return \[len(t) \* self.scale for t in x\]/return [(len(t) ** 2) * self.scale for t in x]/' experiments/encoder.py
git commit -qam "el embedding pasa a ser cuadratico"

# ── 3. Only the weights. Another trial, not another variant. ──
sed -i 's/g.freeze("embed", "pesos-v1")/g.freeze("embed", "pesos-v2")/' experiments/encoder.py
git commit -qam "reentrenado el encoder, mismo codigo"

if [ -n "$ONLY_BUILD" ]; then echo "$WHERE"; exit 0; fi

SOMA_TREE="${SOMA_TREE_BIN:-soma-tree}"
say() { printf '\n\033[1m── %s\033[0m\n\n' "$1"; }

say "El paso 1: un argumento del constructor"
"$SOMA_TREE" diff HEAD~3 HEAD~2 || true
echo "   La clave se mueve, así que la caché falla y recalcula. Correcto."

say "El paso 2: el cuerpo de un forward"
"$SOMA_TREE" diff HEAD~2 HEAD~1 || true
echo "   RANCIO. La clave NO se mueve, así que la caché acierta y te devuelve"
echo "   lo que produjo el código viejo. Esto no lo dice nada más."

say "El paso 3: sólo los pesos"
"$SOMA_TREE" diff HEAD~1 HEAD || true
echo "   Ninguna edición. Reentrenar es otro trial de la misma variante."

say "La línea entera"
"$SOMA_TREE" log HEAD~3..HEAD || true

say "Y ahora se juzga"
"$SOMA_TREE" verdict invalid HEAD~1 -m "El dataloader duplicaba el último batch: nada medido aquí abajo vale."
"$SOMA_TREE" note HEAD~2 -m "Recall 0.61 con umbral 2.0. Mirar si es el split."
"$SOMA_TREE" log HEAD~3..HEAD || true
echo "   El commit de arriba nadie lo ha juzgado: hereda la duda de git, y no"
echo "   hubo que escribirla en ningún sitio."

say "Todo lo dicho sobre un commit"
"$SOMA_TREE" show HEAD~2

printf '\nEl repositorio está en %s\n' "$WHERE"
