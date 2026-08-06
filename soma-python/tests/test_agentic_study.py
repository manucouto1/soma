"""Tuning an agentic graph with a Study.

This is the point of the whole exercise: an agentic flow is a Soma graph, so
the sampler, the pruning and the experiment pool that already exist apply to
it unchanged. Nothing here is agentic-specific machinery — it is the same
``Study`` a computational pipeline uses, over a space that happens to contain
a prompt.

The model is a mock that grades by content, so the search has real signal
and no network.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import soma


class _ByContent(BaseHTTPRequestHandler):
    """Answers as a function of the request, so trials actually differ."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.received.append(body)

        text = self.server.reply(body)
        payload = json.dumps(
            {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


class ContentProvider:
    def __init__(self, reply):
        self.server = HTTPServer(("127.0.0.1", 0), _ByContent)
        self.server.reply = reply
        self.server.received = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"


# The writer parrots its system prompt; the judge scores that text. A
# "detailed" writer scores well, a "terse" one badly — a gradient for the
# sampler to find.
_QUALITY = {"be detailed": 0.9, "be helpful": 0.6, "be terse": 0.2}


def _reply(body):
    messages = body["messages"]
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    if system.startswith("You grade"):
        graded = messages[-1]["content"]
        score = next((v for k, v in _QUALITY.items() if k in graded), 0.0)
        return json.dumps({"score": score, "reason": "graded by content"})
    return system


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A throwaway project dir with a provider catalog pointed at a mock."""

    def _setup(base_url):
        path = tmp_path / "providers.toml"
        path.write_text(
            f'[providers.mock]\nbase_url = "{base_url}"\nauth = {{ type = "none" }}\n'
        )
        monkeypatch.setenv("SOMA_PROVIDERS", str(path))
        monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.chdir(tmp_path)
        return tmp_path

    return _setup


def _refine_graph():
    g = soma.Graph(cache="memory")
    g.node(
        "writer",
        soma.Agent(
            model="mock/any",
            system=soma.search(choices=["be terse", "be helpful", "be detailed"]),
        ),
    )
    g.node("critic", soma.Judge(model="mock/any", rubric="Is it useful?"))
    g.edge("writer", "critic")
    return g


def test_a_study_tunes_an_agents_prompt(lab):
    with ContentProvider(_reply) as provider:
        lab(provider.base_url)

        g = _refine_graph()
        assert [d["name"] for d in g.search_space()] == ["writer.system"]

        def run_trial(trial):
            g.apply_params(trial.params)
            return {"score": g.forward("explain compilers")["score"]}

        study = g.study(
            "prompt-search",
            strategy="grid",
            n_trials=3,
            objectives=[("score", "maximize")],
            tracking=False,
        )
        study.run(run_trial)

        best = study.best_trial
        assert best is not None
        # The search found the prompt that grades best — over the same
        # sampler a computational pipeline uses.
        assert best["params"]["writer.system"] == "be detailed"
        assert best["metrics"]["score"] == pytest.approx(0.9)


def test_a_study_searches_prompt_and_topology_together(lab):
    with ContentProvider(_reply) as provider:
        lab(provider.base_url)

        g = _refine_graph()
        g.optional("writer", "critic")

        names = sorted(d["name"] for d in g.search_space())
        assert names == ["edge:writer->critic", "writer.system"]

        seen = []

        def run_trial(trial):
            g.apply_params(trial.params)
            seen.append(trial.params)
            # With the edge cut the critic grades the raw task instead of the
            # draft, which is exactly the difference the search is measuring.
            return {"score": g.forward("explain compilers")["score"]}

        study = g.study(
            "shape-and-prompt",
            strategy="random",
            n_trials=6,
            objectives=[("score", "maximize")],
            seed=7,
            tracking=False,
        )
        study.run(run_trial)

        assert len(seen) == 6
        assert any(p["edge:writer->critic"] for p in seen)
        assert study.best_trial["metrics"]["score"] > 0.0


def test_every_trial_lands_in_the_experiment_pool(lab):
    with ContentProvider(_reply) as provider:
        root = lab(provider.base_url)

        g = _refine_graph()

        def run_trial(trial):
            g.apply_params(trial.params)
            with g.track_run("pooled", tags=["agentic-study"]):
                score = g.forward("explain compilers")["score"]
            return {"score": score}

        study = g.study(
            "pooled",
            strategy="grid",
            n_trials=3,
            objectives=[("score", "maximize")],
            tracking=False,
        )
        study.run(run_trial)

        pool = root / ".soma" / "experiments.jsonl"
        assert pool.exists(), "a tracked agentic run belongs in the pool like any other"
        records = [json.loads(line) for line in pool.read_text().splitlines() if line]
        assert len(records) == 3
        # Lineage is what makes two topologies comparable after the fact.
        assert all(r.get("architecture") for r in records), sorted(records[0])
        assert any(r.get("parent") for r in records[1:]), "runs chain off HEAD"


def test_the_run_dir_holds_the_agentic_topology(lab):
    with ContentProvider(_reply) as provider:
        root = lab(provider.base_url)

        g = _refine_graph()
        with g.track_run("once"):
            g.forward("explain compilers")

        runs = sorted((root / ".soma" / "runs").glob("*"))
        assert runs, "a tracked run writes a run dir"
        graph_json = json.loads((runs[-1] / "graph.json").read_text())
        kinds = {n["id"]: n["kind"] for n in graph_json["nodes"]}
        # The snapshot has to say these are steps, not filters, or the pool
        # cannot tell an agentic graph from a computational one.
        assert "Step" in json.dumps(kinds["writer"])
        assert os.path.exists(runs[-1] / "graph.mmd")
