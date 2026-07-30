"""`soma` command-line entry point (cache management + run inspection).

Usage::

    soma cache stats [--dir PATH]
    soma cache gc --max-size 20G [--min-age 3600] [--dir PATH]
    soma cache pin NAME ACTION_KEY_HEX [--dir PATH]
    soma cache verify [--dir PATH]
    soma cache purge-v1 [--dir PATH]
    soma runs [--root .soma] [--json]

The cache lives at ``$SOMA_CACHE_DIR`` (default ``~/.soma/cache``).
GC evicts blob payloads only — action records are retained, so evicted
entries are recomputed and re-fill the same content address on the next
run: eviction can never lose correctness, only warm-ness.

``soma runs`` lists the tracked run directories under ``<root>/runs/``
(the ones ``graph.track_run(...)`` and ``Study.run(...)`` write).
"""

from __future__ import annotations

import argparse
import sys


def _parse_size(text: str) -> int:
    t = text.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if t and t[-1] in multipliers:
        return int(float(t[:-1]) * multipliers[t[-1]])
    return int(t)


def _fmt_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def _fmt_duration(ms: int) -> str:
    seconds = ms / 1000
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


_STATE_STYLES = {  # same hues as the HTML report / notebook chips
    "completed": "green",
    "failed": "red",
    "crashed": "red",
    "running": "yellow",
}


def _print_runs(infos: list[dict], plain: bool = False) -> None:
    """Runs table: rich (colored states) when available, plain text
    otherwise or when forced (pipes)."""
    if not plain:
        try:
            from rich.console import Console
            from rich.table import Table

            # Non-tty (pipes, captured output): fixed generous width so
            # nothing truncates; rich already strips colors there.
            width = None if sys.stdout.isatty() else 160
            table = Table(box=None, pad_edge=False, header_style="dim")
            table.add_column("RUN ID", style="cyan", no_wrap=True)
            table.add_column("KIND")
            table.add_column("STATE")
            table.add_column("CREATED", no_wrap=True)
            table.add_column("DURATION", justify="right")
            table.add_column("NAME")
            for info in infos:
                state = info["state"]
                style = _STATE_STYLES.get(state, "dim")
                table.add_row(
                    info["run_id"],
                    info["kind"],
                    f"[{style}]{state}[/{style}]",
                    (info.get("created_at") or "")[:19].replace("T", " "),
                    _fmt_duration(info["duration_ms"]) if info.get("duration_ms") else "-",
                    info["name"],
                )
            Console(width=width).print(table)
            return
        except ImportError:
            pass
    header = f"{'RUN ID':<32} {'KIND':<6} {'STATE':<10} {'CREATED':<20} {'DURATION':>8}  NAME"
    print(header)
    for info in infos:
        duration = _fmt_duration(info["duration_ms"]) if info.get("duration_ms") else "-"
        created = (info.get("created_at") or "")[:19]
        print(
            f"{info['run_id']:<32} {info['kind']:<6} {info['state']:<10} "
            f"{created:<20} {duration:>8}  {info['name']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soma", description="Soma toolbox")
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache", help="manage the persistent computation cache")
    cache_sub = cache.add_subparsers(dest="action", required=True)

    p_stats = cache_sub.add_parser("stats", help="show cache statistics")
    p_gc = cache_sub.add_parser("gc", help="evict low-value blobs down to a size budget")
    p_gc.add_argument("--max-size", required=True, help="target CAS size (e.g. 20G, 500M)")
    p_gc.add_argument(
        "--min-age",
        type=int,
        default=3600,
        help="never evict blobs younger than this many seconds (default 3600)",
    )
    p_pin = cache_sub.add_parser("pin", help="protect an action's outputs from GC")
    p_pin.add_argument("name", help="human-readable pin name")
    p_pin.add_argument("key", help="64-char hex action key")
    p_verify = cache_sub.add_parser("verify", help="check blob integrity against content hashes")
    p_purge = cache_sub.add_parser("purge-v1", help="delete unreachable Phase-1 cache entries")
    for p in (p_stats, p_gc, p_pin, p_verify, p_purge):
        p.add_argument("--dir", default=None, help="cache directory (default $SOMA_CACHE_DIR or ~/.soma/cache)")

    p_runs = sub.add_parser("runs", help="list tracked runs under <root>/runs/")
    p_runs.add_argument("--root", default=".soma", help="tracking root (default .soma)")
    p_runs.add_argument("--json", action="store_true", help="print raw JSON instead of a table")
    p_runs.add_argument(
        "--plain",
        action="store_true",
        help="force the plain-text table (no colors; good for pipes)",
    )

    p_graph = sub.add_parser(
        "graph", help="render a run's graph annotated with timing/cache/health"
    )
    p_graph.add_argument("run", help="run id (resolved under --root) or run directory path")
    p_graph.add_argument(
        "--format", choices=("mermaid", "dot"), default="mermaid", help="output format"
    )
    p_graph.add_argument("--no-overlay", action="store_true", help="plain topology, no annotations")
    p_graph.add_argument("--root", default=".soma", help="tracking root (default .soma)")

    p_kb = sub.add_parser("kb", help="inspect the experiment pool (experiments.jsonl)")
    kb_sub = p_kb.add_subparsers(dest="action", required=True)
    p_reindex = kb_sub.add_parser(
        "reindex",
        help="rebuild experiments.jsonl from <root>/runs/ (migration, backfill, recovery)",
    )
    p_head = kb_sub.add_parser("head", help="show which run the next one will descend from")
    p_checkout = kb_sub.add_parser(
        "checkout", help="point HEAD at an existing run so the next run branches from it"
    )
    p_checkout.add_argument("run", help="run id to branch from")
    p_detach = kb_sub.add_parser("detach", help="clear HEAD; the next run starts a new line")
    for p in (p_reindex, p_head, p_checkout, p_detach):
        p.add_argument("--root", default=".soma", help="tracking root (default .soma)")

    p_report = sub.add_parser(
        "report", help="generate a self-contained HTML report for a run or study"
    )
    p_report.add_argument("run", help="run id (resolved under --root) or run directory path")
    p_report.add_argument("-o", "--output", default=None, help="output file (default <run_id>.html)")
    p_report.add_argument(
        "--inline",
        action="store_true",
        help="fully self-contained: embed plotly.js and render diagrams as SVG (bigger file, works offline)",
    )
    p_report.add_argument("--open", action="store_true", help="open the report in a browser")
    p_report.add_argument("--root", default=".soma", help="tracking root (default .soma)")

    args = parser.parse_args(argv)

    from soma import _soma

    def _resolve_run_dir(run, root):
        import json as _json
        import pathlib

        if pathlib.Path(run).is_dir():
            return str(run)
        infos = _json.loads(_soma.list_runs_json(root=root))
        match = next((i for i in infos if i["run_id"] == run), None)
        if match is None:
            print(f"no run '{run}' under {root}/runs/", file=sys.stderr)
            return None
        return match["dir"]

    if args.command == "kb":
        if args.action == "reindex":
            n = _soma.kb_reindex(root=args.root)
            print(f"reindexed {n} records into {args.root}/experiments.jsonl")
            return 0
        if args.action == "head":
            head = _soma.read_head_run(root=args.root)
            print(head if head else "HEAD is detached; the next run starts a new line")
            return 0
        if args.action == "checkout":
            try:
                _soma.checkout_run(args.run, root=args.root)
            except Exception as exc:  # noqa: BLE001 — surface soma's own message
                print(str(exc), file=sys.stderr)
                return 1
            print(f"HEAD -> {args.run}; the next run will branch from it")
            return 0
        if args.action == "detach":
            _soma.clear_head_run(root=args.root)
            print("HEAD detached; the next run starts a new line")
            return 0

    if args.command == "report":
        from soma._runs import RunView
        from soma.viz import to_html

        run_dir = _resolve_run_dir(args.run, args.root)
        if run_dir is None:
            return 1
        view = RunView(run_dir)
        out = args.output or f"{view.id}.html"
        to_html(view, path=out, inline=args.inline)
        print(f"report written to {out}")
        if args.open:
            import webbrowser

            webbrowser.open(f"file://{__import__('pathlib').Path(out).resolve()}")
        return 0

    if args.command == "graph":
        run_dir = _resolve_run_dir(args.run, args.root)
        if run_dir is None:
            return 1
        render = _soma.run_to_mermaid if args.format == "mermaid" else _soma.run_to_graphviz
        print(render(run_dir, overlay=not args.no_overlay), end="")
        return 0

    if args.command == "runs":
        import json as _json

        infos = _json.loads(_soma.list_runs_json(root=args.root))
        if args.json:
            print(_json.dumps(infos, indent=2))
            return 0
        if not infos:
            print(f"no runs under {args.root}/runs/")
            return 0
        _print_runs(infos, plain=args.plain)
        return 0

    if args.action == "stats":
        s = _soma.cache_stats(path=args.dir)
        print(f"cache root      {s['root']}")
        print(f"action records  {s['actions']}")
        print(
            f"unique outputs  {s['unique_outputs']} "
            f"({s['blobs_available']} on disk, {s['blobs_evicted']} evicted)"
        )
        print(f"CAS size        {_fmt_size(s['cas_bytes'])}")
        print(f"pinned          {s['pinned']}")
        print(f"compute banked  {_fmt_duration(s['saved_compute_ms'])}")
    elif args.action == "gc":
        r = _soma.cache_gc(
            max_bytes=_parse_size(args.max_size),
            min_age_secs=args.min_age,
            path=args.dir,
        )
        print(
            f"evicted {r['blobs_evicted']} blobs "
            f"({_fmt_size(r['bytes_before'])} -> {_fmt_size(r['bytes_after'])}); "
            f"kept {r['blobs_kept']}, pinned {r['pinned_blobs']}"
        )
    elif args.action == "pin":
        _soma.cache_pin(args.name, args.key, path=args.dir)
        print(f"pinned {args.name} -> {args.key[:16]}...")
    elif args.action == "verify":
        r = _soma.cache_verify(path=args.dir)
        print(f"ok {r['ok']}, missing (evicted) {r['missing']}, corrupt {r['corrupt']}")
        if r["corrupt"]:
            return 1
    elif args.action == "purge-v1":
        r = _soma.cache_purge_v1(path=args.dir)
        print(f"removed {r['removed_files']} v1 files ({_fmt_size(r['removed_bytes'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
