"""CLI entry point for soma-worker.

Installed as 'somatize-worker' via pip install somatize.

Usage:
    somatize-worker --port 8080 --tags gpu,training --token sk-xxx
    somatize-worker --coordinator http://coord:9090 --token sk-xxx
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="somatize-worker",
        description="Soma distributed execution worker",
    )
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--tags", type=str, default="", help="Comma-separated tags (e.g. gpu,training)")
    parser.add_argument("--token", type=str, default=None, help="Bearer token for authentication")
    parser.add_argument("--cpus", type=int, default=None, help="Max CPU cores to expose")
    parser.add_argument("--memory", type=str, default=None, help="Max memory (e.g. 8G, 512M)")
    parser.add_argument("--gpus", type=int, default=None, help="Max GPUs to expose")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent plans")
    parser.add_argument("--id", type=str, default=None, help="Worker ID")
    parser.add_argument("--coordinator", type=str, default=None, help="Coordinator URL")

    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    memory_bytes = None
    if args.memory:
        m = args.memory.strip().upper()
        if m.endswith("G"):
            memory_bytes = int(m[:-1]) * 1024 * 1024 * 1024
        elif m.endswith("M"):
            memory_bytes = int(m[:-1]) * 1024 * 1024
        elif m.endswith("T"):
            memory_bytes = int(m[:-1]) * 1024 * 1024 * 1024 * 1024
        else:
            memory_bytes = int(m)

    from soma._soma import Worker

    worker = Worker(
        port=args.port,
        tags=tags,
        token=args.token,
        cpus=args.cpus,
        memory=memory_bytes,
        gpus=args.gpus,
        max_concurrent=args.max_concurrent,
        worker_id=args.id,
        coordinator=args.coordinator,
    )

    print(f"Starting Soma worker on port {args.port}...")
    print(f"Tags: {tags}")
    if args.token:
        print("Authentication: enabled")
    if args.coordinator:
        print(f"Coordinator: {args.coordinator}")

    try:
        worker.serve()
    except KeyboardInterrupt:
        print("\nWorker stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
