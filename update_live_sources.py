import argparse
import json
from datetime import datetime

from live_source_refresh import (
    LIVE_SOURCE_REFRESH_STATUS_PATH,
    LIVE_SOURCES_DIR,
    read_live_source_refresh_status,
    refresh_live_sources_if_needed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refresh local nflverse/nflreadpy live source CSVs. By default this respects "
            "the daily 11:59 PM ET refresh gate so it is safe to call from GitHub Actions, "
            "cron, Task Scheduler, or another external scheduler."
        )
    )
    parser.add_argument(
        "season",
        nargs="?",
        type=int,
        default=datetime.now().year,
        help="Dashboard season to refresh around. Future-only sources fall back per source when unavailable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the daily 11:59 PM ET freshness gate and refresh immediately.",
    )
    parser.add_argument(
        "--skip-if-fresh",
        action="store_true",
        help="Backward-compatible no-op; scheduled freshness gating is now the default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ok, message = refresh_live_sources_if_needed(args.season, force=args.force)
    status = read_live_source_refresh_status()

    print("=" * 70)
    print("Live source refresh")
    print(f"Output folder: {LIVE_SOURCES_DIR}")
    print(f"Status file: {LIVE_SOURCE_REFRESH_STATUS_PATH}")
    print(f"Result: {'updated' if ok else 'skipped or failed'}")
    print(message)

    if status.get("sources"):
        print("-" * 70)
        print(json.dumps(status["sources"], indent=2))

    print("=" * 70)


if __name__ == "__main__":
    main()
