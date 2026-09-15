import argparse
from live_source_refresh import LIVE_SOURCE_REFRESH_STATUS_PATH, LIVE_SOURCES_DIR
from refresh_current_season import refresh_current_season


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refresh current-season PBP, schedule/results, Gini summaries, and roster/context "
            "sources through the authoritative daily pipeline."
        )
    )
    parser.add_argument(
        "season",
        nargs="?",
        type=int,
        default=None,
        help="NFL season to refresh. Defaults to the dynamically detected active season.",
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
    ok, message, status = refresh_current_season(args.season, force=args.force)

    print("=" * 70)
    print("Authoritative current-season refresh")
    print(f"Output folder: {LIVE_SOURCES_DIR}")
    print(f"Status file: {LIVE_SOURCE_REFRESH_STATUS_PATH}")
    result = "updated" if ok else "skipped" if status.get("skipped") else "failed"
    print(f"Result: {result}")
    print(message)

    print("=" * 70)

    if not ok and not status.get("skipped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
