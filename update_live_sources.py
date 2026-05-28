from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
LIVE_DIR = DATA_DIR / "live_sources"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# Current-moment setup:
# The leaderboard can show 2026 roster context, but 2026 player production does not exist yet.
# Use 2025 player production until 2026 regular-season data is available.
SEASONS = [2025]


def to_pandas_df(obj):
    """Convert nflreadpy return objects to pandas safely."""
    if obj is None:
        return pd.DataFrame()

    if isinstance(obj, pd.DataFrame):
        return obj

    # Polars DataFrame
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas()
        except Exception:
            pass

    # Polars LazyFrame
    if hasattr(obj, "collect"):
        try:
            collected = obj.collect()
            if hasattr(collected, "to_pandas"):
                return collected.to_pandas()
        except Exception:
            pass

    return pd.DataFrame()


def save_csv(obj, filename):
    df = to_pandas_df(obj)
    path = LIVE_DIR / filename

    if df.empty:
        print(f"SKIPPED {filename}: no rows returned")
        return False

    df.to_csv(path, index=False)
    print(f"SAVED {filename}: {len(df):,} rows | {path}")
    return True


def try_loader(nfl, function_name, output_filename, seasons=None, **kwargs):
    func = getattr(nfl, function_name, None)

    if func is None:
        print(f"SKIPPED {output_filename}: nflreadpy has no function named {function_name}")
        return False

    print(f"Loading {function_name}...")

    try:
        if seasons is None:
            data = func(**kwargs)
        else:
            data = func(seasons, **kwargs)

        return save_csv(data, output_filename)

    except TypeError:
        try:
            if seasons is None:
                data = func(**kwargs)
            else:
                data = func(seasons=seasons, **kwargs)

            return save_csv(data, output_filename)

        except Exception as exc:
            print(f"FAILED {function_name}: {exc}")
            return False

    except Exception as exc:
        print(f"FAILED {function_name}: {exc}")
        return False


def main():
    print("=" * 70)
    print("Updating local nflverse/nflreadpy cache")
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Output folder: {LIVE_DIR}")
    print(f"Seasons requested: {SEASONS}")
    print("=" * 70)

    try:
        import nflreadpy as nfl
    except Exception as exc:
        print("nflreadpy could not be imported.")
        print("Run this first: pip install nflreadpy pyarrow")
        print(exc)
        return

    saved_any = False

    # These names match what your Live_Leaderboard.py already checks in data/live_sources.
    saved_any |= try_loader(nfl, "load_player_stats", "player_season_stats.csv", SEASONS)
    saved_any |= try_loader(nfl, "load_rosters", "weekly_rosters.csv", SEASONS)
    saved_any |= try_loader(nfl, "load_rosters_weekly", "weekly_rosters.csv", SEASONS)
    saved_any |= try_loader(nfl, "load_snap_counts", "snap_counts.csv", SEASONS)
    saved_any |= try_loader(nfl, "load_injuries", "injuries.csv", SEASONS)
    saved_any |= try_loader(nfl, "load_schedules", "schedules.csv", SEASONS)
    saved_any |= try_loader(nfl, "load_teams", "teams.csv", None)
    saved_any |= try_loader(nfl, "load_players", "players.csv", None)

    print("=" * 70)
    if saved_any:
        print("Done. At least one live_sources CSV was updated.")
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    else:
        print("No files were saved. The Live Leaderboard will keep using roster proxy scoring.")
    print("=" * 70)


if __name__ == "__main__":
    main()