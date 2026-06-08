from pathlib import Path
import re

import numpy as np
import pandas as pd


TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "JAC": "JAX",
    "LAR": "LA",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    "WSH": "WAS",
}

ROSTER_SCORE_COLUMNS = [
    "team",
    "roster_score",
    "qb_score",
    "offense_skill_score",
    "offensive_line_score",
    "defensive_front_score",
    "secondary_score",
    "special_teams_score",
    "premium_position_score",
    "availability_score",
    "rookie_projection_score",
    "transaction_impact_score",
    "experience_score",
    "draft_capital_score",
    "continuity_score",
]

LIVE_ROSTER_SOURCE_KEYS = [
    "weekly_rosters",
    "player_weekly_stats",
    "player_season_stats",
    "player_snap_counts",
    "injuries",
    "transactions",
    "depth_charts",
    "draft_picks",
    "contracts",
    "players",
]


def normalize_team(team):
    value = "" if pd.isna(team) else str(team).strip().upper()
    return TEAM_ALIASES.get(value, value)


def normalize_score(series, center=85, spread=10, lower=60, upper=120):
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series(center, index=numeric.index)
    return (center + spread * ((numeric - numeric.mean()) / std)).clip(lower, upper).fillna(center)


def first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def infer_roster_year_from_path(path):
    match = re.search(r"(?:^|[_-])rosters?[_-](\d{4})", Path(path).stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def season_roster_source_paths(data_dir):
    root = Path(data_dir)
    if not root.exists():
        return []

    paths = []
    for pattern in ("roster_*.csv", "rosters_*.csv"):
        paths.extend(root.glob(pattern))

    unique = []
    seen = set()
    for path in paths:
        year = infer_roster_year_from_path(path)
        if year is None:
            continue
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return sorted(unique, key=lambda path: (infer_roster_year_from_path(path) or 0, path.name.lower()))


def load_season_roster_sources(data_dir):
    sources = {}
    messages = []
    for path in season_roster_source_paths(data_dir):
        year = infer_roster_year_from_path(path)
        if year is None:
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            messages.append(f"Could not read season roster source {path.name}: {exc}")
            continue
        sources[int(year)] = {"df": frame, "path": str(path), "label": path.name}
    return sources, messages


def roster_source_for_year(season_rosters, selected_season):
    if season_rosters is None:
        return pd.DataFrame(), None

    selected_season = int(selected_season)
    if isinstance(season_rosters, dict):
        entry = season_rosters.get(selected_season) or season_rosters.get(str(selected_season))
        if entry is None:
            return pd.DataFrame(), None
        if isinstance(entry, dict):
            frame = entry.get("df", pd.DataFrame())
            label = entry.get("path") or entry.get("label") or f"roster_{selected_season}.csv"
        else:
            frame = entry
            label = f"roster_{selected_season}.csv"
    elif isinstance(season_rosters, pd.DataFrame):
        frame = season_rosters
        label = f"roster_{selected_season}.csv"
    else:
        return pd.DataFrame(), None

    if frame is None or frame.empty:
        return pd.DataFrame(), None

    output = frame.copy()
    if "season" in output.columns:
        seasons = pd.to_numeric(output["season"], errors="coerce")
        selected = output[seasons.eq(selected_season)].copy()
        if selected.empty:
            return pd.DataFrame(), None
        output = selected
    return output, label


def load_local_roster(selected_season, historical_roster, season_rosters=None):
    season_roster, season_roster_label = roster_source_for_year(season_rosters, selected_season)
    if not season_roster.empty:
        return season_roster, season_roster_label or f"roster_{int(selected_season)}.csv", "Local roster file"

    if historical_roster is not None and not historical_roster.empty and "season" in historical_roster.columns:
        hist = historical_roster.copy()
        hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
        selected = hist[hist["season"] == selected_season].copy()
        if not selected.empty:
            return selected, "nfl_season_rosters_clean_2005_2025.csv", "Local roster file"
    return pd.DataFrame(), "No roster file found", "Local roster file"


def load_transaction_feed(future_data=None):
    return future_data.get("transactions", pd.DataFrame()).copy() if future_data else pd.DataFrame()


def load_player_stats(future_data):
    future_data = future_data or {}
    stats = {
        "weekly": future_data.get("player_weekly_stats", pd.DataFrame()).copy(),
        "seasonal": future_data.get("player_season_stats", pd.DataFrame()).copy(),
        "snaps": future_data.get("player_snap_counts", pd.DataFrame()).copy(),
        "injuries": future_data.get("injuries", pd.DataFrame()).copy(),
    }
    production_found = not stats["weekly"].empty or not stats["seasonal"].empty
    return stats, production_found


def standardize_player_stats(player_stats):
    standardized = {}
    for key, df in (player_stats or {}).items():
        if df is None or df.empty:
            standardized[key] = pd.DataFrame()
            continue
        output = df.copy()
        output.columns = [str(column).strip().lower() for column in output.columns]
        standardized[key] = output
    return standardized


def live_roster_rows_for_year(live_roster, selected_season):
    if live_roster is None or live_roster.empty:
        return pd.DataFrame()
    output = live_roster.copy()
    if "season" in output.columns:
        seasons = pd.to_numeric(output["season"], errors="coerce")
        output = output[seasons.eq(int(selected_season))].copy()
    return output


def load_live_roster_feed(future_data=None, selected_season=None):
    future_data = future_data or {}
    live_roster = future_data.get("weekly_rosters", pd.DataFrame()).copy()
    if selected_season is not None:
        live_roster = live_roster_rows_for_year(live_roster, selected_season)
    return live_roster


def player_id_key(df):
    id_col = first_existing_column(df, ["gsis_id", "player_id", "player_gsis_id", "nfl_id"])
    if id_col is None:
        return pd.Series("", index=df.index)
    return df[id_col].fillna("").astype(str).str.lower().str.strip()


def player_name_key(df):
    name_col = first_existing_column(df, ["full_name", "player_name", "player_display_name", "name"])
    if name_col is None:
        return pd.Series("", index=df.index)
    return df[name_col].fillna("").astype(str).str.lower().str.strip()


def build_player_match_key(df):
    key = player_id_key(df)
    name_key = player_name_key(df)
    key.loc[key.eq("")] = name_key.loc[key.eq("")]
    return key


def map_depth_position(value):
    pos = "" if pd.isna(value) else str(value).upper().strip()
    aliases = {
        "LDE": "DE",
        "RDE": "DE",
        "LE": "DE",
        "RE": "DE",
        "LDT": "DT",
        "RDT": "DT",
        "NT": "DT",
        "LOLB": "OLB",
        "ROLB": "OLB",
        "WLB": "OLB",
        "SLB": "OLB",
        "MLB": "LB",
        "ILB": "LB",
        "LCB": "CB",
        "RCB": "CB",
        "NB": "CB",
        "FS": "S",
        "SS": "S",
        "LT": "T",
        "RT": "T",
        "LG": "G",
        "RG": "G",
        "PK": "K",
    }
    return aliases.get(pos, pos)


def apply_depth_chart_context(roster, depth_charts, selected_season):
    if roster.empty or depth_charts is None or depth_charts.empty:
        return roster, False

    depth = depth_charts.copy()
    depth.columns = [str(column).strip().lower() for column in depth.columns]
    team_col = first_existing_column(depth, ["team", "club_code"])
    id_col = first_existing_column(depth, ["gsis_id", "player_id"])
    name_col = first_existing_column(depth, ["player_name", "full_name"])
    pos_col = first_existing_column(depth, ["pos_abb", "pos_name", "position", "depth_chart_position"])
    rank_col = first_existing_column(depth, ["pos_rank", "depth_team", "depth_position"])
    if team_col is None or pos_col is None or (id_col is None and name_col is None):
        return roster, False

    season_col = first_existing_column(depth, ["season"])
    if season_col is not None and selected_season is not None:
        selected_depth = depth[pd.to_numeric(depth[season_col], errors="coerce").eq(int(selected_season))].copy()
        if not selected_depth.empty:
            depth = selected_depth

    date_col = first_existing_column(depth, ["dt", "date", "depth_chart_date"])
    if date_col is not None:
        depth["_depth_chart_date"] = pd.to_datetime(depth[date_col], errors="coerce")
        if depth["_depth_chart_date"].notna().any():
            depth = depth[depth["_depth_chart_date"].eq(depth["_depth_chart_date"].max())].copy()

    depth["_team"] = depth[team_col].map(normalize_team)
    depth["_player_key"] = player_id_key(depth) if id_col is not None else pd.Series("", index=depth.index)
    if name_col is not None:
        depth_name_key = depth[name_col].fillna("").astype(str).str.lower().str.strip()
        depth.loc[depth["_player_key"].eq(""), "_player_key"] = depth_name_key.loc[depth["_player_key"].eq("")]
    depth["_depth_chart_position"] = depth[pos_col].map(map_depth_position)
    depth["_pos_rank"] = pd.to_numeric(depth[rank_col], errors="coerce") if rank_col is not None else pd.NA
    depth = depth[depth["_player_key"].astype(str).str.len() > 0].copy()
    if depth.empty:
        return roster, False

    depth = depth.sort_values(["_team", "_player_key", "_pos_rank"], na_position="last")
    depth = depth.drop_duplicates(["_team", "_player_key"], keep="first")
    depth = depth[["_team", "_player_key", "_depth_chart_position", "_pos_rank"]]

    output = roster.copy()
    output["_player_key"] = player_id_key(output)
    name_key = player_name_key(output)
    output.loc[output["_player_key"].eq(""), "_player_key"] = name_key.loc[output["_player_key"].eq("")]
    output["_team"] = output["team"].map(normalize_team) if "team" in output.columns else ""
    output = output.merge(depth, on=["_team", "_player_key"], how="left")

    if "_depth_chart_position" in output.columns:
        output["depth_chart_position"] = output["_depth_chart_position"].combine_first(output.get("depth_chart_position", pd.Series(pd.NA, index=output.index)))
    if "_pos_rank" in output.columns:
        output["depth_chart_rank"] = output["_pos_rank"]

    output = output.drop(columns=["_team", "_player_key", "_depth_chart_position", "_pos_rank"], errors="ignore")
    return output, output.get("depth_chart_rank", pd.Series(dtype=float)).notna().any()


def apply_injury_context(roster, injuries, selected_season):
    if roster.empty or injuries is None or injuries.empty:
        return roster, False

    injury = injuries.copy()
    injury.columns = [str(column).strip().lower() for column in injury.columns]
    season_col = first_existing_column(injury, ["season"])
    if season_col is not None:
        injury = injury[pd.to_numeric(injury[season_col], errors="coerce").eq(int(selected_season))].copy()
    if injury.empty:
        return roster, False

    team_col = first_existing_column(injury, ["team", "club_code"])
    status_col = first_existing_column(injury, ["report_status", "status", "practice_status"])
    if status_col is None:
        return roster, False
    injury["_team"] = injury[team_col].map(normalize_team) if team_col is not None else ""
    injury["_player_key"] = build_player_match_key(injury)
    injury = injury[injury["_player_key"].astype(str).str.len() > 0].copy()
    if injury.empty:
        return roster, False

    week_col = first_existing_column(injury, ["week"])
    injury["_week"] = pd.to_numeric(injury[week_col], errors="coerce") if week_col is not None else 0
    injury = injury.sort_values(["_team", "_player_key", "_week"], na_position="first")
    injury = injury.drop_duplicates(["_team", "_player_key"], keep="last")
    injury = injury[["_team", "_player_key", status_col]].rename(columns={status_col: "_injury_status"})

    output = roster.copy()
    output["_team"] = output["team"].map(normalize_team) if "team" in output.columns else ""
    output["_player_key"] = build_player_match_key(output)
    output = output.merge(injury, on=["_team", "_player_key"], how="left")
    if "_injury_status" in output.columns:
        output["status"] = output["_injury_status"].combine_first(output.get("status", pd.Series(pd.NA, index=output.index)))
    output = output.drop(columns=["_team", "_player_key", "_injury_status"], errors="ignore")
    return output, True


def build_roster_context(local_roster, future_data=None, selected_season=None):
    selected_season = int(selected_season) if selected_season is not None else None
    future_data = future_data or {}
    live_roster = load_live_roster_feed(future_data, selected_season)
    source_files = []
    source_notes = []

    if live_roster is not None and not live_roster.empty:
        roster = live_roster.copy()
        source_files.append("data/live_sources/weekly_rosters.csv")
        source_notes.append("weekly_rosters")
    else:
        roster = local_roster.copy() if local_roster is not None else pd.DataFrame()
        source_notes.append("local roster")

    roster, depth_used = apply_depth_chart_context(roster, future_data.get("depth_charts", pd.DataFrame()), selected_season)
    if depth_used:
        source_files.append("data/live_sources/depth_charts.csv")
        source_notes.append("depth_charts")

    roster, injuries_used = apply_injury_context(roster, future_data.get("injuries", pd.DataFrame()), selected_season)
    if injuries_used:
        source_files.append("data/live_sources/injuries.csv")
        source_notes.append("injuries")

    source_summary = ", ".join(sorted(set(source_notes))) if source_notes else "local roster"
    return roster, {
        "source_files": sorted(set(source_files)),
        "source_notes": sorted(set(source_notes)),
        "used_live_roster": live_roster is not None and not live_roster.empty,
        "used_depth_charts": depth_used,
        "used_injuries": injuries_used,
        "status": f"Using Live Leaderboard roster score from {source_summary} for {selected_season} roster context.",
    }


def merge_live_roster_updates(local_roster, live_roster):
    return local_roster if live_roster is None or live_roster.empty else live_roster


def derive_player_production_scores(stats_df):
    if stats_df is None or stats_df.empty:
        return pd.DataFrame(columns=["player_key", "production_score", "games_played"])

    stats = stats_df.copy()
    stats["player_key"] = build_player_match_key(stats)
    stats = stats[stats["player_key"].astype(str).str.len() > 0].copy()
    if stats.empty:
        return pd.DataFrame(columns=["player_key", "production_score", "games_played"])

    weighted_columns = {
        "passing_yards": 0.006,
        "pass_yards": 0.006,
        "passing_tds": 1.6,
        "pass_tds": 1.6,
        "interceptions": -2.4,
        "sacks": 2.0,
        "qb_hits": 1.3,
        "tackles_for_loss": 1.5,
        "rushing_yards": 0.010,
        "rush_yards": 0.010,
        "receiving_yards": 0.010,
        "rec_yards": 0.010,
        "receptions": 0.12,
        "targets": 0.04,
        "receiving_tds": 1.2,
        "rushing_tds": 1.2,
        "touchdowns": 1.2,
        "fantasy_points": 0.18,
        "fantasy_points_ppr": 0.15,
        "tackles": 0.16,
        "def_tackles": 0.16,
        "passes_defended": 1.2,
        "pass_defended": 1.2,
        "epa": 10.0,
        "passing_epa": 10.0,
        "receiving_epa": 10.0,
        "rushing_epa": 10.0,
    }
    raw = pd.Series(0.0, index=stats.index)
    used = False
    for column, weight in weighted_columns.items():
        if column in stats.columns:
            raw += pd.to_numeric(stats[column], errors="coerce").fillna(0) * weight
            used = True
    if not used:
        return pd.DataFrame(columns=["player_key", "production_score", "games_played"])

    games_col = first_existing_column(stats, ["games", "games_played", "recent_team_games", "week"])
    games_played = pd.to_numeric(stats[games_col], errors="coerce").fillna(0) if games_col else pd.Series(0, index=stats.index)
    player_scores = pd.DataFrame({"player_key": stats["player_key"], "production_raw": raw, "games_played": games_played})
    player_scores = player_scores.groupby("player_key", as_index=False).agg(
        production_raw=("production_raw", "sum"),
        games_played=("games_played", "max"),
    )
    player_scores["production_score"] = normalize_score(player_scores["production_raw"], center=75, spread=12, lower=45, upper=100)
    return player_scores[["player_key", "production_score", "games_played"]]


def build_draft_lookup(historical_roster):
    if historical_roster is None or historical_roster.empty or not {"full_name", "draft_number"}.issubset(historical_roster.columns):
        return {}
    draft = historical_roster.dropna(subset=["full_name", "draft_number"]).copy()
    draft["name_key"] = draft["full_name"].astype(str).str.lower().str.strip()
    draft["draft_number"] = pd.to_numeric(draft["draft_number"], errors="coerce")
    draft = draft.dropna(subset=["draft_number"])
    return draft.groupby("name_key")["draft_number"].min().to_dict()


def standardize_roster_columns(roster, selected_season, historical_roster=None):
    output = roster.copy()
    if output.empty:
        return output
    for column in ["season", "team", "position", "depth_chart_position", "status", "full_name", "draft_number", "years_exp", "entry_year", "rookie_year", "gsis_id"]:
        if column not in output.columns:
            output[column] = pd.NA
    output["season"] = pd.to_numeric(output["season"], errors="coerce").fillna(selected_season).astype(int)
    output["team"] = output["team"].apply(normalize_team)
    output["position"] = output["position"].fillna("").astype(str).str.upper().str.strip()
    output["depth_chart_position"] = output["depth_chart_position"].fillna(output["position"]).astype(str).str.upper().str.strip()
    output["depth_chart_position"] = output["depth_chart_position"].map(map_depth_position)
    output["status"] = output["status"].fillna("ACT").astype(str).str.upper().str.strip()
    output["full_name"] = output["full_name"].fillna("").astype(str).str.strip()
    output["name_key"] = output["full_name"].str.lower()
    output["draft_number"] = pd.to_numeric(output["draft_number"], errors="coerce")
    if historical_roster is not None and not historical_roster.empty:
        output["draft_number"] = output["draft_number"].fillna(output["name_key"].map(build_draft_lookup(historical_roster)))
    for column in ["years_exp", "entry_year", "rookie_year"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["years_exp"] = output["years_exp"].fillna(selected_season - output["entry_year"])
    output["years_exp"] = output["years_exp"].fillna(selected_season - output["rookie_year"])
    output["years_exp"] = output["years_exp"].clip(lower=0, upper=22).fillna(3)
    return output


def position_group(position, depth_position=""):
    pos = str(depth_position or position).upper()
    base = str(position).upper()
    if pos == "QB" or base == "QB":
        return "qb"
    if pos in {"WR", "RB", "FB", "TE"} or base in {"WR", "RB", "FB", "TE"}:
        return "skill"
    if pos in {"OT", "T", "G", "C", "OL", "OG"} or base in {"OT", "T", "G", "C", "OL", "OG"}:
        return "offensive_line"
    if pos in {"EDGE", "DL", "DT", "DE", "OLB", "NT"} or base in {"EDGE", "DL", "DT", "DE", "OLB", "NT"}:
        return "defensive_front"
    if pos in {"CB", "S", "DB", "FS", "SS"} or base in {"CB", "S", "DB", "FS", "SS"}:
        return "secondary"
    if pos in {"K", "P", "LS"} or base in {"K", "P", "LS"}:
        return "special_teams"
    return "other"


def is_premium_position(position, depth_position=""):
    pos = str(depth_position or position).upper()
    base = str(position).upper()
    return pos in {"QB", "WR", "OT", "T", "EDGE", "DE", "DT", "DL", "CB", "DB"} or base in {"QB", "WR", "OT", "T", "EDGE", "DE", "DT", "DL", "CB", "DB"}


def position_importance(position, depth_position=""):
    pos = str(depth_position or position).upper()
    group = position_group(position, depth_position)

    if pos == "QB":
        return 1.14
    if pos in {"EDGE", "DE", "OT", "T", "CB"}:
        return 1.08
    if pos in {"WR", "DL", "DT", "DB"}:
        return 1.04
    if group in {"offensive_line", "defensive_front", "secondary"}:
        return 1.02
    if group == "special_teams":
        return 0.78
    return 0.98 if group == "skill" else 0.92


def calculate_availability_multiplier(status):
    value = str(status).upper().strip()
    if value in {"ACT", "ACTIVE", "FULL"}:
        return 1.0
    if value in {"QUESTIONABLE", "QUE"}:
        return 0.82
    if value in {"DOUBTFUL", "DBT"}:
        return 0.35
    if value in {"IR", "RES", "PUP", "SUS", "RSR"}:
        return 0.14
    if value == "RSN":
        return 0.25
    if value in {"DEV", "EXE", "PRACTICE", "PS"}:
        return 0.35
    if value == "RFA":
        return 0.72
    if value == "UDF":
        return 0.62
    if value in {"UFA", "CUT", "WAIVED", "RELEASED", "NWT", "OUT"}:
        return 0.08
    return 0.88


def calculate_experience_score(row):
    exp = pd.to_numeric(row.get("years_exp"), errors="coerce")
    exp = 3 if pd.isna(exp) else max(0, min(float(exp), 22))
    if exp <= 1:
        score = 58 + 5 * exp
    elif exp <= 4:
        score = 65 + 6 * (exp - 1)
    elif exp <= 8:
        score = 83 + 1.5 * (8 - abs(6 - exp))
    else:
        score = 86 - 2.2 * (exp - 8)
    return float(np.clip(score, 52, 90))


def calculate_draft_capital_score(row):
    pick = pd.to_numeric(row.get("draft_number"), errors="coerce")
    if pd.isna(pick):
        return 48.0
    pick = float(pick)
    if pick <= 32:
        return float(95 - ((pick - 1) / 31) * 10)
    if pick <= 64:
        return float(85 - ((pick - 33) / 31) * 10)
    if pick <= 100:
        return float(78 - ((pick - 65) / 35) * 10)
    if pick <= 170:
        return float(68 - ((pick - 101) / 69) * 10)
    if pick <= 260:
        return float(55 - ((pick - 171) / 89) * 10)
    return 45.0


def calculate_rookie_projection_score(row):
    return float(np.clip(calculate_draft_capital_score(row) * position_importance(row.get("position"), row.get("depth_chart_position")), 40, 98))


def calculate_player_score_blend(player_row, current_week=None):
    fallback = 0.45 * calculate_draft_capital_score(player_row) + 0.40 * calculate_experience_score(player_row) + 0.15 * calculate_rookie_projection_score(player_row)
    production = pd.to_numeric(player_row.get("production_score"), errors="coerce")
    if pd.isna(production):
        return float(np.clip(fallback, 35, 98))
    games_played = pd.to_numeric(player_row.get("games_played"), errors="coerce")
    games_played = 0 if pd.isna(games_played) else games_played
    fallback_weight = 0.90 if games_played <= 2 else 0.65 if games_played <= 5 else 0.40 if games_played <= 9 else 0.15
    return float(np.clip(fallback_weight * fallback + (1 - fallback_weight) * production, 35, 100))


def build_player_stat_scores(roster, player_stats):
    output = roster.copy()
    output["production_score"] = pd.NA
    output["games_played"] = pd.NA
    player_stats = player_stats or {"seasonal": pd.DataFrame(), "weekly": pd.DataFrame()}
    seasonal_scores = derive_player_production_scores(player_stats.get("seasonal", pd.DataFrame()))
    weekly_scores = derive_player_production_scores(player_stats.get("weekly", pd.DataFrame()))
    score_frames = [df for df in [seasonal_scores, weekly_scores] if not df.empty]
    if not score_frames:
        return output, False

    scores = pd.concat(score_frames, ignore_index=True)
    scores = scores.groupby("player_key", as_index=False).agg(
        production_score=("production_score", "max"),
        games_played=("games_played", "max"),
    )
    roster_keys = output[["name_key"]].copy()
    roster_keys["player_key"] = output.get("gsis_id", pd.Series("", index=output.index)).fillna("").astype(str).str.lower().str.strip()
    roster_keys.loc[roster_keys["player_key"].eq(""), "player_key"] = roster_keys.loc[roster_keys["player_key"].eq(""), "name_key"]
    output["player_key"] = roster_keys["player_key"]
    output = output.merge(scores, on="player_key", how="left", suffixes=("", "_from_stats"))
    output["production_score"] = output["production_score_from_stats"].combine_first(output["production_score"])
    output["games_played"] = output["games_played_from_stats"].combine_first(output["games_played"])
    output = output.drop(columns=["production_score_from_stats", "games_played_from_stats"], errors="ignore")
    return output, output["production_score"].notna().any()


def weighted_top_average(values, top_n, neutral=70):
    numeric = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False).head(top_n)

    if numeric.empty:
        return float(neutral)

    top_average = float(np.average(numeric, weights=np.linspace(1.0, 0.42, len(numeric))))
    group_depth_average = float(numeric.mean())
    return float(np.clip(0.72 * top_average + 0.28 * group_depth_average, 45, 100))


def calculate_continuity_score(team_roster, previous_roster):
    if team_roster.empty or previous_roster.empty:
        return 75.0
    current_ids = set(team_roster.get("gsis_id", pd.Series(dtype=str)).dropna().astype(str))
    previous_ids = set(previous_roster.get("gsis_id", pd.Series(dtype=str)).dropna().astype(str))
    if current_ids and previous_ids:
        ratio = len(current_ids.intersection(previous_ids)) / max(len(previous_ids), 1)
    else:
        current_names = set(team_roster["name_key"].dropna().astype(str))
        previous_names = set(previous_roster["name_key"].dropna().astype(str))
        ratio = len(current_names.intersection(previous_names)) / max(len(previous_names), 1)
    return float(50 + 50 * np.clip(ratio, 0, 1))


def build_position_group_scores(team_roster):
    active = team_roster.copy()
    active["weighted_player_score"] = active["player_score"] * active["availability_multiplier"]
    return {
        "qb_score": weighted_top_average(active.loc[active["position_group"] == "qb", "weighted_player_score"], 2, 68),
        "offense_skill_score": weighted_top_average(active.loc[active["position_group"] == "skill", "weighted_player_score"], 8, 70),
        "offensive_line_score": weighted_top_average(active.loc[active["position_group"] == "offensive_line", "weighted_player_score"], 8, 70),
        "defensive_front_score": weighted_top_average(active.loc[active["position_group"] == "defensive_front", "weighted_player_score"], 9, 70),
        "secondary_score": weighted_top_average(active.loc[active["position_group"] == "secondary", "weighted_player_score"], 8, 70),
        "special_teams_score": weighted_top_average(active.loc[active["position_group"] == "special_teams", "weighted_player_score"], 3, 72),
        "premium_position_score": weighted_top_average(active.loc[active["premium_position"], "weighted_player_score"], 10, 70),
    }


def calculate_transaction_impact(team=None, transactions=None):
    return 0.0


def calculate_team_roster_score(roster, selected_season, team_universe, historical_roster=None, transactions=None, player_stats=None):
    neutral = {column: 75.0 for column in ROSTER_SCORE_COLUMNS if column != "team"}
    if roster is None or roster.empty:
        return pd.DataFrame([{**{"team": team}, **neutral} for team in team_universe], columns=ROSTER_SCORE_COLUMNS)
    standardized = standardize_roster_columns(roster, selected_season, historical_roster)
    standardized, _ = build_player_stat_scores(standardized, player_stats or {"seasonal": pd.DataFrame(), "weekly": pd.DataFrame()})
    standardized["availability_multiplier"] = standardized["status"].apply(calculate_availability_multiplier)
    standardized["position_group"] = standardized.apply(lambda row: position_group(row["position"], row["depth_chart_position"]), axis=1)
    standardized["premium_position"] = standardized.apply(lambda row: is_premium_position(row["position"], row["depth_chart_position"]), axis=1)
    standardized["draft_capital_score"] = standardized.apply(calculate_draft_capital_score, axis=1)
    standardized["experience_score"] = standardized.apply(calculate_experience_score, axis=1)
    standardized["rookie_player_score"] = standardized.apply(calculate_rookie_projection_score, axis=1)
    standardized["player_score"] = standardized.apply(calculate_player_score_blend, axis=1)
    standardized["player_score"] = (
        standardized["player_score"]
        * standardized.apply(lambda row: position_importance(row["position"], row["depth_chart_position"]), axis=1)
    ).clip(35, 100)
    previous_all = pd.DataFrame()
    if historical_roster is not None and not historical_roster.empty:
        previous_all = standardize_roster_columns(historical_roster, selected_season - 1, historical_roster)
        previous_all = previous_all[previous_all["season"] == selected_season - 1].copy()
    rows = []
    for team in team_universe:
        team_roster = standardized[standardized["team"] == team].copy()
        if team_roster.empty:
            rows.append({**{"team": team}, **neutral})
            continue
        group_scores = build_position_group_scores(team_roster)
        availability_score = float(np.clip(50 + 50 * team_roster["availability_multiplier"].mean(), 50, 100))
        rookie_mask = (team_roster["years_exp"] <= 1) | (team_roster["entry_year"] >= selected_season - 1)
        rookie_score = weighted_top_average(team_roster.loc[rookie_mask, "rookie_player_score"], 5, 65)
        experience_score = weighted_top_average(team_roster["experience_score"], 30, 72)
        draft_capital_score = weighted_top_average(team_roster["draft_capital_score"], 20, 60)
        previous_roster = previous_all[previous_all["team"] == team].copy() if not previous_all.empty else pd.DataFrame()
        continuity_score = calculate_continuity_score(team_roster, previous_roster)
        transaction_impact = calculate_transaction_impact(team, transactions)
        base = (
            0.18 * group_scores["qb_score"]
            + 0.17 * group_scores["offensive_line_score"]
            + 0.13 * group_scores["offense_skill_score"]
            + 0.17 * group_scores["defensive_front_score"]
            + 0.14 * group_scores["secondary_score"]
            + 0.08 * group_scores["premium_position_score"]
            + 0.08 * availability_score
            + 0.03 * rookie_score
            + 0.02 * continuity_score
        )
        rows.append(
            {
                "team": team,
                "roster_score": float(np.clip(base + transaction_impact, 50, 100)),
                **group_scores,
                "availability_score": availability_score,
                "rookie_projection_score": float(np.clip(rookie_score, 40, 98)),
                "transaction_impact_score": float(transaction_impact),
                "experience_score": float(experience_score),
                "draft_capital_score": float(draft_capital_score),
                "continuity_score": float(continuity_score),
            }
        )
    return pd.DataFrame(rows, columns=ROSTER_SCORE_COLUMNS)
