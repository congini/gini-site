from pathlib import Path
from io import BytesIO
import base64
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Super Square",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# PATH SETUP
# -----------------------------

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav


PRIMARY = "#F15A24"
SECONDARY = "#0073B7"
BRONCOS_LOGO_PATH = APP_DIR / "assets" / "broncos_logo_centered.png"
BRONCOS_LOGO_SOURCE = str(BRONCOS_LOGO_PATH)


SUPER_BOWL_WINNERS = {
    2005: "PIT",
    2006: "IND",
    2007: "NYG",
    2008: "PIT",
    2009: "NO",
    2010: "GB",
    2011: "NYG",
    2012: "BAL",
    2013: "SEA",
    2014: "NE",
    2015: "DEN",
    2016: "NE",
    2017: "PHI",
    2018: "NE",
    2019: "KC",
    2020: "TB",
    2021: "LA",
    2022: "KC",
    2023: "KC",
    2024: "PHI",
    2025: "SEA",
}


@st.cache_data
def load_super_square_data():
    team_season = pd.read_csv(DATA_DIR / "team_season_estat.csv")
    team_game = pd.read_csv(DATA_DIR / "team_game_estat.csv")
    assets_path = DATA_DIR / "teams_colors_logos.csv"
    team_assets = pd.read_csv(assets_path) if assets_path.exists() else pd.DataFrame()
    return team_season, team_game, team_assets


@st.cache_data(show_spinner=False)
def logo_url_to_data_uri(url, grayscale=False):
    if not isinstance(url, str) or not url:
        return ""

    try:
        if url.startswith("http"):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGBA")
        else:
            path = Path(url)
            if not path.is_absolute():
                path = APP_DIR / path
            if not path.exists():
                return ""
            image = Image.open(path).convert("RGBA")

        if grayscale:
            alpha = image.getchannel("A")
            image = ImageOps.grayscale(image).convert("RGBA")
            image.putalpha(alpha)

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"

    except Exception:
        return ""


def build_team_lookups(team_assets):
    logo_lookup = {}
    name_lookup = {}

    if team_assets.empty or "team_abbr" not in team_assets.columns:
        return logo_lookup, name_lookup

    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        abbr = str(row.get("team_abbr"))

        if "team_logo_espn" in team_assets.columns and pd.notna(row.get("team_logo_espn")):
            logo_lookup[abbr] = str(row.get("team_logo_espn"))

        if "team_name" in team_assets.columns and pd.notna(row.get("team_name")):
            name_lookup[abbr] = str(row.get("team_name"))
        else:
            name_lookup[abbr] = abbr

    logo_lookup["DEN"] = BRONCOS_LOGO_SOURCE
    return logo_lookup, name_lookup


def get_score_column(df):
    if "custom_estat" in df.columns:
        return "custom_estat"
    if "overall_estat" in df.columns:
        return "overall_estat"
    raise ValueError("No Gini/EStat score column found.")


CHECKPOINT_PROFILE_COMPONENTS = [
    ("week_6_best_epa_unit_rank", "Week 6 best EPA unit"),
    ("week_8_estat_rank", "Week 8 Gini/EStat"),
    ("week_8_point_diff_rank", "Week 8 point differential"),
    ("week_10_best_epa_unit_rank", "Week 10 best EPA unit"),
    ("week_10_epa_diff_rank", "Week 10 EPA differential"),
    ("week_12_estat_rank", "Week 12 Gini/EStat"),
    ("week_14_epa_diff_rank", "Week 14 EPA differential"),
]

CUSP_CAP = 5
CUSP_FILL_MISS_LIMIT = 8.0


def rank_metric_by_season(df, metric_col, rank_col, ascending=False):
    ranked = df.groupby("season")[metric_col].rank(
        ascending=ascending,
        method="min",
    )
    df[rank_col] = ranked
    return df


def calculate_checkpoint_metrics(team_game, seasons):
    required_cols = [
        "season",
        "season_type",
        "week",
        "team",
        "game_id",
        "estat_game",
        "off_adj_epa",
        "def_adj_epa",
        "point_diff",
    ]
    missing_cols = [column for column in required_cols if column not in team_game.columns]

    if missing_cols:
        raise ValueError(f"Missing required team-game columns: {missing_cols}")

    regular_games = team_game[
        (team_game["season"].isin(seasons))
        & (team_game["season_type"] == "REG")
    ].copy()

    checkpoint_metrics = pd.DataFrame(
        regular_games[["season", "team"]].drop_duplicates()
    )

    for week in (5, 6, 8, 10, 12, 14):
        checkpoint = (
            regular_games[regular_games["week"] <= week]
            .groupby(["season", "team"], as_index=False)
            .agg(
                checkpoint_estat=("estat_game", "mean"),
                checkpoint_off_epa=("off_adj_epa", "mean"),
                checkpoint_def_epa=("def_adj_epa", "mean"),
                checkpoint_point_diff=("point_diff", "mean"),
            )
        )

        checkpoint["checkpoint_epa_diff"] = (
            checkpoint["checkpoint_off_epa"]
            + checkpoint["checkpoint_def_epa"]
        )
        checkpoint["checkpoint_best_epa_unit"] = checkpoint[
            ["checkpoint_off_epa", "checkpoint_def_epa"]
        ].max(axis=1)

        prefix = f"week_{week}"
        rank_metric_by_season(
            checkpoint,
            "checkpoint_estat",
            f"{prefix}_estat_rank",
        )
        rank_metric_by_season(
            checkpoint,
            "checkpoint_epa_diff",
            f"{prefix}_epa_diff_rank",
        )
        rank_metric_by_season(
            checkpoint,
            "checkpoint_best_epa_unit",
            f"{prefix}_best_epa_unit_rank",
        )
        rank_metric_by_season(
            checkpoint,
            "checkpoint_point_diff",
            f"{prefix}_point_diff_rank",
        )

        keep_cols = [
            "season",
            "team",
            f"{prefix}_estat_rank",
            f"{prefix}_epa_diff_rank",
            f"{prefix}_best_epa_unit_rank",
            f"{prefix}_point_diff_rank",
        ]
        checkpoint_metrics = checkpoint_metrics.merge(
            checkpoint[keep_cols],
            on=["season", "team"],
            how="left",
        )

    return checkpoint_metrics


def prepare_super_square_metrics(team_season, team_game):
    score_col = get_score_column(team_season)
    seasons = sorted(
        [
            int(season)
            for season in team_season["season"].dropna().unique()
            if 2005 <= int(season) <= 2025
        ]
    )

    required_cols = [
        score_col,
        "point_diff_per_game",
        "offense_estat",
        "defense_estat",
        "off_adj_epa",
        "def_adj_epa",
    ]
    missing_cols = [column for column in required_cols if column not in team_season.columns]

    if missing_cols:
        raise ValueError(f"Missing required team-season columns: {missing_cols}")

    metrics = team_season[team_season["season"].isin(seasons)].copy()
    metrics["Team"] = metrics["team"]
    metrics["Gini Score"] = pd.to_numeric(metrics[score_col], errors="coerce")
    metrics["Point Differential per Game"] = pd.to_numeric(
        metrics["point_diff_per_game"],
        errors="coerce",
    )
    metrics["Offense EStat"] = pd.to_numeric(
        metrics["offense_estat"],
        errors="coerce",
    )
    metrics["Defense EStat"] = pd.to_numeric(
        metrics["defense_estat"],
        errors="coerce",
    )
    metrics["Adjusted Offensive EPA"] = pd.to_numeric(
        metrics["off_adj_epa"],
        errors="coerce",
    )
    metrics["Adjusted Defensive EPA"] = pd.to_numeric(
        metrics["def_adj_epa"],
        errors="coerce",
    )
    metrics["EPA Differential"] = (
        metrics["Adjusted Offensive EPA"]
        + metrics["Adjusted Defensive EPA"]
    )
    metrics["Best Unit Score"] = metrics[
        ["Offense EStat", "Defense EStat"]
    ].max(axis=1)

    metrics = metrics.dropna(
        subset=[
            "Gini Score",
            "Point Differential per Game",
            "Offense EStat",
            "Defense EStat",
            "Adjusted Offensive EPA",
            "Adjusted Defensive EPA",
            "EPA Differential",
            "Best Unit Score",
        ]
    ).copy()

    rank_metric_by_season(metrics, "Gini Score", "gini_rank")
    rank_metric_by_season(
        metrics,
        "Point Differential per Game",
        "point_diff_rank",
    )
    rank_metric_by_season(metrics, "EPA Differential", "epa_diff_rank")
    rank_metric_by_season(metrics, "Offense EStat", "offense_rank_calc")
    rank_metric_by_season(metrics, "Defense EStat", "defense_rank_calc")

    metrics["best_unit_rank"] = metrics[
        ["offense_rank_calc", "defense_rank_calc"]
    ].min(axis=1)

    checkpoint_metrics = calculate_checkpoint_metrics(team_game, seasons)
    metrics = metrics.merge(
        checkpoint_metrics,
        on=["season", "team"],
        how="left",
    )

    return metrics


def _champion_rows(metrics, winners):
    rows = []

    for season, team in winners.items():
        champion_row = metrics[
            (metrics["season"] == season)
            & (metrics["team"] == team)
        ]

        if champion_row.empty:
            continue

        rows.append(champion_row.iloc[0])

    return pd.DataFrame(rows)


def research_five_gate_super_square(metrics, winners):
    champion_df = _champion_rows(metrics, winners)

    if champion_df.empty:
        raise ValueError("No Super Bowl winner rows found for Super Square research.")

    checkpoint_components = []
    for column, label in CHECKPOINT_PROFILE_COMPONENTS:
        checkpoint_components.append(
            {
                "column": column,
                "label": label,
                "cutoff": int(np.ceil(champion_df[column].max())),
            }
        )

    return {
        "gini_rank_cutoff": int(np.ceil(champion_df["gini_rank"].max())),
        "gini_score_floor": float(champion_df["Gini Score"].min()),
        "point_diff_rank_cutoff": int(np.ceil(champion_df["point_diff_rank"].max())),
        "point_diff_floor": float(champion_df["Point Differential per Game"].min()),
        "epa_diff_rank_cutoff": int(np.ceil(champion_df["epa_diff_rank"].max())),
        "epa_diff_floor": float(champion_df["EPA Differential"].min()),
        "best_unit_rank_cutoff": int(np.ceil(champion_df["best_unit_rank"].max())),
        "best_unit_score_floor": float(champion_df["Best Unit Score"].min()),
        "checkpoint_components": checkpoint_components,
        "valid_champion_count": int(len(champion_df)),
    }


def _value_floor_miss(value_series, floor):
    scale = max(abs(float(floor)), 1.0)
    return ((floor - value_series) / scale).clip(lower=0) * 2


def apply_five_gate_rule(metrics, config):
    rule_df = metrics.copy()

    rule_df["gate_1_overall_gini"] = (
        (rule_df["gini_rank"] <= config["gini_rank_cutoff"])
        & (rule_df["Gini Score"] >= config["gini_score_floor"])
    ).fillna(False)

    rule_df["gate_2_scoreboard_control"] = (
        (rule_df["point_diff_rank"] <= config["point_diff_rank_cutoff"])
        & (
            rule_df["Point Differential per Game"]
            >= config["point_diff_floor"]
        )
    ).fillna(False)

    rule_df["gate_3_efficiency_epa"] = (
        (rule_df["epa_diff_rank"] <= config["epa_diff_rank_cutoff"])
        & (rule_df["EPA Differential"] >= config["epa_diff_floor"])
    ).fillna(False)

    rule_df["gate_4_championship_unit"] = (
        (rule_df["best_unit_rank"] <= config["best_unit_rank_cutoff"])
        & (rule_df["Best Unit Score"] >= config["best_unit_score_floor"])
    ).fillna(False)

    checkpoint_scores = []
    for component in config["checkpoint_components"]:
        checkpoint_scores.append(
            rule_df[component["column"]] / component["cutoff"]
        )

    rule_df["checkpoint_profile_score"] = pd.concat(
        checkpoint_scores,
        axis=1,
    ).max(axis=1)
    rule_df["gate_5_checkpoint_signal"] = (
        rule_df["checkpoint_profile_score"] <= 1
    ).fillna(False)
    rank_metric_by_season(
        rule_df,
        "checkpoint_profile_score",
        "checkpoint_rank",
        ascending=True,
    )

    gate_cols = [
        "gate_1_overall_gini",
        "gate_2_scoreboard_control",
        "gate_3_efficiency_epa",
        "gate_4_championship_unit",
        "gate_5_checkpoint_signal",
    ]
    rule_df["gate_pass_count"] = rule_df[gate_cols].sum(axis=1)
    rule_df["Inside Super Square"] = rule_df[gate_cols].all(axis=1)

    gate_1_miss = np.where(
        rule_df["gate_1_overall_gini"],
        0,
        (rule_df["gini_rank"] - config["gini_rank_cutoff"]).clip(lower=0)
        + _value_floor_miss(rule_df["Gini Score"], config["gini_score_floor"]),
    )
    gate_2_miss = np.where(
        rule_df["gate_2_scoreboard_control"],
        0,
        (
            rule_df["point_diff_rank"]
            - config["point_diff_rank_cutoff"]
        ).clip(lower=0)
        + _value_floor_miss(
            rule_df["Point Differential per Game"],
            config["point_diff_floor"],
        ),
    )
    gate_3_miss = np.where(
        rule_df["gate_3_efficiency_epa"],
        0,
        (rule_df["epa_diff_rank"] - config["epa_diff_rank_cutoff"]).clip(lower=0)
        + _value_floor_miss(rule_df["EPA Differential"], config["epa_diff_floor"]),
    )
    gate_4_miss = np.where(
        rule_df["gate_4_championship_unit"],
        0,
        (
            rule_df["best_unit_rank"]
            - config["best_unit_rank_cutoff"]
        ).clip(lower=0)
        + _value_floor_miss(
            rule_df["Best Unit Score"],
            config["best_unit_score_floor"],
        ),
    )
    gate_5_miss = np.where(
        rule_df["gate_5_checkpoint_signal"],
        0,
        ((rule_df["checkpoint_profile_score"] - 1) * 10).clip(lower=0),
    )

    rule_df["gate_miss_score"] = (
        gate_1_miss
        + gate_2_miss
        + gate_3_miss
        + gate_4_miss
        + gate_5_miss
    )

    return rule_df


def calculate_cusp_status(rule_df, cusp_cap=CUSP_CAP):
    rule_df = rule_df.copy()
    rule_df["On the Cusp"] = False

    cusp_candidates = rule_df[
        (~rule_df["Inside Super Square"])
        & (
            (rule_df["gate_pass_count"] >= 4)
            | (
                (rule_df["gate_pass_count"] == 3)
                & (rule_df["gate_miss_score"] <= CUSP_FILL_MISS_LIMIT)
            )
        )
    ].sort_values(
        [
            "season",
            "gate_pass_count",
            "gate_miss_score",
            "gini_rank",
            "point_diff_rank",
        ],
        ascending=[True, False, True, True, True],
    )

    cusp_index = cusp_candidates.groupby("season").head(cusp_cap).index
    rule_df.loc[cusp_index, "On the Cusp"] = True

    return rule_df


def validate_champions_internal(rule_df, winners):
    records = []

    for season, team in winners.items():
        champion_row = rule_df[
            (rule_df["season"] == season)
            & (rule_df["team"] == team)
        ]

        if champion_row.empty:
            records.append(
                {
                    "season": season,
                    "team": team,
                    "valid_data": False,
                    "inside": False,
                }
            )
            continue

        row = champion_row.iloc[0]
        records.append(
            {
                "season": season,
                "team": team,
                "valid_data": True,
                "inside": bool(row["Inside Super Square"]),
                "status": row["Status"] if "Status" in row else "",
            }
        )

    return pd.DataFrame(records)


def status_label(row):
    if row["Inside Super Square"]:
        return "Inside Super Square"
    if row["On the Cusp"]:
        return "On the Cusp"
    return "Outside"


# Final researched quadrant profile. Each requirement is learned once from the
# historical champions, then mapped into one of the two fixed chart axes.
QUADRANT_REQUIREMENTS = [
    {
        "column": "pd_pos_share_rank",
        "direction": "low",
        "axis": "control",
        "label": "Scoreboard consistency",
        "description": "positive point-differential game share",
    },
    {
        "column": "thru12_estat_rank",
        "direction": "low",
        "axis": "control",
        "label": "Week 12 Gini/EStat",
        "description": "overall EStat profile through Week 12",
    },
    {
        "column": "late12_epa",
        "direction": "high",
        "axis": "control",
        "label": "Late-season EPA floor",
        "description": "EPA differential from Week 12 on",
    },
    {
        "column": "success_rank",
        "direction": "low",
        "axis": "control",
        "label": "Success margin",
        "description": "full-season success-rate margin",
    },
    {
        "column": "thru8_estat",
        "direction": "high",
        "axis": "control",
        "label": "Week 8 EStat floor",
        "description": "early-season EStat signal through Week 8",
    },
    {
        "column": "thru14_epa_rank",
        "direction": "low",
        "axis": "control",
        "label": "Week 14 EPA",
        "description": "EPA differential rank through Week 14",
    },
    {
        "column": "thru10_unit",
        "direction": "high",
        "axis": "pressure",
        "label": "Week 10 unit EPA floor",
        "description": "best EPA unit through Week 10",
    },
    {
        "column": "late14_unit_rank",
        "direction": "low",
        "axis": "pressure",
        "label": "Late unit strength",
        "description": "best EPA unit from Week 14 on",
    },
]


def _zscore_by_season(df, column):
    mean = df.groupby("season")[column].transform("mean")
    std = df.groupby("season")[column].transform("std").replace(0, np.nan)
    return ((df[column] - mean) / std).fillna(0)


def calculate_checkpoint_metrics(team_game, seasons):
    required_cols = [
        "season",
        "season_type",
        "week",
        "team",
        "estat_game",
        "off_adj_epa",
        "def_adj_epa",
        "point_diff",
        "success_margin",
    ]
    missing_cols = [column for column in required_cols if column not in team_game.columns]

    if missing_cols:
        raise ValueError(f"Missing required team-game columns: {missing_cols}")

    regular_games = team_game[
        (team_game["season"].isin(seasons))
        & (team_game["season_type"] == "REG")
    ].copy()
    regular_games["epa_diff_game"] = (
        regular_games["off_adj_epa"] + regular_games["def_adj_epa"]
    )
    regular_games["best_epa_unit_game"] = regular_games[
        ["off_adj_epa", "def_adj_epa"]
    ].max(axis=1)

    checkpoint_metrics = pd.DataFrame(
        regular_games[["season", "team"]].drop_duplicates()
    )

    for week in (5, 6, 8, 10, 12, 14):
        checkpoint = (
            regular_games[regular_games["week"] <= week]
            .groupby(["season", "team"], as_index=False)
            .agg(
                checkpoint_estat=("estat_game", "mean"),
                checkpoint_epa=("epa_diff_game", "mean"),
                checkpoint_unit=("best_epa_unit_game", "mean"),
                checkpoint_point_diff=("point_diff", "mean"),
                checkpoint_success=("success_margin", "mean"),
            )
        )
        prefix = f"thru{week}"

        for suffix, source_col in [
            ("estat", "checkpoint_estat"),
            ("epa", "checkpoint_epa"),
            ("unit", "checkpoint_unit"),
            ("pd", "checkpoint_point_diff"),
            ("success", "checkpoint_success"),
        ]:
            checkpoint[f"{prefix}_{suffix}"] = checkpoint[source_col]
            rank_metric_by_season(
                checkpoint,
                source_col,
                f"{prefix}_{suffix}_rank",
            )

        keep_cols = [
            "season",
            "team",
        ] + [column for column in checkpoint.columns if column.startswith(f"{prefix}_")]
        checkpoint_metrics = checkpoint_metrics.merge(
            checkpoint[keep_cols],
            on=["season", "team"],
            how="left",
        )

    for start_week in (10, 12, 14):
        checkpoint = (
            regular_games[regular_games["week"] >= start_week]
            .groupby(["season", "team"], as_index=False)
            .agg(
                checkpoint_estat=("estat_game", "mean"),
                checkpoint_epa=("epa_diff_game", "mean"),
                checkpoint_unit=("best_epa_unit_game", "mean"),
                checkpoint_point_diff=("point_diff", "mean"),
                checkpoint_success=("success_margin", "mean"),
            )
        )
        prefix = f"late{start_week}"

        for suffix, source_col in [
            ("estat", "checkpoint_estat"),
            ("epa", "checkpoint_epa"),
            ("unit", "checkpoint_unit"),
            ("pd", "checkpoint_point_diff"),
            ("success", "checkpoint_success"),
        ]:
            checkpoint[f"{prefix}_{suffix}"] = checkpoint[source_col]
            rank_metric_by_season(
                checkpoint,
                source_col,
                f"{prefix}_{suffix}_rank",
            )

        keep_cols = [
            "season",
            "team",
        ] + [column for column in checkpoint.columns if column.startswith(f"{prefix}_")]
        checkpoint_metrics = checkpoint_metrics.merge(
            checkpoint[keep_cols],
            on=["season", "team"],
            how="left",
        )

    return checkpoint_metrics


def prepare_super_square_metrics(team_season, team_game):
    seasons = sorted(
        [
            int(season)
            for season in team_game["season"].dropna().unique()
            if 2005 <= int(season) <= 2025
        ]
    )
    regular_games = team_game[
        (team_game["season"].isin(seasons))
        & (team_game["season_type"] == "REG")
    ].copy()

    regular_games["epa_diff_game"] = (
        regular_games["off_adj_epa"] + regular_games["def_adj_epa"]
    )
    regular_games["best_epa_unit_game"] = regular_games[
        ["off_adj_epa", "def_adj_epa"]
    ].max(axis=1)

    metrics = (
        regular_games.groupby(["season", "team"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            point_diff_per_game=("point_diff", "mean"),
            off_epa=("off_adj_epa", "mean"),
            def_epa=("def_adj_epa", "mean"),
            success_margin=("success_margin", "mean"),
            estat=("estat_game", "mean"),
            pd_pos_share=("point_diff", lambda values: (values > 0).mean()),
        )
    )

    metrics["EPA Differential"] = metrics["off_epa"] + metrics["def_epa"]
    metrics["best_epa_unit"] = metrics[["off_epa", "def_epa"]].max(axis=1)
    metrics["off_z"] = _zscore_by_season(metrics, "off_epa")
    metrics["def_z"] = _zscore_by_season(metrics, "def_epa")
    metrics["pd_z"] = _zscore_by_season(metrics, "point_diff_per_game")
    metrics["success_z"] = _zscore_by_season(metrics, "success_margin")
    metrics["Offense EStat"] = 100 + 15 * metrics["off_z"]
    metrics["Defense EStat"] = 100 + 15 * metrics["def_z"]
    metrics["Gini Score"] = 100 + 15 * (
        0.35 * metrics["off_z"]
        + 0.35 * metrics["def_z"]
        + 0.15 * metrics["pd_z"]
        + 0.15 * metrics["success_z"]
    )
    metrics["Best Unit Score"] = metrics[
        ["Offense EStat", "Defense EStat"]
    ].max(axis=1)

    rank_metric_by_season(metrics, "Gini Score", "gini_rank")
    rank_metric_by_season(
        metrics,
        "point_diff_per_game",
        "point_diff_rank",
    )
    rank_metric_by_season(metrics, "EPA Differential", "epa_diff_rank")
    rank_metric_by_season(metrics, "success_margin", "success_rank")
    rank_metric_by_season(metrics, "pd_pos_share", "pd_pos_share_rank")
    rank_metric_by_season(metrics, "Offense EStat", "offense_rank_calc")
    rank_metric_by_season(metrics, "Defense EStat", "defense_rank_calc")
    metrics["best_unit_rank"] = metrics[
        ["offense_rank_calc", "defense_rank_calc"]
    ].min(axis=1)

    checkpoint_metrics = calculate_checkpoint_metrics(team_game, seasons)
    metrics = metrics.merge(
        checkpoint_metrics,
        on=["season", "team"],
        how="left",
    )

    metrics["Team"] = metrics["team"]
    metrics["Point Differential per Game"] = metrics["point_diff_per_game"]
    metrics["checkpoint_rank"] = metrics["thru12_estat_rank"]

    return metrics


def research_five_gate_super_square(metrics, winners):
    champion_df = _champion_rows(metrics, winners)

    if champion_df.empty:
        raise ValueError("No Super Bowl winner rows found for Super Square research.")

    requirements = []
    for requirement in QUADRANT_REQUIREMENTS:
        column = requirement["column"]
        direction = requirement["direction"]
        cutoff = (
            float(champion_df[column].max())
            if direction == "low"
            else float(champion_df[column].min())
        )
        requirements.append({**requirement, "cutoff": cutoff})

    return {
        "requirements": requirements,
        "x_cut": 100.0,
        "y_cut": 100.0,
        "valid_champion_count": int(len(champion_df)),
    }


def _requirement_pass(series, requirement):
    if requirement["direction"] == "low":
        return series <= requirement["cutoff"]

    return series >= requirement["cutoff"]


def _requirement_miss(series, requirement):
    cutoff = requirement["cutoff"]

    if requirement["direction"] == "low":
        return (series - cutoff).clip(lower=0)

    scale = max(abs(float(cutoff)), 0.05)
    return ((cutoff - series) / scale).clip(lower=0)


def _gate_score(rule_df, requirement):
    series = pd.to_numeric(rule_df[requirement["column"]], errors="coerce")
    cutoff = float(requirement["cutoff"])

    if requirement["direction"] == "low":
        score = 100 + 5 * (cutoff - series)
    else:
        season_std = (
            rule_df.groupby("season")[requirement["column"]]
            .transform("std")
            .replace(0, np.nan)
        )
        fallback_scale = max(abs(cutoff), 0.05)
        scale = season_std.fillna(fallback_scale).clip(lower=fallback_scale)
        score = 100 + 18 * ((series - cutoff) / scale)

    return score.clip(lower=30, upper=170).fillna(30)


def apply_five_gate_rule(metrics, config):
    rule_df = metrics.copy()
    gate_cols = []
    miss_cols = []
    control_scores = []
    pressure_scores = []

    for index, requirement in enumerate(config["requirements"], start=1):
        gate_col = f"gate_{index}"
        miss_col = f"gate_{index}_miss"
        score_col = f"gate_{index}_score"
        rule_df[gate_col] = _requirement_pass(
            rule_df[requirement["column"]],
            requirement,
        ).fillna(False)
        rule_df[miss_col] = _requirement_miss(
            rule_df[requirement["column"]],
            requirement,
        ).fillna(999)
        rule_df[score_col] = _gate_score(rule_df, requirement)
        gate_cols.append(gate_col)
        miss_cols.append(miss_col)

        if requirement["axis"] == "control":
            control_scores.append(score_col)
        else:
            pressure_scores.append(score_col)

    rule_df["gate_pass_count"] = rule_df[gate_cols].sum(axis=1)
    rule_df["gate_miss_score"] = rule_df[miss_cols].sum(axis=1)
    rule_df["Control Profile Score"] = rule_df[control_scores].min(axis=1)
    rule_df["Unit Pressure Score"] = rule_df[pressure_scores].min(axis=1)
    rule_df["Clears Control Axis"] = rule_df["Control Profile Score"] >= config["x_cut"]
    rule_df["Clears Unit Pressure Axis"] = rule_df["Unit Pressure Score"] >= config["y_cut"]
    rule_df["Inside Super Square"] = rule_df[gate_cols].all(axis=1)
    rule_df["Quadrant"] = np.select(
        [
            rule_df["Clears Control Axis"] & rule_df["Clears Unit Pressure Axis"],
            ~rule_df["Clears Control Axis"] & rule_df["Clears Unit Pressure Axis"],
            rule_df["Clears Control Axis"] & ~rule_df["Clears Unit Pressure Axis"],
        ],
        [
            "Q4 - Super Bowl Contenders",
            "Q3 - Playoff Contenders",
            "Q2 - Middle Tier",
        ],
        default="Q1 - Non-Contenders",
    )

    return rule_df


def calculate_cusp_status(rule_df, cusp_cap=CUSP_CAP):
    rule_df = rule_df.copy()
    rule_df["On the Cusp"] = False
    rule_df.loc[rule_df["Quadrant"] == "Q3 - Playoff Contenders", "On the Cusp"] = True

    return rule_df


# -----------------------------
# LOAD DATA + INTERNAL RESEARCH
# -----------------------------

team_season, team_game, team_assets = load_super_square_data()
logo_lookup, name_lookup = build_team_lookups(team_assets)

try:
    super_square_data = prepare_super_square_metrics(team_season, team_game)
    gate_config = research_five_gate_super_square(
        super_square_data,
        SUPER_BOWL_WINNERS,
    )
    super_square_data = apply_five_gate_rule(super_square_data, gate_config)
    super_square_data = calculate_cusp_status(super_square_data)
    super_square_data["Status"] = super_square_data.apply(status_label, axis=1)
    champion_validation = validate_champions_internal(
        super_square_data,
        SUPER_BOWL_WINNERS,
    )
except ValueError as err:
    st.error(str(err))
    st.stop()

valid_champion_validation = champion_validation[
    champion_validation["valid_data"]
].copy()
all_valid_champions_inside = (
    not valid_champion_validation.empty
    and bool(valid_champion_validation["inside"].all())
)


# -----------------------------
# PAGE CSS
# -----------------------------

st.markdown(
    f"""
<style>
.stApp,
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 8% 18%, rgba(0,115,183,0.11), transparent 34%),
        radial-gradient(circle at 92% 22%, rgba(241,90,36,0.10), transparent 36%),
        #F6F8FB !important;
    overflow-x: hidden;
}}

.block-container {{
    position: relative;
    z-index: 1;
    padding-top: 0.35rem !important;
    padding-bottom: 2.8rem !important;
}}

.super-page {{
    position: relative;
    z-index: 1;
    max-width: 1500px;
    margin: -1.85rem auto 0 auto;
    padding: 0 1.5rem 3rem 1.5rem;
}}

.super-hero {{
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 0.95fr) minmax(360px, 1.05fr);
    gap: 1.5rem;
    align-items: center;
    margin-top: 0.35rem;
    margin-bottom: 1rem;
    padding: 2.35rem 2.6rem;
    border-radius: 20px;
    background:
        radial-gradient(circle at 88% 20%, rgba(241,90,36,0.28), transparent 32%),
        radial-gradient(circle at 70% 92%, rgba(0,115,183,0.22), transparent 30%),
        linear-gradient(135deg, #07111f 0%, #172033 56%, #273447 100%);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 22px 46px rgba(15, 23, 42, 0.16);
}}

.super-hero::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 5px;
    width: 100%;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
}}

.super-kicker {{
    color: {PRIMARY};
    font-size: 0.8rem;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 0.55rem;
}}

.super-title {{
    color: #FFFFFF;
    font-size: clamp(2.15rem, 4vw, 3.2rem);
    line-height: 1;
    font-weight: 950;
    margin-bottom: 0.72rem;
}}

.super-subtitle {{
    max-width: 920px;
    color: rgba(255,255,255,0.84);
    font-size: 0.98rem;
    line-height: 1.68;
}}

.super-proof-card {{
    position: relative;
    justify-self: stretch;
    width: 100%;
    max-width: none;
    min-height: 150px;
    padding: 1.2rem 1.45rem;
    border-radius: 17px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.18), rgba(255,255,255,0.075)),
        linear-gradient(135deg, rgba(7,17,31,0.08), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.24);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.14),
        0 16px 34px rgba(0,0,0,0.13);
    backdrop-filter: blur(12px);
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 0.7fr) minmax(220px, 1fr);
    gap: 1rem;
    align-items: center;
}}

.super-proof-card::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 5px;
    height: 100%;
    background: linear-gradient(180deg, {PRIMARY}, {SECONDARY});
    opacity: 0.95;
}}

.super-proof-card::after {{
    content: "";
    position: absolute;
    right: -55px;
    bottom: -65px;
    width: 150px;
    height: 150px;
    border-radius: 999px;
    background: rgba(255,255,255,0.055);
    pointer-events: none;
}}

.super-proof-label {{
    color: rgba(255,255,255,0.66);
    font-size: 0.72rem;
    line-height: 1.2;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.super-proof-number {{
    color: #FFFFFF;
    font-size: 2.05rem;
    line-height: 1;
    font-weight: 950;
    letter-spacing: -0.04em;
    margin-bottom: 0.4rem;
}}

.super-proof-text {{
    color: rgba(255,255,255,0.9);
    font-size: 1rem;
    line-height: 1.6;
    font-weight: 750;
    max-width: none;
    width: 100%;
}}

.super-proof-text span {{
    color: #FFFFFF;
    font-weight: 950;
}}

.streak-main {{
    position: relative;
    z-index: 1;
}}

.streak-progress-card {{
    position: relative;
    z-index: 1;
    min-height: 92px;
    padding: 0.95rem 1.05rem;
    border-radius: 15px;
    background: rgba(255,255,255,0.11);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.streak-progress-track {{
    position: relative;
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    overflow: hidden;
    margin-bottom: 0.72rem;
}}

.streak-progress-fill {{
    position: absolute;
    inset: 0;
    width: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, {SECONDARY}, {PRIMARY});
    box-shadow: 0 0 16px rgba(241,90,36,0.28);
}}

.streak-progress-label-row {{
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 0.75rem;
}}

.streak-progress-year {{
    color: rgba(255,255,255,0.68);
    font-size: 0.72rem;
    line-height: 1;
    font-weight: 850;
}}

.streak-progress-year:last-child {{
    text-align: right;
}}

.streak-progress-center {{
    color: rgba(255,255,255,0.90);
    font-size: 0.82rem;
    line-height: 1.15;
    font-weight: 950;
    text-align: center;
    white-space: nowrap;
}}

.super-proof-label,
.super-proof-number,
.super-proof-text {{
    position: relative;
    z-index: 1;
}}

.super-controls-card-marker,
.super-reveal-field-marker {{
    display: none;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) {{
    display: grid;
    grid-template-columns: minmax(360px, 720px) minmax(260px, 340px);
    align-items: start;
    justify-content: center;
    gap: 1.25rem;

    width: min(100%, 1120px) !important;
    margin: 0.85rem auto 1rem auto;
    padding: 0.9rem 1.05rem 1rem 1.05rem;

    border-radius: 16px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {PRIMARY};
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
    backdrop-filter: blur(9px);
    overflow: hidden;
    transform: translateX(5px);
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) > div {{
    width: 100% !important;
    min-width: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) > div:has(.super-reveal-field-marker) {{
    padding-top: 2.53rem !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stSelectbox"] {{
    margin-bottom: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stSelectbox"] > label {{
    margin-bottom: 0.3rem !important;
}}

.super-controls-title {{
    color: #07111f;
    font-size: 1.05rem;
    line-height: 1.15;
    font-weight: 950;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] {{
    margin: 0 !important;
    padding: 0 !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] label {{
    margin: 0 !important;
    padding: 0 !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.34rem !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] label > span:first-child {{
    transform: translateY(-2px);
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] label > div {{
    padding-left: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] p {{
    margin: 0 !important;
    color: #07111f !important;
    font-size: 0.95rem !important;
    font-weight: 850 !important;
    line-height: 1.2 !important;
}}

.super-info-box {{
    margin: 1rem auto 1.2rem auto;
    width: min(100%, 1120px);
    padding: 1.05rem 1.15rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {SECONDARY};
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
    color: #334155;
    font-size: 0.94rem;
    line-height: 1.65;
}}

.super-info-title {{
    color: #07111f;
    font-size: 1.02rem;
    font-weight: 950;
    margin-bottom: 0.45rem;
}}

.super-info-box b {{
    color: #07111f;
}}

.super-info-box p {{
    margin: 0 0 0.95rem 0;
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.65;
}}

.super-info-box p:last-child {{
    margin-bottom: 0;
}}

@media screen and (max-width: 900px) {{
    div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) {{
        grid-template-columns: 1fr;
        width: 100% !important;
        gap: 0.75rem;
        transform: none;
    }}

    div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] {{
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
}}

.square-summary {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    width: min(100%, 1120px);
    margin: 0.6rem auto 1rem auto;
}}

.super-note {{
    width: min(100%, 1120px);
    margin: 0.85rem auto 0.75rem auto;
    color: #64748B;
    font-size: 0.86rem;
    line-height: 1.55;
    text-align: center;
}}

.square-pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 36px;
    padding: 0.55rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(15,23,42,0.10);
    color: #07111f;
    font-size: 0.84rem;
    font-weight: 850;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
}}

.square-pill span {{
    color: {PRIMARY};
    font-weight: 950;
}}

div[data-testid="stSelectbox"] label p,
div[data-testid="stCheckbox"] label p {{
    color: #07111f !important;
    font-size: 0.95rem !important;
    font-weight: 850 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045) !important;
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF !important;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 18px;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.075);
}}

.super-expander-note {{
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.65;
}}

@media screen and (max-width: 1100px) {{
    .super-hero {{
        grid-template-columns: 1fr;
        padding: 2rem;
    }}

    .super-proof-card {{
        justify-self: start;
    }}

    .super-page {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}
}}

@media screen and (max-width: 650px) {{
    .super-page {{
        margin-top: -0.75rem;
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }}

    .super-hero {{
        padding: 1.25rem;
        border-radius: 16px;
    }}

    .super-proof-card {{
        grid-template-columns: 1fr;
        padding: 1rem;
    }}

    .super-title {{
        font-size: clamp(1.85rem, 10vw, 2.35rem);
    }}

    .super-proof-number {{
        font-size: 1.65rem;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# NAV + HERO
# -----------------------------

render_top_nav("Super Square", PRIMARY, SECONDARY)

st.markdown('<div class="super-page">', unsafe_allow_html=True)

validated_seasons_text = f"{gate_config['valid_champion_count']} Straight Seasons"
validation_claim = (
    "The Super Bowl champion has landed in Q4."
    if all_valid_champions_inside
    else "The page checks each listed champion against the fixed quadrant profile using the available project data."
)
hero_claim = (
    "For every season with valid project data, the eventual Super Bowl champion finished inside the fixed Q4 contender zone."
    if all_valid_champions_inside
    else "The Super Square checks whether each eventual champion was inside or near the fixed quadrant profile."
)

championship_progress_label = (
    f"{gate_config['valid_champion_count']} of "
    f"{gate_config['valid_champion_count']} champs captured"
)

st.markdown(
f"""
<div class="super-hero">
<div class="super-hero-copy">
<div class="super-kicker">Historical Super Bowl winner profile</div>
<div class="super-title">Super Square</div>
<div class="super-subtitle">
The Super Square studies the regular-season traits that repeatedly show up in Super Bowl champions. {hero_claim}
</div>
</div>

<div class="super-proof-card">
<div class="streak-main">
<div class="super-proof-label">Championship Streak</div>
<div class="super-proof-number">{validated_seasons_text}</div>
<div class="super-proof-text">
{validation_claim}
</div>
</div>

<div class="streak-progress-card">
<div class="streak-progress-track">
<div class="streak-progress-fill"></div>
</div>

<div class="streak-progress-label-row">
<div class="streak-progress-year">2005 season</div>
<div class="streak-progress-center">{championship_progress_label}</div>
<div class="streak-progress-year">2025 season</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# CONTROLS
# -----------------------------

available_seasons = sorted(
    [
        int(season)
        for season in super_square_data["season"].dropna().unique()
        if 2005 <= int(season) <= 2025
    ],
    reverse=True,
)

season_col, reveal_col = st.columns([1, 1], gap="large")

with season_col:
    st.markdown(
        '<div class="super-controls-card-marker"></div><div class="super-controls-title">Super Square Setup</div>',
        unsafe_allow_html=True,
    )

    selected_season = st.selectbox(
        "Season",
        available_seasons,
        index=0,
        key="super_square_selected_season",
        width="stretch",
    )

with reveal_col:
    st.markdown('<div class="super-reveal-field-marker"></div>', unsafe_allow_html=True)

    reveal_winner = st.checkbox(
        "Reveal Super Bowl winner",
        value=False,
        key="super_square_reveal_winner",
        help="Greys out every logo except the team that won the Super Bowl after this regular season.",
    )

st.markdown(
    """
<div class="super-note">
    The 2005 NFL season maps to the Super Bowl played in 2006. The chart uses fixed composite axes: Control Profile Score and Unit Pressure Score. The 100-point quadrant lines do not move by season.
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# SUPER SQUARE LOGIC
# -----------------------------

season_df = super_square_data[
    super_square_data["season"] == selected_season
].copy()

if season_df.empty:
    st.error("No Super Square data found for the selected season.")
    st.stop()

season_df["Team Name"] = season_df["Team"].map(name_lookup).fillna(season_df["Team"])

champion = SUPER_BOWL_WINNERS.get(selected_season)
season_df["Is Champion"] = season_df["Team"] == champion
season_df["Super Bowl Champion"] = np.where(
    season_df["Is Champion"],
    "Yes",
    "No",
)

for rank_col in [
    "gini_rank",
    "point_diff_rank",
    "epa_diff_rank",
    "best_unit_rank",
    "checkpoint_rank",
]:
    season_df[f"{rank_col}_label"] = season_df[rank_col].apply(
        lambda value: "N/A" if pd.isna(value) else str(int(value))
    )

season_df["control_score_label"] = season_df["Control Profile Score"].round(1)
season_df["unit_pressure_score_label"] = season_df["Unit Pressure Score"].round(1)

inside_count = int(season_df["Inside Super Square"].sum())
cusp_count = int(season_df["On the Cusp"].sum())

winner_text = "Hidden"
if reveal_winner and champion:
    winner_name = season_df.loc[season_df["Team"] == champion, "Team Name"]
    winner_text = winner_name.iloc[0] if not winner_name.empty else champion

st.markdown(
    f"""
<div class="square-summary">
    <div class="square-pill">Inside Super Square:&nbsp;<span>{inside_count}</span></div>
    <div class="square-pill">On the Cusp:&nbsp;<span>{cusp_count}</span></div>
    <div class="square-pill">Super Bowl Winner:&nbsp;<span>{winner_text}</span></div>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# CHART
# -----------------------------

x_cut = float(gate_config.get("x_cut", 100.0))
y_cut = float(gate_config.get("y_cut", 100.0))

x_values = season_df["Control Profile Score"]
y_values = season_df["Unit Pressure Score"]

x_span = max(x_values.max() - x_values.min(), 1)
y_span = max(y_values.max() - y_values.min(), 1)

x_pad = x_span * 0.08
y_pad = y_span * 0.12

x_min = min(x_values.min() - x_pad, x_cut - 45)
x_max = max(x_values.max() + x_pad, x_cut + 45)
y_min = min(y_values.min() - y_pad, y_cut - 45)
y_max = max(y_values.max() + y_pad, y_cut + 45)

fig = px.scatter(
    season_df,
    x="Control Profile Score",
    y="Unit Pressure Score",
    hover_name="Team Name",
    custom_data=[
        "Team",
        "Status",
        "Quadrant",
        "control_score_label",
        "unit_pressure_score_label",
        "gini_rank_label",
        "point_diff_rank_label",
        "epa_diff_rank_label",
        "best_unit_rank_label",
        "checkpoint_rank_label",
        "Super Bowl Champion",
    ],
    title=f"{selected_season} Super Square",
)

fig.update_traces(
    marker=dict(
        size=18,
        color="rgba(0,0,0,0.01)",
        line=dict(width=0),
    ),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Team: %{customdata[0]}<br>"
        "Status: %{customdata[1]}<br>"
        "Quadrant: %{customdata[2]}<br>"
        "Control Profile Score: %{customdata[3]}<br>"
        "Unit Pressure Score: %{customdata[4]}<br>"
        "Gini Rank: %{customdata[5]}<br>"
        "Point Differential Rank: %{customdata[6]}<br>"
        "EPA Differential Rank: %{customdata[7]}<br>"
        "Best Unit Rank: %{customdata[8]}<br>"
        "Checkpoint Rank: %{customdata[9]}<br>"
        "Super Bowl Champion: %{customdata[10]}<extra></extra>"
    ),
)

quadrant_shapes = [
    (x_min, x_cut, y_min, y_cut, "rgba(148,163,184,0.045)"),
    (x_cut, x_max, y_min, y_cut, "rgba(15,23,42,0.035)"),
    (x_min, x_cut, y_cut, y_max, "rgba(0,115,183,0.055)"),
    (x_cut, x_max, y_cut, y_max, "rgba(241,90,36,0.08)"),
]

for x0, x1, y0, y1, fillcolor in quadrant_shapes:
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line=dict(width=0),
        fillcolor=fillcolor,
        layer="below",
    )

fig.add_shape(
    type="line",
    x0=x_cut,
    x1=x_cut,
    y0=y_min,
    y1=y_max,
    line=dict(color="#CBD5E1", width=2, dash="dash"),
    layer="below",
)
fig.add_shape(
    type="line",
    x0=x_min,
    x1=x_max,
    y0=y_cut,
    y1=y_cut,
    line=dict(color="#CBD5E1", width=2, dash="dash"),
    layer="below",
)

fig.add_annotation(
    x=x_cut + (x_max - x_cut) * 0.63,
    y=y_cut + (y_max - y_cut) * 0.90,
    text="<b>Q4</b> · Super Bowl Contenders",
    showarrow=False,
    align="center",
    font=dict(size=12, color="#07111f", family="Arial, sans-serif"),
    bgcolor="rgba(255,255,255,0.94)",
    bordercolor=PRIMARY,
    borderwidth=2,
    borderpad=5,
)

fig.add_annotation(
    x=x_min + (x_cut - x_min) * 0.33,
    y=y_cut + (y_max - y_cut) * 0.90,
    text="<b>Q3</b> · Playoff Contenders",
    showarrow=False,
    align="center",
    font=dict(size=12, color="#07111f", family="Arial, sans-serif"),
    bgcolor="rgba(255,255,255,0.92)",
    bordercolor=SECONDARY,
    borderwidth=2,
    borderpad=5,
)

fig.add_annotation(
    x=x_cut + (x_max - x_cut) * 0.44,
    y=y_min + (y_cut - y_min) * 0.16,
    text="<b>Q2</b> · Middle Tier",
    showarrow=False,
    align="center",
    font=dict(size=11, color="#475569", family="Arial, sans-serif"),
    bgcolor="rgba(255,255,255,0.88)",
    bordercolor="#CBD5E1",
    borderwidth=1,
    borderpad=5,
)

fig.add_annotation(
    x=x_min + (x_cut - x_min) * 0.28,
    y=y_min + (y_cut - y_min) * 0.16,
    text="<b>Q1</b> · Non-Contenders",
    showarrow=False,
    align="center",
    font=dict(size=11, color="#64748B", family="Arial, sans-serif"),
    bgcolor="rgba(255,255,255,0.88)",
    bordercolor="#CBD5E1",
    borderwidth=1,
    borderpad=5,
)

logo_df = season_df.copy()
logo_df["Logo URL"] = logo_df["Team"].map(logo_lookup).fillna("")

logo_df["Draw Order"] = 0
logo_df.loc[logo_df["On the Cusp"], "Draw Order"] = 1
logo_df.loc[logo_df["Inside Super Square"], "Draw Order"] = 2
logo_df.loc[logo_df["Is Champion"], "Draw Order"] = 3
logo_df = logo_df.sort_values("Draw Order")

base_logo_width = x_span * 0.052
base_logo_height = y_span * 0.085

for _, row in logo_df.iterrows():
    is_champion = bool(row["Is Champion"])

    if reveal_winner and champion:
        grayscale = not is_champion
        opacity = 1.0 if is_champion else 0.13
        size_boost = 1.40 if is_champion else 0.82
    else:
        if row["Inside Super Square"]:
            grayscale = False
            opacity = 1.0
            size_boost = 1.14
        elif row["On the Cusp"]:
            grayscale = False
            opacity = 0.70
            size_boost = 0.98
        else:
            grayscale = True
            opacity = 0.25
            size_boost = 0.86

    if row["Team"] == "DEN":
        size_boost *= 1.10

    logo_source = logo_url_to_data_uri(row["Logo URL"], grayscale=grayscale)

    if not logo_source:
        continue

    fig.add_layout_image(
        dict(
            source=logo_source,
            xref="x",
            yref="y",
            x=row["Control Profile Score"],
            y=row["Unit Pressure Score"],
            sizex=base_logo_width * size_boost,
            sizey=base_logo_height * size_boost,
            xanchor="center",
            yanchor="middle",
            sizing="contain",
            opacity=opacity,
            layer="above",
        )
    )

if reveal_winner and champion:
    winner_row = season_df[season_df["Team"] == champion]
    if not winner_row.empty:
        winner = winner_row.iloc[0]
        fig.add_annotation(
            x=winner["Control Profile Score"],
            y=winner["Unit Pressure Score"],
            text=f"{winner['Team Name']}<br>Super Bowl Champion",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-52,
            font=dict(size=13, color="#07111f"),
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=PRIMARY,
            borderwidth=2,
            borderpad=6,
        )

fig.update_layout(
    height=650,
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    title=dict(
        text=f"{selected_season} Super Square",
        font=dict(color="#07111f", size=18),
        x=0.01,
        xanchor="left",
    ),
    xaxis=dict(
        title="Control Profile Score",
        range=[x_min, x_max],
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#334155"),
    ),
    yaxis=dict(
        title="Unit Pressure Score",
        range=[y_min, y_max],
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#334155"),
    ),
    margin=dict(l=65, r=35, t=60, b=60),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# EXPLANATION
# -----------------------------

st.markdown(
    """
<div class="super-info-box">
<div class="super-info-title">How the Super Square works</div>

<p>The Super Square is a historical contender test built from regular-season team profiles. Instead of only looking at record or final score, it asks whether a team matched the type of profile that past Super Bowl winners had before the playoffs.</p>

<p>The chart uses two fixed scores:</p>

<p><b>Control Profile Score</b> is the horizontal axis. It measures the team’s broader control profile through scoreboard consistency, Week 8 and Week 12 Gini/EStat checkpoints, success margin, late-season EPA, and Week 14 EPA strength.</p>

<p><b>Unit Pressure Score</b> is the vertical axis. It measures whether the team had a strong enough offensive or defensive EPA unit, especially around midseason and late in the season.</p>

<p><b>Inside Super Square</b> means the team clears both fixed axes and passes every underlying championship-profile requirement. <b>On the Cusp</b> means the team has strong unit pressure, but falls short of the full control profile. <b>Outside</b> means the team did not match enough of the historical championship profile.</p>

<p><b>Reveal Super Bowl winner</b> highlights the eventual champion from that season so you can compare where the champion landed relative to the Super Square.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
