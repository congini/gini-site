from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


st.set_page_config(
    page_title="Predictive Model",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav


PRIMARY = "#F15A24"
SECONDARY = "#0073B7"
PAGE_BG = "#F6F8FB"
TEXT = "#07111f"
MUTED = "#64748B"
GRID = "#E5E7EB"

DISPLAY_LABELS = {
    "season": "Season",
    "team": "Team",
    "current_wins": "Selected Season Wins",
    "current_losses": "Selected Season Losses",
    "current_ties": "Selected Season Ties",
    "next_season_wins": "Following-Season Wins",
    "wins_change": "Win Change Next Year",
    "predicted_next_wins": "Projected Following-Season Wins",
    "prob_10_plus_wins": "10+ Win Probability",
    "gini_score": "Gini Score",
    "gini_rank": "Gini Rank",
    "overall_estat": "Gini Score",
    "offense_estat": "Offense Gini",
    "defense_estat": "Defense Gini",
    "point_diff_per_game": "Point Differential per Game",
    "point_diff_rank": "Point Differential Rank",
    "off_adj_epa": "Adjusted Offensive EPA",
    "def_adj_epa": "Adjusted Defensive EPA",
    "success_margin": "Success Margin",
    "turnover_margin_per_game": "Turnover Margin per Game",
    "penalty_yards_margin_per_game": "Penalty Yard Margin per Game",
    "schedule_strength": "Schedule Strength",
    "sos_rank": "Schedule Strength Rank",
    "offense_rank": "Offense Rank",
    "defense_rank": "Defense Rank",
    "balance_gap": "Offense/Defense Balance Gap",
    "elite_offense": "Elite Offense",
    "elite_defense": "Elite Defense",
    "two_way_team": "Two-Way Team",
    "top_gini_team": "Top Gini Team",
    "top_point_diff_team": "Top Point Differential Team",
    "super_square_profile": "Super Square Profile",
    "underperformer_signal": "Underperformer",
    "overperformer_signal": "Overperformer",
    "postseason_wins": "Postseason Wins",
    "postseason_games": "Postseason Games",
}

BINARY_COLUMNS = {
    "elite_offense",
    "elite_defense",
    "two_way_team",
    "top_gini_team",
    "top_point_diff_team",
    "super_square_profile",
    "underperformer_signal",
    "overperformer_signal",
}

INTEGER_DISPLAY_COLUMNS = {
    "Season",
    "Selected Season Wins",
    "Following-Season Wins",
    "Projected Following-Season Wins",
    "Actual Following Season Wins",
    "Win Change Next Year",
    "Gini Rank",
    "Point Differential Rank",
    "Offense Rank",
    "Defense Rank",
    "Schedule Strength Rank",
    "Team-Seasons",
}

ONE_DECIMAL_DISPLAY_COLUMNS = {
    "Gini Score",
    "Offense Gini",
    "Defense Gini",
    "Point Differential per Game",
    "Schedule Strength",
    "Average Following-Season Wins",
}

THREE_DECIMAL_DISPLAY_COLUMNS = {
    "Adjusted Offensive EPA",
    "Adjusted Defensive EPA",
    "Success Margin",
}


# -----------------------------
# Data helpers
# -----------------------------


@st.cache_data(show_spinner=False)
def load_data():
    files = {
        "team_season": DATA_DIR / "team_season_estat.csv",
        "team_game": DATA_DIR / "team_game_estat.csv",
        "games": DATA_DIR / "games_2005_onward.csv",
        "roster": DATA_DIR / "nfl_season_rosters_clean_2005_2025.csv",
        "team_assets": DATA_DIR / "teams_colors_logos.csv",
    }

    data = {}
    messages = []

    for key, path in files.items():
        if not path.exists():
            data[key] = pd.DataFrame()
            messages.append(f"Missing file: {path.name}")
            continue

        try:
            data[key] = pd.read_csv(path)
        except Exception as exc:
            data[key] = pd.DataFrame()
            messages.append(f"Could not read {path.name}: {exc}")

    return data, messages


def clean_numeric_columns(df, columns):
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def standardize_team_season(df):
    output = df.copy()
    if "team" in output.columns:
        output["team"] = output["team"].astype(str).str.strip()
    if "season" in output.columns:
        output["season"] = pd.to_numeric(output["season"], errors="coerce")
        output = output.dropna(subset=["season"])
        output["season"] = output["season"].astype(int)
    return output


def calculate_team_records(team_game, games):
    record_cols = ["season", "team", "current_wins", "current_losses", "current_ties", "scored_games"]
    postseason_cols = ["season", "team", "postseason_wins", "postseason_games"]

    records = pd.DataFrame(columns=record_cols)
    postseason_records = pd.DataFrame(columns=postseason_cols)
    notes = []

    if not team_game.empty and {"season", "team", "points_for", "points_against"}.issubset(team_game.columns):
        game_df = standardize_team_season(team_game)
        if "season_type" in game_df.columns:
            game_df = game_df[game_df["season_type"].astype(str).str.upper().eq("REG")]

        game_df = clean_numeric_columns(game_df, ["points_for", "points_against"])
        game_df = game_df.dropna(subset=["season", "team", "points_for", "points_against"])

        if not game_df.empty:
            game_df["win"] = (game_df["points_for"] > game_df["points_against"]).astype(int)
            game_df["loss"] = (game_df["points_for"] < game_df["points_against"]).astype(int)
            game_df["tie"] = (game_df["points_for"] == game_df["points_against"]).astype(int)

            records = (
                game_df.groupby(["season", "team"], as_index=False)
                .agg(
                    current_wins=("win", "sum"),
                    current_losses=("loss", "sum"),
                    current_ties=("tie", "sum"),
                    scored_games=("game_id", "nunique") if "game_id" in game_df.columns else ("win", "size"),
                )
            )
    else:
        notes.append("Regular-season wins could not be calculated because team game scores are missing.")

    if not games.empty and {"season", "game_type", "home_team", "away_team", "home_score", "away_score"}.issubset(games.columns):
        playoff_df = games.copy()
        playoff_df["season"] = pd.to_numeric(playoff_df["season"], errors="coerce")
        playoff_df = playoff_df.dropna(subset=["season"])
        playoff_df["season"] = playoff_df["season"].astype(int)
        playoff_df["game_type"] = playoff_df["game_type"].astype(str).str.upper()
        playoff_df = playoff_df[~playoff_df["game_type"].eq("REG")]
        playoff_df = clean_numeric_columns(playoff_df, ["home_score", "away_score"])
        playoff_df = playoff_df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])

        if not playoff_df.empty:
            winners = []
            appearances = []
            for _, row in playoff_df.iterrows():
                home = str(row["home_team"]).strip()
                away = str(row["away_team"]).strip()
                appearances.append({"season": row["season"], "team": home, "postseason_games": 1})
                appearances.append({"season": row["season"], "team": away, "postseason_games": 1})

                if row["home_score"] > row["away_score"]:
                    winners.append({"season": row["season"], "team": home, "postseason_wins": 1})
                elif row["away_score"] > row["home_score"]:
                    winners.append({"season": row["season"], "team": away, "postseason_wins": 1})

            appearance_df = pd.DataFrame(appearances)
            winner_df = pd.DataFrame(winners)

            postseason_records = (
                appearance_df.groupby(["season", "team"], as_index=False)["postseason_games"]
                .sum()
                .merge(
                    winner_df.groupby(["season", "team"], as_index=False)["postseason_wins"].sum(),
                    on=["season", "team"],
                    how="left",
                )
            )
            postseason_records["postseason_wins"] = postseason_records["postseason_wins"].fillna(0)
        else:
            notes.append("Postseason prediction is limited because no playoff rows were found in the games file.")
    else:
        notes.append("Postseason prediction is limited because the games file does not include complete playoff fields.")

    return records, postseason_records, notes


def build_model_dataset(team_season, team_game, games):
    notes = []
    missing_columns = []

    if team_season.empty:
        return pd.DataFrame(), [], ["team_season_estat.csv is required for the predictive model."]

    required = {"season", "team"}
    if not required.issubset(team_season.columns):
        missing = sorted(required - set(team_season.columns))
        return pd.DataFrame(), missing, [f"Missing required season columns: {', '.join(missing)}"]

    model_df = standardize_team_season(team_season)

    numeric_candidates = [
        "games",
        "points_for",
        "points_against",
        "point_diff",
        "off_epa_per_play",
        "off_adj_epa",
        "off_success_rate",
        "def_epa_allowed_per_play",
        "def_adj_epa",
        "def_success_allowed",
        "penalty_yards_margin",
        "turnover_margin",
        "point_diff_per_game",
        "turnover_margin_per_game",
        "penalty_yards_margin_per_game",
        "success_margin",
        "yards_per_play",
        "offense_estat",
        "defense_estat",
        "overall_estat",
        "custom_estat",
        "schedule_strength",
        "overall_rank",
        "offense_rank",
        "defense_rank",
        "sos_rank",
    ]
    model_df = clean_numeric_columns(model_df, numeric_candidates)

    records, postseason_records, record_notes = calculate_team_records(team_game, games)
    notes.extend(record_notes)

    if not records.empty:
        model_df = model_df.merge(records, on=["season", "team"], how="left")
    else:
        missing_columns.append("current_wins")

    if not postseason_records.empty:
        model_df = model_df.merge(postseason_records, on=["season", "team"], how="left")
        model_df["postseason_wins"] = model_df["postseason_wins"].fillna(0)
        model_df["postseason_games"] = model_df["postseason_games"].fillna(0)
    else:
        model_df["postseason_wins"] = pd.NA
        model_df["postseason_games"] = pd.NA

    if "custom_estat" in model_df.columns:
        model_df["gini_score"] = model_df["custom_estat"]
    elif "overall_estat" in model_df.columns:
        model_df["gini_score"] = model_df["overall_estat"]
    elif "estat_raw" in model_df.columns:
        model_df["gini_score"] = pd.to_numeric(model_df["estat_raw"], errors="coerce")
    else:
        missing_columns.append("gini_score")

    if "gini_score" in model_df.columns:
        model_df["gini_rank"] = model_df.groupby("season")["gini_score"].rank(ascending=False, method="min")
    elif "overall_rank" in model_df.columns:
        model_df["gini_rank"] = model_df["overall_rank"]

    if "point_diff_per_game" in model_df.columns:
        model_df["point_diff_rank"] = model_df.groupby("season")["point_diff_per_game"].rank(ascending=False, method="min")

    if {"offense_rank", "defense_rank"}.issubset(model_df.columns):
        model_df["balance_gap"] = (model_df["offense_rank"] - model_df["defense_rank"]).abs()
        model_df["elite_offense"] = (model_df["offense_rank"] <= 8).astype(int)
        model_df["elite_defense"] = (model_df["defense_rank"] <= 8).astype(int)
        model_df["two_way_team"] = ((model_df["offense_rank"] <= 12) & (model_df["defense_rank"] <= 12)).astype(int)
        model_df["strong_unit"] = ((model_df["offense_rank"] <= 12) | (model_df["defense_rank"] <= 12)).astype(int)
    else:
        missing_columns.append("offense_rank/defense_rank")

    if "gini_rank" in model_df.columns:
        model_df["top_gini_team"] = (model_df["gini_rank"] <= 6).astype(int)

    if "point_diff_rank" in model_df.columns:
        model_df["top_point_diff_team"] = (model_df["point_diff_rank"] <= 6).astype(int)

    if {"top_gini_team", "top_point_diff_team", "strong_unit"}.issubset(model_df.columns):
        model_df["super_square_profile"] = (
            (model_df["top_gini_team"] == 1)
            & (model_df["top_point_diff_team"] == 1)
            & (model_df["strong_unit"] == 1)
        ).astype(int)

    if "current_wins" in model_df.columns:
        model_df["wins_rank"] = model_df.groupby("season")["current_wins"].rank(ascending=False, method="min")
        if "gini_rank" in model_df.columns:
            model_df["profile_win_gap"] = model_df["wins_rank"] - model_df["gini_rank"]
            model_df["underperformer_signal"] = (model_df["profile_win_gap"] >= 4).astype(int)
            model_df["overperformer_signal"] = (model_df["profile_win_gap"] <= -4).astype(int)

        model_df = model_df.sort_values(["team", "season"])
        model_df["next_season"] = model_df.groupby("team")["season"].shift(-1)
        model_df["next_season_wins"] = model_df.groupby("team")["current_wins"].shift(-1)
        model_df.loc[model_df["next_season"] != model_df["season"] + 1, "next_season_wins"] = pd.NA
        model_df["wins_change"] = model_df["next_season_wins"] - model_df["current_wins"]
        model_df["strong_next_season"] = (model_df["next_season_wins"] >= 10).astype("Int64")
    else:
        notes.append("Next-season targets are unavailable because current-season wins could not be calculated.")

    return model_df, sorted(set(missing_columns)), notes


def display_label(column):
    return DISPLAY_LABELS.get(column, str(column).replace("_", " ").title())


def yes_no(value):
    if pd.isna(value):
        return "-"
    return "Yes" if int(value) == 1 else "No"


def profile_text(value, positive_label, negative_label):
    if pd.isna(value):
        return "-"
    return positive_label if int(value) == 1 else negative_label


def get_feature_candidates(model_df):
    candidates = [
        "current_wins",
        "gini_score",
        "offense_estat",
        "defense_estat",
        "point_diff_per_game",
        "off_adj_epa",
        "def_adj_epa",
        "success_margin",
        "turnover_margin_per_game",
        "penalty_yards_margin_per_game",
        "schedule_strength",
        "sos_rank",
        "offense_rank",
        "defense_rank",
        "balance_gap",
        "elite_offense",
        "elite_defense",
        "two_way_team",
        "top_gini_team",
        "top_point_diff_team",
        "super_square_profile",
        "underperformer_signal",
        "overperformer_signal",
    ]
    return [column for column in candidates if column in model_df.columns]


def limit_to_last_target_seasons(df, max_seasons=20):
    if df.empty or "season" not in df.columns:
        return df
    seasons = sorted(df["season"].dropna().astype(int).unique())
    if len(seasons) <= max_seasons:
        return df
    keep = set(seasons[-max_seasons:])
    return df[df["season"].isin(keep)].copy()


@st.cache_data(show_spinner=False)
def train_models(model_df):
    if not SKLEARN_AVAILABLE:
        return {
            "available": False,
            "message": "scikit-learn is not available in this environment.",
        }

    if model_df.empty or "next_season_wins" not in model_df.columns:
        return {
            "available": False,
            "message": "Following-season wins are not available for modeling.",
        }

    features = get_feature_candidates(model_df)
    if not features:
        return {
            "available": False,
            "message": "No usable team traits were found for modeling.",
        }

    training_df = model_df.dropna(subset=["next_season_wins"]).copy()
    training_df = limit_to_last_target_seasons(training_df, 20)
    training_df = clean_numeric_columns(training_df, features + ["next_season_wins", "strong_next_season"])
    training_df = training_df.dropna(subset=["next_season_wins"])

    if len(training_df) < 80 or training_df["season"].nunique() < 6:
        return {
            "available": False,
            "message": "The modeling dataset is too small for a useful time-aware train/test split.",
            "features": features,
            "rows": len(training_df),
        }

    seasons = sorted(training_df["season"].dropna().astype(int).unique())
    test_count = max(3, min(5, len(seasons) // 4))
    test_seasons = seasons[-test_count:]
    train_seasons = seasons[:-test_count]

    train_df = training_df[training_df["season"].isin(train_seasons)].copy()
    test_df = training_df[training_df["season"].isin(test_seasons)].copy()

    if train_df.empty or test_df.empty:
        return {
            "available": False,
            "message": "The time-aware train/test split did not produce enough rows.",
            "features": features,
        }

    X_train = train_df[features]
    y_train = train_df["next_season_wins"]
    X_test = test_df[features]
    y_test = test_df["next_season_wins"]

    model_specs = {
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=450,
                        min_samples_leaf=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=180,
                        learning_rate=0.035,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Ridge Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=4.0)),
            ]
        ),
        "Elastic Net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.08, l1_ratio=0.25, random_state=42, max_iter=20000)),
            ]
        ),
    }

    model_results = []
    fitted_models = {}
    for name, model in model_specs.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_results.append(
            {
                "Model": name,
                "Test MAE": float(mean_absolute_error(y_test, predictions)),
                "Test R2": float(r2_score(y_test, predictions)) if len(test_df) > 1 else None,
            }
        )
        fitted_models[name] = (model, predictions)

    model_results_df = pd.DataFrame(model_results).sort_values("Test MAE", ascending=True)
    best_model_name = str(model_results_df.iloc[0]["Model"])
    reg_model, best_predictions = fitted_models[best_model_name]
    test_df["predicted_next_wins"] = best_predictions

    baseline_source = (
        test_df["current_wins"]
        if "current_wins" in test_df.columns
        else pd.Series(y_train.mean(), index=test_df.index)
    )
    baseline_mae = float(mean_absolute_error(y_test, baseline_source))
    metrics = {
        "model_mae": float(mean_absolute_error(y_test, test_df["predicted_next_wins"])),
        "baseline_mae": baseline_mae,
        "mae_edge": baseline_mae - float(mean_absolute_error(y_test, test_df["predicted_next_wins"])),
        "r2": float(r2_score(y_test, test_df["predicted_next_wins"])) if len(test_df) > 1 else None,
        "best_model_name": best_model_name,
    }

    best_estimator = reg_model.named_steps["model"]
    if hasattr(best_estimator, "feature_importances_"):
        importance_values = best_estimator.feature_importances_
    elif hasattr(best_estimator, "coef_"):
        importance_values = abs(best_estimator.coef_)
    else:
        importance_values = [0] * len(features)

    importances = pd.DataFrame(
        {
            "Feature": features,
            "Team Trait": [display_label(feature) for feature in features],
            "Importance": importance_values,
        }
    ).sort_values("Importance", ascending=False)

    classifier = None
    class_metrics = {}
    class_test_df = test_df.copy()

    if "strong_next_season" in training_df.columns:
        class_train = train_df.dropna(subset=["strong_next_season"]).copy()
        class_test = test_df.dropna(subset=["strong_next_season"]).copy()
        if class_train["strong_next_season"].nunique() == 2 and class_test["strong_next_season"].nunique() >= 1:
            classifier = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=350,
                            min_samples_leaf=4,
                            random_state=42,
                        ),
                    ),
                ]
            )
            classifier.fit(class_train[features], class_train["strong_next_season"].astype(int))
            class_predictions = classifier.predict(class_test[features])
            class_probabilities = classifier.predict_proba(class_test[features])[:, 1]

            class_test_df = class_test.copy()
            class_test_df["predicted_10_win_team"] = class_predictions
            class_test_df["prob_10_plus_wins"] = class_probabilities
            class_metrics = {
                "accuracy": float(accuracy_score(class_test["strong_next_season"].astype(int), class_predictions)),
                "confusion_matrix": confusion_matrix(
                    class_test["strong_next_season"].astype(int),
                    class_predictions,
                    labels=[0, 1],
                ).tolist(),
            }

    return {
        "available": True,
        "reg_model": reg_model,
        "classifier": classifier,
        "features": features,
        "metrics": metrics,
        "class_metrics": class_metrics,
        "importances": importances,
        "model_results": model_results_df,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "test_df": test_df,
        "class_test_df": class_test_df,
        "training_rows": len(training_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }


def build_team_name_lookup(team_assets):
    if team_assets.empty or not {"team_abbr", "team_name"}.issubset(team_assets.columns):
        return {}

    lookup = {}
    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        lookup[str(row["team_abbr"])] = str(row.get("team_name", row["team_abbr"]))
    return lookup


def build_team_theme_lookup(team_assets):
    if team_assets.empty or "team_abbr" not in team_assets.columns:
        return {}

    lookup = {}
    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        abbr = str(row["team_abbr"]).strip()
        primary = row.get("team_color", PRIMARY)
        secondary = row.get("team_color2", SECONDARY)
        if not isinstance(primary, str) or not primary.startswith("#"):
            primary = PRIMARY
        if not isinstance(secondary, str) or not secondary.startswith("#"):
            secondary = SECONDARY
        lookup[abbr] = {"primary": primary, "secondary": secondary}
    return lookup


def get_team_theme(team, team_theme_lookup):
    return team_theme_lookup.get(str(team), {"primary": PRIMARY, "secondary": SECONDARY})


def team_display_name(team, team_name_lookup, include_abbr=True):
    team = str(team)
    full_name = team_name_lookup.get(team, team)
    if include_abbr and full_name != team:
        return f"{full_name} ({team})"
    return full_name


def render_metric_card(label, value, helper="", accent=PRIMARY):
    st.markdown(
        f"""
<div class="predict-card metric-card" style="border-left-color:{accent};">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-helper">{helper}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_takeaway(text, title="Key Takeaway", accent=PRIMARY):
    st.markdown(
        f"""
<div class="predict-card takeaway-card" style="border-left-color:{accent};">
    <div class="takeaway-title">{title}</div>
    <div class="takeaway-text">{text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_caption(text):
    st.markdown(
        f"""
<div class="chart-caption">
    <span>What this shows:</span> {text}
</div>
""",
        unsafe_allow_html=True,
    )


def format_display_table(df, fmt=None):
    display_df = df.copy()

    for column in display_df.columns:
        if column in {"Super Square Profile", "Overperformer", "Underperformer", "Elite Offense", "Elite Defense", "Top Gini Team", "Top Point Differential Team"}:
            display_df[column] = display_df[column].apply(yes_no)
        elif column == "Two-Way Team":
            display_df[column] = display_df[column].apply(lambda value: profile_text(value, "Two-Way Team", "Not Two-Way"))

    if fmt:
        for column, formatter in fmt.items():
            if column in display_df.columns:
                display_df[column] = display_df[column].apply(lambda value: formatter.format(value) if pd.notna(value) else "-")
    else:
        for column in display_df.columns:
            if column in INTEGER_DISPLAY_COLUMNS:
                display_df[column] = display_df[column].apply(lambda value: f"{float(value):.0f}" if pd.notna(value) and str(value) != "-" else value)
            elif column in ONE_DECIMAL_DISPLAY_COLUMNS:
                display_df[column] = display_df[column].apply(lambda value: f"{float(value):.1f}" if pd.notna(value) and str(value) != "-" else value)
            elif column in THREE_DECIMAL_DISPLAY_COLUMNS:
                display_df[column] = display_df[column].apply(lambda value: f"{float(value):.3f}" if pd.notna(value) and str(value) != "-" else value)

    return display_df


def rename_for_display(df, team_name_lookup=None):
    display_df = df.copy()
    if team_name_lookup and "team" in display_df.columns:
        display_df["team"] = display_df["team"].apply(lambda team: team_display_name(team, team_name_lookup))
    return display_df.rename(columns={column: display_label(column) for column in display_df.columns})


def show_clean_table(df, fmt=None, max_height=460):
    display_df = format_display_table(df, fmt=fmt)

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=max_height,
    )


def chart_layout(fig, height=480, x_title=None, y_title=None, legend_title=None):
    fig.update_layout(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=TEXT, size=12),
        margin=dict(l=70, r=40, t=72, b=78),
        title=dict(font=dict(size=15, color=TEXT), x=0.01, xanchor="left"),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=TEXT, bordercolor=PRIMARY),
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if legend_title:
        fig.update_layout(legend_title_text=legend_title)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.20,
                xanchor="center",
                x=0.5,
            )
        )
    return fig


def percent_text(value):
    if pd.isna(value):
        return "-"
    return f"{100 * float(value):.1f}%"


def number_text(value, decimals=1):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def win_text(value, decimals=0):
    if value is None or pd.isna(value):
        return "Not available yet"
    return f"{float(value):.{decimals}f}"


def get_prediction_for_row(row, model_bundle):
    if not model_bundle.get("available"):
        return None, None

    features = model_bundle["features"]
    X = pd.DataFrame([{feature: row.get(feature, pd.NA) for feature in features}])
    predicted_wins = float(model_bundle["reg_model"].predict(X)[0])

    probability = None
    classifier = model_bundle.get("classifier")
    if classifier is not None:
        probability = float(classifier.predict_proba(X)[0, 1])

    return predicted_wins, probability


def risk_label(row, predicted_wins, probability):
    reasons = []
    selected_wins = row.get("current_wins")
    gini_rank = row.get("gini_rank")
    pd_rank = row.get("point_diff_rank")
    balance_gap = row.get("balance_gap")
    point_diff = row.get("point_diff_per_game")

    if row.get("super_square_profile", 0) == 1:
        reasons.append("it matched the Super Square profile")
    if row.get("two_way_team", 0) == 1:
        reasons.append("it had a balanced offense-defense profile")
    if pd.notna(point_diff) and point_diff > 5:
        reasons.append("it controlled scoring margin")
    if row.get("underperformer_signal", 0) == 1:
        reasons.append("its Gini profile was stronger than its selected-season win total")
    if row.get("overperformer_signal", 0) == 1:
        reasons.append("its selected-season win total looked stronger than its underlying profile")

    high_profile = pd.notna(gini_rank) and gini_rank <= 8 and pd.notna(pd_rank) and pd_rank <= 8
    balanced = row.get("two_way_team", 0) == 1 or (pd.notna(balance_gap) and balance_gap <= 8)
    weaker_underlying = (
        pd.notna(gini_rank)
        and gini_rank > 14
        and (pd.isna(point_diff) or point_diff < 2)
    )
    one_sided = (
        pd.notna(balance_gap)
        and balance_gap >= 14
        and (
            row.get("elite_offense", 0) == 1
            or row.get("elite_defense", 0) == 1
        )
    )

    if high_profile and balanced:
        label = "Strong carry-forward profile"
    elif pd.notna(selected_wins) and selected_wins >= 10 and weaker_underlying:
        label = "Regression risk"
    elif row.get("underperformer_signal", 0) == 1 and high_profile:
        label = "Improvement candidate"
    elif one_sided:
        label = "Volatile profile"
    else:
        label = "Stable profile"

    if not reasons:
        if predicted_wins is not None and pd.notna(selected_wins):
            if predicted_wins > selected_wins + 1:
                reasons.append("similar historical teams tended to improve the following season")
            elif predicted_wins < selected_wins - 1:
                reasons.append("similar historical teams tended to give back some wins the following season")
            else:
                reasons.append("similar historical teams tended to land close to the same win range")
        else:
            reasons.append("its profile landed near the historical middle of the model inputs")

    explanation = (
        f"This team projects as a {label.lower()} because "
        + ", ".join(reasons[:3])
        + ". This projection is based on how similar team profiles performed in past seasons. "
        + "It is not a guarantee, and it does not fully know future injuries, quarterback changes, coaching changes, or roster movement."
    )

    return label, explanation


def profile_interpretation(row, predicted_wins, season_df):
    selected_wins = row.get("current_wins")
    gini_score = row.get("gini_score")
    point_diff = row.get("point_diff_per_game")
    offense_rank = row.get("offense_rank")
    defense_rank = row.get("defense_rank")
    balance_gap = row.get("balance_gap")

    league_avg_gini = season_df["gini_score"].mean() if "gini_score" in season_df.columns else pd.NA
    reasons = []

    if pd.notna(gini_score) and pd.notna(league_avg_gini):
        gap = gini_score - league_avg_gini
        if gap >= 5:
            reasons.append(f"its Gini Score was {gap:.1f} points above the selected-season league average")
        elif gap <= -5:
            reasons.append(f"its Gini Score was {abs(gap):.1f} points below the selected-season league average")
        else:
            reasons.append("its Gini Score was near the selected-season league average")

    if pd.notna(point_diff):
        if point_diff >= 5:
            reasons.append(f"it had a strong scoring margin at {point_diff:.1f} points per game")
        elif point_diff <= -3:
            reasons.append(f"its point differential was negative at {point_diff:.1f} points per game")
        else:
            reasons.append("its scoring margin was close to even")

    if pd.notna(offense_rank) and pd.notna(defense_rank):
        if offense_rank <= 12 and defense_rank <= 12:
            reasons.append("both offense and defense ranked in the top 12")
        elif offense_rank <= 10 and defense_rank > 20:
            reasons.append("the offense was strong, but the defense lagged behind")
        elif defense_rank <= 10 and offense_rank > 20:
            reasons.append("the defense was strong, but the offense lagged behind")
        elif offense_rank > 20 and defense_rank > 20:
            reasons.append("both offense and defense ranked outside the top 20")
        elif pd.notna(balance_gap) and balance_gap <= 6:
            reasons.append("the offense and defense profile was fairly balanced")

    if row.get("underperformer_signal", 0) == 1:
        reasons.append("the team looked better by profile than by record")
    elif row.get("overperformer_signal", 0) == 1:
        reasons.append("the team looked stronger by record than by profile")
    elif pd.notna(selected_wins):
        if selected_wins >= 11:
            reasons.append("the selected-season record was already strong")
        elif selected_wins <= 5:
            reasons.append("the selected-season record was weak")

    if not reasons:
        reasons.append("its profile was close to the historical middle of the dataset")

    if predicted_wins is not None and pd.notna(selected_wins):
        if predicted_wins >= selected_wins + 2:
            reasons.append("similar historical profiles tended to improve the following season")
        elif predicted_wins <= selected_wins - 2:
            reasons.append("similar historical profiles tended to give back wins the following season")
        else:
            reasons.append("similar historical profiles usually stayed near the same win range")

    return reasons[:4]


# -----------------------------
# Page styling
# -----------------------------


st.markdown(
    f"""
<style>
.stApp,
[data-testid="stAppViewContainer"] {{
    background: {PAGE_BG} !important;
    color: {TEXT};
    overflow-x: hidden;
}}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}

.block-container {{
    position: relative;
    z-index: 2;
    padding-top: 0.65rem !important;
    padding-bottom: 2.6rem !important;
}}

.predict-bg {{
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    pointer-events: none;
    z-index: 0;
}}

.predict-bg::before {{
    content: "";
    position: absolute;
    top: -18%;
    left: -10%;
    width: 145%;
    height: 145%;
    background: radial-gradient(ellipse at center, rgba(241, 90, 36, 0.22) 0%, rgba(241, 90, 36, 0.07) 30%, transparent 62%);
    animation: predictGlowPrimary 24s ease-in-out infinite;
}}

.predict-bg::after {{
    content: "";
    position: absolute;
    bottom: -18%;
    right: -10%;
    width: 145%;
    height: 145%;
    background: radial-gradient(ellipse at center, rgba(0, 115, 183, 0.20) 0%, rgba(0, 115, 183, 0.07) 30%, transparent 62%);
    animation: predictGlowSecondary 26s ease-in-out infinite;
}}

@keyframes predictGlowPrimary {{
    0% {{ opacity: 0.45; transform: translate(-8%, 0%) scale(1); }}
    50% {{ opacity: 0.8; transform: translate(78%, 28%) scale(1.24); }}
    100% {{ opacity: 0.45; transform: translate(-8%, 0%) scale(1); }}
}}

@keyframes predictGlowSecondary {{
    0% {{ opacity: 0.75; transform: translate(8%, 0%) scale(1); }}
    50% {{ opacity: 0.62; transform: translate(-58%, -18%) scale(1.22); }}
    100% {{ opacity: 0.75; transform: translate(8%, 0%) scale(1); }}
}}

.predict-page {{
    position: relative;
    z-index: 2;
    max-width: 1500px;
    margin: 0 auto;
    padding: 0.25rem 1.1rem 2.5rem 1.1rem;
}}

.predict-hero {{
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(360px, 0.55fr);
    gap: 1.4rem;
    align-items: center;
    margin: 0.85rem 0 1.25rem 0;
    padding: 1.8rem 1.9rem;
    border-radius: 20px;
    overflow: hidden;
    background:
        radial-gradient(circle at 92% 12%, {PRIMARY}44 0%, transparent 28%),
        radial-gradient(circle at 78% 85%, {SECONDARY}38 0%, transparent 30%),
        linear-gradient(135deg, #07111f 0%, #172033 58%, #273447 100%);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 22px 46px rgba(15, 23, 42, 0.18);
}}

.predict-hero::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
}}

.predict-kicker {{
    color: rgba(255,255,255,0.72);
    font-size: 0.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.55rem;
}}

.predict-title {{
    color: #FFFFFF;
    font-size: clamp(2.3rem, 4vw, 3.45rem);
    line-height: 1;
    font-weight: 950;
    letter-spacing: 0;
    margin-bottom: 0.75rem;
}}

.predict-subtitle {{
    max-width: 860px;
    color: rgba(255,255,255,0.86);
    font-size: 1rem;
    line-height: 1.68;
}}

.research-card {{
    justify-self: end;
    width: min(100%, 520px);
    padding: 1.15rem 1.2rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.24);
    border-left: 5px solid {PRIMARY};
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.16), 0 14px 30px rgba(0,0,0,0.12);
    backdrop-filter: blur(10px);
}}

.research-label {{
    color: rgba(255,255,255,0.68);
    font-size: 0.72rem;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.research-stat {{
    color: #FFFFFF;
    font-size: 2rem;
    line-height: 1.1;
    font-weight: 950;
    margin-bottom: 0.45rem;
}}

.research-text {{
    color: rgba(255,255,255,0.83);
    font-size: 0.95rem;
    line-height: 1.55;
}}

.predict-card {{
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {PRIMARY};
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
    backdrop-filter: blur(9px);
    overflow: hidden;
}}

.metric-card {{
    min-height: 118px;
    padding: 0.95rem 1.05rem;
}}

.metric-label {{
    color: {MUTED};
    font-size: 0.74rem;
    line-height: 1.2;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.metric-value {{
    color: {TEXT};
    font-size: 1.72rem;
    line-height: 1.15;
    font-weight: 950;
    margin-bottom: 0.45rem;
}}

.metric-helper {{
    color: {MUTED};
    font-size: 0.84rem;
    line-height: 1.4;
    font-weight: 750;
}}

.explain-box {{
    padding: 1rem 1.1rem;
    margin: 0.85rem 0 1rem 0;
    color: #334155;
    font-size: 0.94rem;
    line-height: 1.65;
}}

.explain-title {{
    color: {TEXT};
    font-size: 1rem;
    font-weight: 950;
    margin-bottom: 0.4rem;
}}

.takeaway-card {{
    padding: 0.95rem 1.05rem;
    margin: 0.65rem 0 1rem 0;
}}

.takeaway-title {{
    color: {TEXT};
    font-size: 0.82rem;
    line-height: 1.2;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}}

.takeaway-text {{
    color: #334155;
    font-size: 0.94rem;
    line-height: 1.58;
    font-weight: 700;
}}

.chart-caption {{
    margin: 0.4rem 0 1.05rem 0;
    padding: 0.75rem 0.9rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(15, 23, 42, 0.09);
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.55;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
}}

.chart-caption span {{
    color: {TEXT};
    font-weight: 950;
}}

.feature-group-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 0.75rem 0 1rem 0;
}}

.feature-group-card {{
    padding: 1rem 1.05rem;
}}

.feature-group-title {{
    color: {TEXT};
    font-size: 0.96rem;
    line-height: 1.2;
    font-weight: 950;
    margin-bottom: 0.65rem;
}}

.feature-list {{
    margin: 0;
    padding-left: 1.05rem;
    color: #334155;
    font-size: 0.9rem;
    line-height: 1.65;
}}

.team-accent-badge {{
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    color: #ffffff;
    font-size: 0.78rem;
    line-height: 1;
    font-weight: 950;
    margin-bottom: 0.7rem;
}}

.section-title {{
    color: {TEXT};
    font-size: 1.35rem;
    line-height: 1.2;
    font-weight: 950;
    letter-spacing: -0.02em;
    margin: 0.45rem 0 0.65rem 0;
}}

.section-title::after {{
    content: "";
    display: block;
    width: 58px;
    height: 4px;
    border-radius: 999px;
    margin-top: 0.45rem;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
}}

.prediction-summary {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(280px, 0.45fr);
    gap: 1rem;
    margin-top: 0.75rem;
}}

.profile-panel {{
    padding: 1.05rem 1.15rem;
}}

.profile-heading {{
    color: {TEXT};
    font-size: 1.2rem;
    font-weight: 950;
    margin-bottom: 0.35rem;
}}

.profile-body {{
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.65;
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF !important;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 18px;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
}}

div[data-testid="stSelectbox"] label p {{
    color: {TEXT} !important;
    font-size: 0.94rem !important;
    font-weight: 900 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.45rem;
    background: rgba(255,255,255,0.76);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 14px;
    padding: 0.35rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.72rem 0.85rem;
    font-size: 0.92rem;
    font-weight: 850;
    color: #334155;
}}

.stTabs [aria-selected="true"] {{
    background: #FFFFFF;
    color: {PRIMARY} !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.075);
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY} !important;
    height: 3px;
}}

@media (max-width: 900px) {{
    .predict-hero,
    .prediction-summary {{
        grid-template-columns: 1fr;
    }}

    .feature-group-grid {{
        grid-template-columns: 1fr;
    }}

    .research-card {{
        justify-self: start;
    }}
}}
</style>
<div class="predict-bg" aria-hidden="true"></div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Load and prepare data
# -----------------------------


data, load_messages = load_data()
team_season = data["team_season"]
team_game = data["team_game"]
games = data["games"]
team_assets = data["team_assets"]

model_df, skipped_columns, build_notes = build_model_dataset(team_season, team_game, games)
model_bundle = train_models(model_df)
team_name_lookup = build_team_name_lookup(team_assets)
team_theme_lookup = build_team_theme_lookup(team_assets)

seasons_available = sorted(model_df["season"].dropna().astype(int).unique()) if not model_df.empty and "season" in model_df.columns else []
profile_seasons = seasons_available[-20:] if len(seasons_available) > 20 else seasons_available
analysis_df = (
    limit_to_last_target_seasons(model_df.dropna(subset=["next_season_wins"]).copy(), 20)
    if "next_season_wins" in model_df.columns
    else pd.DataFrame()
)
analysis_seasons = sorted(analysis_df["season"].dropna().astype(int).unique()) if not analysis_df.empty else []
completed_target_rows = int(len(analysis_df))
postseason_rows_available = bool(
    "postseason_games" in model_df.columns
    and pd.to_numeric(model_df["postseason_games"], errors="coerce").fillna(0).sum() > 0
)


# -----------------------------
# Header
# -----------------------------


render_top_nav("Predictive Model", PRIMARY, SECONDARY)

st.markdown('<div class="predict-page">', unsafe_allow_html=True)

st.markdown(
    """
<div class="predict-hero">
    <div>
        <div class="predict-kicker">Historical Team Forecasting</div>
        <div class="predict-title">Predictive Model</div>
        <div class="predict-subtitle">
            This page uses historical Gini Metric data to study which team profiles carry forward.
            It looks at past seasons to identify patterns of sustained success, regression risk,
            and future win potential.
        </div>
    </div>
    <div class="research-card">
        <div class="research-label">Research Goal</div>
        <div class="research-stat">20 Seasons</div>
        <div class="research-text">Use past team profiles to understand what tends to happen next.</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if load_messages:
    for message in load_messages:
        st.warning(message)

if model_df.empty:
    st.error("The predictive model page could not build a team-season dataset from the available files.")
    st.stop()


# -----------------------------
# Tabs
# -----------------------------


tab_overview, tab_patterns, tab_regression, tab_tool, tab_details = st.tabs(
    [
        "Model Overview",
        "Patterns of Success",
        "Regression / Failure Signals",
        "Prediction Tool",
        "Model Details",
    ]
)


with tab_overview:
    st.markdown('<div class="section-title">What the Model Studies</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="predict-card explain-box">
    <div class="explain-title">How to read this page</div>
    This page studies whether a team's regular-season profile can help explain what happens one year later.
    Instead of only looking at record, the model compares team quality, scoring margin, offense, defense,
    consistency, turnovers, penalties, and schedule context. The goal is to estimate following-season wins
    and identify which team profiles tend to sustain success, regress, or improve.
    <br><br>
    This model is most useful for understanding team profiles, not making exact win guarantees. The NFL changes
    quickly through injuries, quarterback changes, coaching changes, roster movement, and schedule changes, so this
    should be treated as research and context, not betting advice.
</div>
""",
        unsafe_allow_html=True,
    )
    render_takeaway(
        "This model compares selected-season team profiles to what happened the following regular season. Lower model error means the projections were closer to actual results."
    )

    metric_cols = st.columns(4)
    mae_edge = model_bundle.get("metrics", {}).get("mae_edge") if model_bundle.get("available") else None
    with metric_cols[0]:
        render_metric_card(
            "Seasons Analyzed",
            len(profile_seasons),
            f"{min(profile_seasons)}-{max(profile_seasons)} completed seasons used in the study"
            if profile_seasons
            else "Team profiles unavailable",
        )
    with metric_cols[1]:
        render_metric_card("Model Rows Tested", completed_target_rows, "Team-seasons with known following-season wins")
    with metric_cols[2]:
        render_metric_card("Model Target", "Following-Season Wins", "Following-season regular-season wins")
    with metric_cols[3]:
        if model_bundle.get("available"):
            if mae_edge >= 0:
                helper = f"Beat the simple baseline by {mae_edge:.2f} wins"
            else:
                helper = f"{abs(mae_edge):.2f} wins worse than the simple baseline"
            render_metric_card("Model MAE", f"{model_bundle['metrics']['model_mae']:.2f}", helper)
        else:
            render_metric_card("Model Status", "Limited", model_bundle.get("message", "Model unavailable"))

    if profile_seasons:
        render_caption(
            f"The study shows the last {len(profile_seasons)} available profile seasons ({min(profile_seasons)}-{max(profile_seasons)}). "
            "Model testing only uses team-seasons where the following season is already known."
        )

    if model_bundle.get("available"):
        if model_bundle["metrics"]["mae_edge"] < 0:
            st.warning(
                "On the held-out seasons, the best model is worse than the simple baseline that uses selected-season wins. "
                "That is useful to know: following-season NFL wins are noisy, and the page should be read as research context."
            )
        else:
            st.success(
                f"The best model beat the simple baseline by {model_bundle['metrics']['mae_edge']:.2f} wins on average in the held-out seasons."
            )

        st.markdown(
            f"""
<div class="predict-card explain-box">
    <div class="explain-title">How to read the error</div>
    An MAE of {model_bundle['metrics']['model_mae']:.1f} means a projection of 9 wins should be read more like a rough range,
    not an exact prediction.
</div>
""",
            unsafe_allow_html=True,
        )

        overview_cols = st.columns(2)
        with overview_cols[0]:
            st.markdown('<div class="section-title">Model Error vs Simple Baseline</div>', unsafe_allow_html=True)
            compare_df = pd.DataFrame(
                {
                    "Forecast Method": [
                        f"Best model: {model_bundle['metrics']['best_model_name']}",
                        "Simple baseline: selected-season wins",
                    ],
                    "Average Miss in Wins": [
                        model_bundle["metrics"]["model_mae"],
                        model_bundle["metrics"]["baseline_mae"],
                    ],
                }
            )
            fig = px.bar(
                compare_df,
                x="Forecast Method",
                y="Average Miss in Wins",
                color="Forecast Method",
                color_discrete_sequence=[PRIMARY, SECONDARY],
                title="Model Error vs Simple Baseline",
                labels={"Average Miss in Wins": "Average Miss in Wins", "Forecast Method": "Forecast Method"},
            )
            fig.update_traces(texttemplate="%{y:.2f}", textposition="outside", showlegend=False)
            st.plotly_chart(chart_layout(fig, height=390, y_title="Average Miss in Wins"), use_container_width=True)
            render_caption(
                "Lower is better. This chart compares the model's average miss in wins against a simple baseline that assumes a team will win about the same number of games the following season."
            )

        with overview_cols[1]:
            st.markdown('<div class="section-title">Most Important Team Traits</div>', unsafe_allow_html=True)
            importance_df = model_bundle["importances"].head(10).sort_values("Importance", ascending=True)
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Team Trait",
                orientation="h",
                color_discrete_sequence=[PRIMARY],
                title="Most Important Team Traits",
                labels={"Importance": "Model Importance", "Team Trait": "Team Trait"},
            )
            st.plotly_chart(chart_layout(fig, height=390, x_title="Model Importance"), use_container_width=True)
            render_caption(
                "These are the team traits the model relied on most when estimating following-season wins. Higher importance means the trait helped more in the model's historical predictions."
            )

        st.markdown('<div class="section-title">Models Compared</div>', unsafe_allow_html=True)
        model_results = model_bundle["model_results"].rename(
            columns={"Model": "Model", "Test MAE": "Average Miss in Wins", "Test R2": "R-Squared"}
        )
        show_clean_table(
            model_results,
            fmt={"Average Miss in Wins": "{:.2f}", "R-Squared": "{:.2f}"},
            max_height=220,
        )
        render_caption(
            "This table compares the tested forecasting methods on held-out seasons. The selected model is the one with the lowest average miss in wins."
        )
    else:
        st.info(model_bundle.get("message", "The model could not be trained from the available files."))


with tab_patterns:
    st.markdown('<div class="section-title">Patterns of Success</div>', unsafe_allow_html=True)
    render_takeaway(
        "Stronger Gini profiles and balanced teams tend to carry forward better than weaker or one-dimensional profiles."
    )

    pattern_df = analysis_df.copy()
    if pattern_df.empty:
        st.info("Following-season outcomes are not available, so success-pattern charts cannot be built yet.")
    else:
        if "gini_rank" in pattern_df.columns:
            pattern_df["Gini Rank Group"] = pd.cut(
                pattern_df["gini_rank"],
                bins=[0, 6, 12, 20, 40],
                labels=["Top 6", "7-12", "13-20", "21+"],
                include_lowest=True,
            )
            gini_bucket = (
                pattern_df.groupby("Gini Rank Group", observed=False)["next_season_wins"]
                .mean()
                .reset_index()
                .rename(columns={"next_season_wins": "Average Following-Season Wins"})
            )
            fig = px.bar(
                gini_bucket,
                x="Gini Rank Group",
                y="Average Following-Season Wins",
                color="Gini Rank Group",
                color_discrete_sequence=[PRIMARY, SECONDARY, "#94A3B8", "#CBD5E1"],
                title="Average Following-Season Wins by Gini Rank Group",
                labels={"Gini Rank Group": "Gini Rank Group", "Average Following-Season Wins": "Average Following-Season Wins"},
            )
            fig.update_traces(texttemplate="%{y:.1f}", textposition="outside", showlegend=False)
            st.plotly_chart(chart_layout(fig, y_title="Average Following-Season Wins"), use_container_width=True)
            render_caption(
                "Teams with stronger Gini rankings tended to win more games the following season, which suggests overall team quality often carries forward."
            )

        chart_cols = st.columns(2)
        with chart_cols[0]:
            if {"gini_score", "next_season_wins"}.issubset(pattern_df.columns):
                scatter_df = pattern_df.copy()
                scatter_df["Team Name"] = scatter_df["team"].apply(
                    lambda team: team_display_name(team, team_name_lookup, include_abbr=True)
                )
                scatter_df["Team Profile"] = (
                    scatter_df["two_way_team"].map({1: "Two-Way Team", 0: "Other Team"})
                    if "two_way_team" in scatter_df.columns
                    else "Team"
                )
                fig = px.scatter(
                    scatter_df,
                    x="gini_score",
                    y="next_season_wins",
                    color="Team Profile",
                    hover_name="Team Name",
                    hover_data={
                        "season": True,
                        "Team Name": False,
                        "current_wins": ":.0f" if "current_wins" in scatter_df.columns else False,
                        "gini_score": ":.1f",
                        "next_season_wins": ":.0f",
                        "Team Profile": True,
                    },
                    labels={
                        "gini_score": "Gini Score",
                        "next_season_wins": "Following Season Wins",
                        "season": "Season",
                        "current_wins": "Selected Season Wins",
                    },
                    color_discrete_map={"Two-Way Team": PRIMARY, "Other Team": "#94A3B8", "Team": PRIMARY},
                    title="Gini Score vs Following-Season Wins",
                )
                st.plotly_chart(
                    chart_layout(fig, x_title="Gini Score", y_title="Following Season Wins", legend_title="Team Profile"),
                    use_container_width=True,
                )
                render_caption(
                    "Each dot is a team-season. The chart shows whether stronger Gini Scores were usually followed by more wins one year later."
                )

        with chart_cols[1]:
            if "two_way_team" in pattern_df.columns:
                two_way_summary = (
                    pattern_df.assign(Profile=pattern_df["two_way_team"].map({1: "Two-Way Teams", 0: "Other Teams"}))
                    .groupby("Profile", as_index=False)["next_season_wins"]
                    .mean()
                    .rename(columns={"next_season_wins": "Average Following-Season Wins"})
                )
                fig = px.bar(
                    two_way_summary,
                    x="Profile",
                    y="Average Following-Season Wins",
                    color="Profile",
                    color_discrete_sequence=[PRIMARY, "#94A3B8"],
                    title="Two-Way Teams vs Other Teams",
                    labels={"Profile": "Team Profile", "Average Following-Season Wins": "Average Following-Season Wins"},
                )
                fig.update_traces(texttemplate="%{y:.1f}", textposition="outside", showlegend=False)
                st.plotly_chart(chart_layout(fig, y_title="Average Following-Season Wins"), use_container_width=True)
                render_caption(
                    "Two-way teams are teams with strong offense and defense profiles. This comparison shows whether balanced teams tended to perform better the following season."
                )

        if "super_square_profile" in pattern_df.columns:
            super_summary = (
                pattern_df.assign(Profile=pattern_df["super_square_profile"].map({1: "Super Square", 0: "Outside"}))
                .groupby("Profile", as_index=False)
                .agg(
                    Avg_Next_Wins=("next_season_wins", "mean"),
                    Strong_Next_Season_Rate=("strong_next_season", "mean"),
                    Team_Seasons=("team", "count"),
                )
                .rename(
                    columns={
                        "Avg_Next_Wins": "Average Following-Season Wins",
                        "Strong_Next_Season_Rate": "10+ Win Following Season Rate",
                        "Team_Seasons": "Team-Seasons",
                    }
                )
            )
            st.markdown('<div class="section-title">Super Square Carry-Forward</div>', unsafe_allow_html=True)
            show_clean_table(
                super_summary,
                fmt={
                    "Average Following-Season Wins": "{:.1f}",
                    "10+ Win Following Season Rate": "{:.1%}",
                },
                max_height=190,
            )
            render_caption(
                "This table compares Super Square teams with all other teams. It shows whether the strongest contender profiles were more likely to remain successful the following season."
            )

        top_cols = [
            "season",
            "team",
            "current_wins",
            "next_season_wins",
            "wins_change",
            "gini_score",
            "gini_rank",
            "point_diff_per_game",
            "offense_rank",
            "defense_rank",
            "super_square_profile",
        ]
        top_cols = [column for column in top_cols if column in pattern_df.columns]
        top_profiles = pattern_df.sort_values(["gini_score", "point_diff_per_game"], ascending=False).head(30)[top_cols]
        st.markdown('<div class="section-title">Top Historical Profiles</div>', unsafe_allow_html=True)
        show_clean_table(rename_for_display(top_profiles, team_name_lookup), max_height=520)
        render_caption(
            "These are the strongest historical profiles by Gini Score and point differential. The table shows what happened to each team in the following regular season."
        )


with tab_regression:
    st.markdown('<div class="section-title">Regression and Failure Signals</div>', unsafe_allow_html=True)
    render_takeaway(
        "Regression teams are teams that won a lot in the selected season but dropped by at least three wins the following year. This section looks for signs that their success may have been fragile."
    )
    st.markdown(
        """
<div class="predict-card explain-box">
    Regression teams are teams that won at least 10 games in the selected season and then dropped by at least
    three wins the following season.
    <br><br>
    The goal is to see whether some winning teams relied on advantages that may be harder to repeat,
    such as turnover margin or a record that was stronger than the underlying team profile.
</div>
""",
        unsafe_allow_html=True,
    )
    regression_df = analysis_df.copy()

    if regression_df.empty or "current_wins" not in regression_df.columns:
        st.info("Regression analysis needs both selected-season wins and following-season wins.")
    else:
        regression_df["Regression Team"] = (
            (regression_df["current_wins"] >= 10)
            & (regression_df["next_season_wins"] <= regression_df["current_wins"] - 3)
        )
        dropoffs = regression_df.sort_values("wins_change").head(30)

        reg_cols = st.columns(3)
        with reg_cols[0]:
            render_metric_card("Regression Teams", int(regression_df["Regression Team"].sum()), "10+ wins, then down 3+ wins")
        with reg_cols[1]:
            if "turnover_margin_per_game" in regression_df.columns:
                avg_reg_to = regression_df.loc[regression_df["Regression Team"], "turnover_margin_per_game"].mean()
                render_metric_card("Average Turnover Margin per Game", number_text(avg_reg_to, 2), "Among regression teams")
            else:
                render_metric_card("Average Turnover Margin per Game", "Missing", "Column not available")
        with reg_cols[2]:
            if "overperformer_signal" in regression_df.columns:
                rate = regression_df.loc[regression_df["Regression Team"], "overperformer_signal"].mean()
                render_metric_card("Overperformer Rate", percent_text(rate), "Among regression teams")
            else:
                render_metric_card("Overperformer Rate", "Missing", "Column not available")

        if "turnover_margin_per_game" in regression_df.columns:
            box_df = regression_df.assign(
                Group=regression_df["Regression Team"].map({True: "Regression Team", False: "Other Team"})
            )
            fig = px.box(
                box_df,
                x="Group",
                y="turnover_margin_per_game",
                color="Group",
                color_discrete_map={"Regression Team": PRIMARY, "Other Team": "#94A3B8"},
                title="Turnover Margin and Next-Year Regression",
                labels={"Group": "Team Group", "turnover_margin_per_game": "Turnover Margin per Game"},
            )
            st.plotly_chart(
                chart_layout(fig, y_title="Turnover Margin per Game", legend_title="Team Group"),
                use_container_width=True,
            )
            render_caption(
                "This chart checks whether teams that fell off the next year may have relied more on turnover margin, which can be harder to repeat."
            )

        reg_columns = [
            "season",
            "team",
            "current_wins",
            "next_season_wins",
            "wins_change",
            "gini_score",
            "point_diff_per_game",
            "offense_rank",
            "defense_rank",
            "turnover_margin_per_game",
            "schedule_strength",
            "overperformer_signal",
        ]
        reg_columns = [column for column in reg_columns if column in dropoffs.columns]
        st.markdown('<div class="section-title">Biggest Drop-Offs</div>', unsafe_allow_html=True)
        show_clean_table(rename_for_display(dropoffs[reg_columns], team_name_lookup), max_height=560)
        render_caption(
            "These are teams that won at least 10 games and then dropped by at least three wins the next year."
        )


with tab_tool:
    st.markdown('<div class="section-title">Team Prediction Tool</div>', unsafe_allow_html=True)
    render_takeaway(
        "Choose a team-season to see how similar historical profiles performed and what the model projects for the following regular season."
    )

    tool_seasons = sorted(model_df["season"].dropna().astype(int).unique(), reverse=True)
    setup_cols = st.columns([1, 1])
    selected_season = setup_cols[0].selectbox("Season", tool_seasons, index=0, key="predictive_tool_season")
    following_season = selected_season + 1
    season_slice = model_df[model_df["season"] == selected_season].copy()

    team_labels = {
        team: team_name_lookup.get(team, team)
        for team in sorted(season_slice["team"].dropna().astype(str).unique())
    }
    label_to_team = {label: team for team, label in team_labels.items()}
    selected_label = setup_cols[1].selectbox("Team", list(label_to_team.keys()), key="predictive_tool_team")
    selected_team = label_to_team[selected_label]

    selected_rows = season_slice[season_slice["team"] == selected_team]
    if selected_rows.empty:
        st.warning("No team-season row found for this selection.")
    else:
        selected_row = selected_rows.iloc[0]
        predicted_wins, probability = get_prediction_for_row(selected_row, model_bundle)
        projected_wins = max(0, min(17, predicted_wins)) if predicted_wins is not None else None
        model_mae = model_bundle.get("metrics", {}).get("model_mae") if model_bundle.get("available") else None
        range_text = "Not available yet"
        if projected_wins is not None and model_mae is not None:
            lower_bound = max(0, round(projected_wins - model_mae))
            upper_bound = min(17, round(projected_wins + model_mae))
            range_text = f"{lower_bound} to {upper_bound} wins"
        label, explanation = risk_label(selected_row, predicted_wins, probability)
        profile_reasons = profile_interpretation(selected_row, predicted_wins, season_slice)
        reasons_html = "".join(f"<li>{reason}</li>" for reason in profile_reasons)
        actual_following_wins = selected_row.get("next_season_wins")
        selected_theme = get_team_theme(selected_team, team_theme_lookup)
        team_primary = selected_theme["primary"]
        team_secondary = selected_theme["secondary"]

        tool_cols = st.columns(5)
        with tool_cols[0]:
            render_metric_card(
                f"{selected_season} Wins",
                win_text(selected_row.get("current_wins"), 0),
                "Selected regular season",
                accent=team_primary,
            )
        with tool_cols[1]:
            render_metric_card(
                f"Projected {following_season} Wins",
                win_text(projected_wins, 1) if projected_wins is not None else "Model unavailable",
                model_bundle["metrics"]["best_model_name"] if model_bundle.get("available") else "Projection unavailable",
                accent=team_secondary,
            )
        with tool_cols[2]:
            render_metric_card("Reasonable Range", range_text, "Based on model MAE", accent=team_primary)
        with tool_cols[3]:
            render_metric_card(
                f"Actual {following_season} Wins",
                win_text(actual_following_wins, 0),
                "Known only after the following season",
                accent=team_secondary,
            )
        with tool_cols[4]:
            render_metric_card(
                f"{following_season} 10+ Win Probability",
                percent_text(probability),
                "Classifier estimate" if probability is not None else "Classifier unavailable",
                accent=team_primary,
            )

        st.markdown(
            f"""
<div class="prediction-summary">
    <div class="predict-card profile-panel" style="border-left-color:{team_primary};">
        <div class="team-accent-badge" style="background:{team_primary};">{selected_season} Team Profile</div>
        <div class="profile-heading">{selected_season} {selected_label}</div>
        <div class="profile-body">
            <ul>{reasons_html}</ul>
            <p>{explanation}</p>
        </div>
    </div>
    <div class="predict-card profile-panel" style="border-left-color:{team_secondary};">
        <div class="team-accent-badge" style="background:{team_secondary};">Model Profile</div>
        <div class="profile-heading">{label}</div>
        <div class="profile-body">
            Gini Score: {number_text(selected_row.get("gini_score"), 1)}<br>
            Point Differential per Game: {number_text(selected_row.get("point_diff_per_game"), 1)}<br>
            Offense Rank: {number_text(selected_row.get("offense_rank"), 0)} / Defense Rank: {number_text(selected_row.get("defense_rank"), 0)}
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        comparison_values = [
            (f"{selected_season} Wins", selected_row.get("current_wins")),
            (f"Projected {following_season} Wins", projected_wins),
        ]
        if pd.notna(actual_following_wins):
            comparison_values.append((f"Actual {following_season} Wins", actual_following_wins))
        comparison_df = pd.DataFrame(comparison_values, columns=["Win View", "Wins"]).dropna()
        if not comparison_df.empty:
            fig = px.bar(
                comparison_df,
                x="Win View",
                y="Wins",
                color="Win View",
                color_discrete_sequence=[team_secondary, team_primary, "#94A3B8"],
                title=f"{selected_season} {selected_label}: Following-Season Projection",
                labels={"Win View": "Win View", "Wins": "Wins"},
            )
            fig.update_traces(texttemplate="%{y:.1f}", textposition="outside", showlegend=False)
            st.plotly_chart(chart_layout(fig, height=380, y_title="Wins"), use_container_width=True)
            render_caption(
                "The projection should be read as a range, not an exact number. The model does not know future injuries, quarterback changes, coaching changes, or roster movement."
            )


with tab_details:
    st.markdown('<div class="section-title">Model Details</div>', unsafe_allow_html=True)
    render_takeaway(
        "This section explains what the model uses, how it was tested, and why the results should be treated as exploratory."
    )

    if model_bundle.get("available"):
        tested_text = (
            f"Earlier seasons ({min(model_bundle['train_seasons'])}-{max(model_bundle['train_seasons'])}) are used to train the model. "
            f"Later seasons ({min(model_bundle['test_seasons'])}-{max(model_bundle['test_seasons'])}) are held out to test whether the model works on seasons it has not seen."
        )
        mae_text = f"If MAE is {model_bundle['metrics']['model_mae']:.1f}, the model is usually off by about {model_bundle['metrics']['model_mae']:.1f} wins."
    else:
        tested_text = model_bundle.get("message", "The model could not be tested from the available files.")
        mae_text = "MAE is unavailable because the model could not be trained."

    st.markdown(
        f"""
<div class="predict-card explain-box">
    <div class="explain-title">What the Model Is Trying to Predict</div>
    The model uses a team's selected regular-season profile to estimate how many games it may win the following regular season.
</div>
<div class="predict-card explain-box">
    <div class="explain-title">How the Model Was Tested</div>
    Earlier seasons are used to train the model. Later seasons are held out to test whether the model works on seasons it has not seen.
    <br><br>{tested_text}
</div>
<div class="predict-card explain-box">
    <div class="explain-title">How to Read the Error</div>
    {mae_text} Lower MAE is better. Accuracy on the 10+ win classifier is useful context, but it can be misleading if most teams fall into one class.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">What the Model Uses</div>', unsafe_allow_html=True)
    available_features = set(model_bundle.get("features", get_feature_candidates(model_df)))
    feature_groups = {
        "Overall Team Quality": ["current_wins", "gini_score", "point_diff_per_game", "success_margin"],
        "Unit Strength": ["offense_estat", "defense_estat", "offense_rank", "defense_rank", "balance_gap"],
        "Context and Volatility": ["turnover_margin_per_game", "penalty_yards_margin_per_game", "schedule_strength", "sos_rank"],
    }
    group_cards = []
    for group_title, group_features in feature_groups.items():
        readable_items = [display_label(feature) for feature in group_features if feature in available_features]
        if not readable_items:
            readable_items = ["No supported fields available in the current files"]
        item_html = "".join(f"<li>{item}</li>" for item in readable_items)
        group_cards.append(
            f"""
<div class="predict-card feature-group-card">
    <div class="feature-group-title">{group_title}</div>
    <ul class="feature-list">{item_html}</ul>
</div>
"""
        )
    st.markdown(f"<div class='feature-group-grid'>{''.join(group_cards)}</div>", unsafe_allow_html=True)
    render_caption(
        "The model uses selected-season traits only. It does not use following-season wins or future results as inputs."
    )

    with st.expander("Technical column names"):
        feature_names = [display_label(feature) for feature in model_bundle.get("features", get_feature_candidates(model_df))]
        technical_features = pd.DataFrame(
            {
                "Readable Feature": feature_names,
                "Technical Column Name": model_bundle.get("features", get_feature_candidates(model_df)),
            }
        )
        st.dataframe(technical_features, hide_index=True, use_container_width=True)

    if model_bundle.get("available") and model_bundle.get("class_metrics"):
        st.markdown('<div class="section-title">10+ Win Classifier</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="predict-card explain-box">
    The classifier estimates whether a team is likely to win at least 10 games the following season.
    It should be treated as a rough signal, not a playoff guarantee.
</div>
""",
            unsafe_allow_html=True,
        )
        cm = model_bundle["class_metrics"]["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Under 10", "Actual 10+"],
            columns=["Predicted Under 10", "Predicted 10+"],
        )
        metric_cols = st.columns([1, 2])
        with metric_cols[0]:
            render_metric_card("Classifier Accuracy", percent_text(model_bundle["class_metrics"]["accuracy"]), "Held-out seasons")
        with metric_cols[1]:
            st.dataframe(cm_df, use_container_width=True)
        render_caption(
            "Accuracy can be misleading if one class is much more common, so this table shows where the classifier was right and wrong."
        )

    notes = build_notes.copy()
    if skipped_columns:
        readable_skips = ", ".join(display_label(column) for column in skipped_columns)
        notes.append(f"Some fields were missing or had to be derived around: {readable_skips}.")
    if not notes:
        notes.append("All required core modeling fields were available.")

    st.markdown('<div class="section-title">Future Expansion: Postseason Predictor</div>', unsafe_allow_html=True)
    postseason_note = (
        "The project files include postseason rows, so a future tab could study whether regular-season Gini profiles predict playoff wins, playoff advancement, or Super Bowl paths. "
        "I am keeping that as a future expansion here so this page stays focused on following regular-season wins."
        if postseason_rows_available
        else "A postseason prediction model could be added later, but this page currently focuses on following regular-season wins because postseason rows are not clearly available in the current modeling dataset."
    )
    st.markdown(
        f"""
<div class="predict-card explain-box">
    {postseason_note}
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Notes and Limitations</div>', unsafe_allow_html=True)
    limitations = [
        "Injuries, quarterback changes, coaching changes, roster turnover, schedule changes, and offseason events are not fully captured.",
        "This is an exploratory football research model, not betting advice.",
        "The model avoids using following-season results as features.",
        "The current page focuses on following regular-season wins.",
    ]
    details_text = "".join(f"<li>{item}</li>" for item in notes + limitations)
    st.markdown(
        f"""
<div class="predict-card explain-box">
    <div class="explain-title">Notes</div>
    <ul>{details_text}</ul>
</div>
""",
        unsafe_allow_html=True,
    )
    render_caption(
        "These limitations are why the page should be read as football research and context, not a guaranteed forecasting tool."
    )


st.markdown("</div>", unsafe_allow_html=True)
