import pandas as pd

df = pd.read_csv("nfl_season_rosters_2005_2025.csv", low_memory=False)

clean = df[
    [
        "season",
        "team",
        "position",
        "full_name",
        "first_name",
        "last_name",
        "jersey_number",
        "status",
        "height",
        "weight",
        "college",
        "rookie_year",
        "entry_year",
        "draft_club",
        "draft_number"
    ]
].copy()

clean.to_csv("nfl_season_rosters_clean_2005_2025.csv", index=False)

print("Done.")
print(clean.head(20))
print(clean.shape)