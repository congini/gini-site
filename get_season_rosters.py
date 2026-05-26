import pandas as pd

seasons = range(2005, 2026)
all_rosters = []

for season in seasons:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"

    try:
        df = pd.read_csv(url)
        df["season"] = season
        all_rosters.append(df)
        print(f"Loaded {season}")

    except Exception as e:
        print(f"Skipped {season}: {e}")

season_rosters = pd.concat(all_rosters, ignore_index=True)

season_rosters.to_csv("nfl_season_rosters_2005_2025.csv", index=False)

print("Done.")
print(season_rosters.head())
print(season_rosters.shape)