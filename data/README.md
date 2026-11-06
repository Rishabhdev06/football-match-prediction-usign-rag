# Dataset Documentation: Football Match Outcomes

## 1. Overview and Data Source
This dataset contains historical English Premier League match records collected from **[Football-Data.co.uk](https://www.football-data.co.uk/data.php)**, a widely recognized, public repository maintained by sports statistician Joseph Buchdahl.

- **Primary League**: English Premier League (Division code `E0`)
- **Seasons Included**:
  - `2022–2023` (380 matches)
  - `2023–2024` (380 matches)
- **Total Bundled Records**: 760 real professional league matches
- **Target Variable**: `FTR` (Full Time Result: `H` = Home Win, `D` = Draw, `A` = Away Win)

---

## 2. Key Data Columns

| Column Name | Category | Description | Availability for ML |
|:---|:---|:---|:---|
| `Div` | Metadata | League Division (`E0` = English Premier League) | Metadata |
| `Date` | Metadata | Match date (`DD/MM/YYYY` or `DD/MM/YY`) | Chronological Sorting |
| `Time` | Metadata | Kickoff time (UK local time) | Metadata |
| `HomeTeam` | Match Info | Name of the home club | Team Identity |
| `AwayTeam` | Match Info | Name of the away club | Team Identity |
| `FTHG` | Match Result | Full Time Home Goals scored | **Post-match target calculation only** |
| `FTAG` | Match Result | Full Time Away Goals scored | **Post-match target calculation only** |
| `FTR` | **Target** | **Full Time Result (`H`, `D`, `A`)** | **Prediction Target (y)** |
| `HTHG`, `HTAG` | In-game | Half Time Goals | **Excluded** (Not known pre-match) |
| `HS`, `AS` | In-game | Total Shots | **Excluded** (In-match statistic) |
| `HST`, `AST` | In-game | Shots on Target | **Excluded** (In-match statistic) |
| `HC`, `AC` | In-game | Corner Kicks | **Excluded** (In-match statistic) |
| `HF`, `AF` | In-game | Fouls Committed | **Excluded** (In-match statistic) |
| `HY`, `AY` | In-game | Yellow Cards | **Excluded** (In-match statistic) |
| `HR`, `AR` | In-game | Red Cards | **Excluded** (In-match statistic) |
| `B365H` | Pre-match Odds | Bet365 Home Win decimal odds | **Pre-match feature** |
| `B365D` | Pre-match Odds | Bet365 Draw decimal odds | **Pre-match feature** |
| `B365A` | Pre-match Odds | Bet365 Away Win decimal odds | **Pre-match feature** |
| `AvgH`, `AvgD`, `AvgA` | Pre-match Odds | Market Average decimal odds | **Pre-match feature** |

---

## 3. Strict Data Leakage Prevention

In sports analytics, **data leakage** occurs when information from the future or in-game events is mistakenly provided to a model making predictions about the game.

### Strict Rules Followed in this Project:
1. **Never use in-match stats**: Metrics such as shots on target (`HST`), fouls (`HF`), and half-time goals (`HTHG`) occur *during* the match. Including them would produce falsely inflated accuracy that fails in real-world scenarios.
2. **Strict Chronological Splitting**: We sort matches strictly by date. We train exclusively on past matches and test on future matches:
   28684\text{Train Set} = \text{Earliest 75% of matches in time}28684
   28684\text{Test Set} = \text{Latest 25% of matches in time}28684
   *Random shuffling is strictly forbidden* because training on a match from April to predict a match from September constitutes temporal leakage.
3. **Pre-Match Rolling Form**: When computing past form (e.g. goals scored in last 5 games), the rolling window looks *strictly backward* at matches finished before the kickoff date of the current match.

---

## 4. Feature Engineering: Pre-Match Signals

From raw match dates and historical results, `train_model.py` derives the following pre-kickoff features:

1. **Pre-Match Rolling Form (Last 5 Games)**:
   - `home_points_last5` / `away_points_last5`: Points earned in the team's last 5 games (3 for win, 1 for draw, 0 for loss).
   - `home_goals_scored_last5` / `away_goals_scored_last5`: Goals scored over the last 5 games.
   - `home_goals_conceded_last5` / `away_goals_conceded_last5`: Goals conceded over the last 5 games.
   - `home_goal_diff_last5` / `away_goal_diff_last5`: Net goal difference in recent matches.
   - `home_win_rate_last5` / `away_win_rate_last5`: Proportion of wins in last 5 matches.
2. **Rest & Schedule Fatigue**:
   - `home_rest_days` / `away_rest_days`: Number of days since the team's previous competitive fixture (capped between 2 and 30 days to mitigate off-season distortions).
3. **Pre-Match Implied Probabilities**:
   Bookmaker decimal odds represent the pre-game consensus probability distribution with an added margin ("overround"). We remove the overround and normalize to valid probabilities:
   28684P_{\text{home}} = \frac{1 / \text{odds}_H}{(1 / \text{odds}_H) + (1 / \text{odds}_D) + (1 / \text{odds}_A)}28684
   28684P_{\text{draw}} = \frac{1 / \text{odds}_D}{(1 / \text{odds}_H) + (1 / \text{odds}_D) + (1 / \text{odds}_A)}28684
   28684P_{\text{away}} = \frac{1 / \text{odds}_A}{(1 / \text{odds}_H) + (1 / \text{odds}_D) + (1 / \text{odds}_A)}28684

---

## 5. Missing Values & Data Cleaning Protocol

1. **Date Parsing**: Dates formatted as `DD/MM/YYYY` or `DD/MM/YY` are normalized into unified `datetime` objects.
2. **Early Season Matches (Cold Start)**:
   - In Matchweeks 1 to 4, a team has fewer than 5 historical games.
   - The pipeline computes rolling averages over whatever games exist. If 0 games exist (Matchweek 1), historical league baseline means are imputed (e.g., 1.35 points/match, 1.4 goals/match).
3. **Missing Odds**:
   - If a specific bookmaker column has a missing entry, market consensus averages (`AvgH`, `AvgD`, `AvgA`) or column median values are substituted.
4. **Invalid Records**:
   - Any row missing `HomeTeam`, `AwayTeam`, or target `FTR` is dropped.
