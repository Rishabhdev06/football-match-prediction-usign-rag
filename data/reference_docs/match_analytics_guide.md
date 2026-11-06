# Football Match Outcome Analytics and Prediction Principles

## 1. Strictly Pre-Match Predictive Signals
To predict outcomes without future data leakage, only variables known before kickoff may be considered:
- **Rolling Form (Last 5 Games)**: Points accumulated, goals scored, and goals conceded in immediate prior matches reflect current momentum and tactical cohesion.
- **Goal Differentials**: A positive rolling goal difference indicates offensive efficiency and defensive solidity.
- **Rest & Schedule Fatigue**: Teams playing with fewer than 4 days of rest between high-intensity fixtures show measurable dips in late-game pressing and defensive transitions.
- **Pre-Match Betting Odds**: Decimal odds from bookmakers (e.g. Bet365) reflect public knowledge, injury rumors, tactical matchups, and professional handicapper consensus prior to the match.

## 2. In-Match Statistics vs. Pre-Match Predictors
- In-match statistics (full-time goals, shots on target, corners, fouls, yellow/red cards) are **results** of the match, not prior predictors.
- Using in-match statistics to predict the final outcome constitutes **target leakage** and must never be permitted in predictive modeling.

## 3. Correlation vs. Causation in Sports Analytics
- Historical correlation between team statistics and outcomes does not guarantee future results.
- Football is a low-scoring sport with significant variance. Single moments (e.g., deflection, early red card, missed penalty) frequently alter the outcome of a match.
- Analysts must avoid stating that historical head-to-head records or recent form cause a future win or loss.
