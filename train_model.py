"""
AI-Based Football Match Outcome Prediction Using Machine Learning
==================================================================
Module: train_model.py
Description:
    End-to-end data loading, pre-match feature engineering,
    chronological train/test splitting, model training, evaluation,
    and model artifact serialization.

Authors: Student ML Project
Dataset Source: Football-Data.co.uk
"""

import os
import json
import argparse
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# Suppress harmless macOS Accelerate BLAS floating point status flags and sklearn deprecations
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Label mappings
CLASS_NAMES = ['Home Win', 'Draw', 'Away Win']
TARGET_MAP = {'H': 0, 'D': 1, 'A': 2}
INV_TARGET_MAP = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

FEATURE_COLUMNS = [
    'home_points_last5',
    'away_points_last5',
    'home_goals_scored_last5',
    'away_goals_scored_last5',
    'home_goals_conceded_last5',
    'away_goals_conceded_last5',
    'home_goal_diff_last5',
    'away_goal_diff_last5',
    'home_win_rate_last5',
    'away_win_rate_last5',
    'home_rest_days',
    'away_rest_days',
    'implied_prob_home',
    'implied_prob_draw',
    'implied_prob_away'
]


def load_dataset(file_path_or_buffer):
    """
    Loads football match CSV data, standardizes date formats,
    and sorts rows chronologically.
    Handles DataFrame, filepath string, or file-like buffers.
    """
    if isinstance(file_path_or_buffer, pd.DataFrame):
        df = file_path_or_buffer.copy()
    elif isinstance(file_path_or_buffer, str):
        if not os.path.exists(file_path_or_buffer):
            raise FileNotFoundError(f'Dataset file not found at: {file_path_or_buffer}')
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_csv(file_path_or_buffer)

    # Validate essential columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Dataset is missing required columns: {missing}')

    # Clean rows with missing target or teams
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).copy()
    df = df[df['FTR'].isin(['H', 'D', 'A'])].copy()

    # Parse dates safely handling multiple potential formats
    def parse_match_date(date_val):
        for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d'):
            try:
                return datetime.strptime(str(date_val).strip(), fmt)
            except (ValueError, TypeError):
                continue
        return pd.to_datetime(date_val, dayfirst=True, errors='coerce')

    df['ParsedDate'] = df['Date'].apply(parse_match_date)
    df = df.dropna(subset=['ParsedDate']).copy()

    # Sort strictly chronologically to preserve temporal order
    df = df.sort_values(by=['ParsedDate']).reset_index(drop=True)

    return df


def calculate_implied_probabilities(row):
    """
    Converts pre-match bookmaker odds into normalized implied probabilities.
    Prefers Bet365 odds (B365H, B365D, B365A), falls back to Market Average (AvgH, AvgD, AvgA),
    or balanced defaults if unrecorded.
    """
    odds_h, odds_d, odds_a = np.nan, np.nan, np.nan

    has_key = (lambda k: k in row.index) if hasattr(row, 'index') else (lambda k: k in row)
    for prefix in ['B365', 'Avg', 'Max', 'WH', 'BW']:
        h_col, d_col, a_col = f'{prefix}H', f'{prefix}D', f'{prefix}A'
        if has_key(h_col) and has_key(d_col) and has_key(a_col):
            try:
                vh = float(row[h_col])
                vd = float(row[d_col])
                va = float(row[a_col])
                if vh > 1.0 and vd > 1.0 and va > 1.0:
                    odds_h, odds_d, odds_a = vh, vd, va
                    break
            except (ValueError, TypeError):
                continue

    if np.isnan(odds_h) or np.isnan(odds_d) or np.isnan(odds_a):
        return 0.45, 0.25, 0.30

    raw_h = 1.0 / odds_h
    raw_d = 1.0 / odds_d
    raw_a = 1.0 / odds_a
    total = raw_h + raw_d + raw_a

    return raw_h / total, raw_d / total, raw_a / total


def build_pre_match_features(df):
    """
    Computes pre-match rolling form features using strictly past match history.
    NO future data or post-match statistics are leaked into feature generation.
    """
    df = df.sort_values(by=['ParsedDate']).reset_index(drop=True)

    team_history = {}
    rows = []

    for idx, row in df.iterrows():
        match_date = row['ParsedDate']
        home_team = str(row['HomeTeam']).strip()
        away_team = str(row['AwayTeam']).strip()

        def get_team_rolling_stats(team_name, current_date):
            hist = team_history.get(team_name, [])
            past_matches = [m for m in hist if m['date'] < current_date]

            if not past_matches:
                return {
                    'points_last5': 6.5,
                    'goals_scored_last5': 6.5,
                    'goals_conceded_last5': 6.5,
                    'goal_diff_last5': 0.0,
                    'win_rate_last5': 0.38,
                    'rest_days': 7.0
                }

            last_matches = past_matches[-5:]
            n = len(last_matches)
            pts = sum(m['points'] for m in last_matches)
            gf = sum(m['goals_for'] for m in last_matches)
            ga = sum(m['goals_against'] for m in last_matches)
            wins = sum(1 for m in last_matches if m['is_win'])

            scale = 5.0 / n
            scaled_pts = pts * scale
            scaled_gf = gf * scale
            scaled_ga = ga * scale
            win_rate = wins / n

            last_date = past_matches[-1]['date']
            days_diff = (current_date - last_date).total_seconds() / 86400.0
            rest_days = max(2.0, min(30.0, float(days_diff)))

            return {
                'points_last5': round(scaled_pts, 2),
                'goals_scored_last5': round(scaled_gf, 2),
                'goals_conceded_last5': round(scaled_ga, 2),
                'goal_diff_last5': round(scaled_gf - scaled_ga, 2),
                'win_rate_last5': round(win_rate, 3),
                'rest_days': round(rest_days, 1)
            }

        home_stats = get_team_rolling_stats(home_team, match_date)
        away_stats = get_team_rolling_stats(away_team, match_date)

        prob_h, prob_d, prob_a = calculate_implied_probabilities(row)

        feature_dict = {
            'match_idx': idx,
            'Date': row['Date'],
            'ParsedDate': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTR': row['FTR'],
            'target': TARGET_MAP[row['FTR']],
            'home_points_last5': home_stats['points_last5'],
            'home_goals_scored_last5': home_stats['goals_scored_last5'],
            'home_goals_conceded_last5': home_stats['goals_conceded_last5'],
            'home_goal_diff_last5': home_stats['goal_diff_last5'],
            'home_win_rate_last5': home_stats['win_rate_last5'],
            'home_rest_days': home_stats['rest_days'],
            'away_points_last5': away_stats['points_last5'],
            'away_goals_scored_last5': away_stats['goals_scored_last5'],
            'away_goals_conceded_last5': away_stats['goals_conceded_last5'],
            'away_goal_diff_last5': away_stats['goal_diff_last5'],
            'away_win_rate_last5': away_stats['win_rate_last5'],
            'away_rest_days': away_stats['rest_days'],
            'implied_prob_home': prob_h,
            'implied_prob_draw': prob_d,
            'implied_prob_away': prob_a
        }
        rows.append(feature_dict)

        has_row_key = (lambda k: k in row.index) if hasattr(row, 'index') else (lambda k: k in row)
        fthg = float(row['FTHG']) if has_row_key('FTHG') and not pd.isna(row['FTHG']) else 0.0
        ftag = float(row['FTAG']) if has_row_key('FTAG') and not pd.isna(row['FTAG']) else 0.0
        ftr = row['FTR']

        if ftr == 'H':
            home_pts, away_pts = 3, 0
            home_win, away_win = True, False
        elif ftr == 'A':
            home_pts, away_pts = 0, 3
            home_win, away_win = False, True
        else:
            home_pts, away_pts = 1, 1
            home_win, away_win = False, False

        team_history.setdefault(home_team, []).append({
            'date': match_date,
            'points': home_pts,
            'goals_for': fthg,
            'goals_against': ftag,
            'is_win': home_win
        })
        team_history.setdefault(away_team, []).append({
            'date': match_date,
            'points': away_pts,
            'goals_for': ftag,
            'goals_against': fthg,
            'is_win': away_win
        })

    featured_df = pd.DataFrame(rows)
    return featured_df, team_history


def get_latest_team_profile(team_name, team_history):
    """
    Returns the most recent pre-match form profile for a team from history.
    """
    hist = team_history.get(team_name, [])
    if not hist:
        return {
            'points_last5': 6.5,
            'goals_scored_last5': 6.5,
            'goals_conceded_last5': 6.5,
            'goal_diff_last5': 0.0,
            'win_rate_last5': 0.38,
            'rest_days': 7.0
        }
    last_matches = hist[-5:]
    n = len(last_matches)
    scale = 5.0 / n
    pts = sum(m['points'] for m in last_matches) * scale
    gf = sum(m['goals_for'] for m in last_matches) * scale
    ga = sum(m['goals_against'] for m in last_matches) * scale
    wins = sum(1 for m in last_matches if m['is_win'])
    return {
        'points_last5': round(pts, 2),
        'goals_scored_last5': round(gf, 2),
        'goals_conceded_last5': round(ga, 2),
        'goal_diff_last5': round(gf - ga, 2),
        'win_rate_last5': round(wins / n, 3),
        'rest_days': 7.0
    }


def split_chronologically(featured_df, train_ratio=0.75):
    """
    Splits the dataset sequentially in time to avoid temporal data leakage.
    """
    n_total = len(featured_df)
    n_train = int(n_total * train_ratio)

    train_df = featured_df.iloc[:n_train].copy()
    test_df = featured_df.iloc[n_train:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['target'].values

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['target'].values

    return train_df, test_df, X_train, y_train, X_test, y_test


def evaluate_model(model, X_test, y_test, is_scaled=False, scaler=None):
    """
    Evaluates a trained classifier on the test set.
    """
    X_eval = scaler.transform(X_test) if (is_scaled and scaler is not None) else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    return {
        'accuracy': round(float(acc), 4),
        'macro_precision': round(float(prec), 4),
        'macro_recall': round(float(rec), 4),
        'macro_f1': round(float(f1), 4),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_proba.tolist()
    }


def train_and_save_pipeline(data_path='data/matches.csv', output_dir='models', train_ratio=0.75):
    """
    High-level training pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f'[1/5] Loading match dataset from {data_path}...')
    df = load_dataset(data_path)
    print(f'      Total valid matches loaded: {len(df)}')

    print('[2/5] Engineering pre-match rolling form and implied probability features...')
    featured_df, team_history = build_pre_match_features(df)

    print(f'[3/5] Chronologically splitting dataset ({int(train_ratio*100)}% train / {int((1-train_ratio)*100)}% test)...')
    train_df, test_df, X_train, y_train, X_test, y_test = split_chronologically(featured_df, train_ratio=train_ratio)

    train_start = train_df['ParsedDate'].min().strftime('%Y-%m-%d')
    train_end = train_df['ParsedDate'].max().strftime('%Y-%m-%d')
    test_start = test_df['ParsedDate'].min().strftime('%Y-%m-%d')
    test_end = test_df['ParsedDate'].max().strftime('%Y-%m-%d')
    print(f'      Train set: {len(train_df)} matches ({train_start} to {train_end})')
    print(f'      Test set:  {len(test_df)} matches ({test_start} to {test_end})')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. Logistic Regression
    print('[4/5] Training Logistic Regression model...')
    lr_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, is_scaled=True, scaler=scaler)

    # 2. Random Forest
    print('[5/5] Training Random Forest model...')
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, is_scaled=False)

    print("\n================ MODEL EVALUATION SUMMARY ================")
    print('Metric              | Logistic Regression | Random Forest')
    print('--------------------+---------------------+--------------')
    print(f'Accuracy            | {lr_metrics["accuracy"]:^19.4f} | {rf_metrics["accuracy"]:^14.4f}')
    print(f'Macro Precision     | {lr_metrics["macro_precision"]:^19.4f} | {rf_metrics["macro_precision"]:^14.4f}')
    print(f'Macro Recall        | {lr_metrics["macro_recall"]:^19.4f} | {rf_metrics["macro_recall"]:^14.4f}')
    print(f'Macro F1-Score      | {lr_metrics["macro_f1"]:^19.4f} | {rf_metrics["macro_f1"]:^14.4f}')
    print('==========================================================')

    joblib.dump(lr_model, os.path.join(output_dir, 'logistic_regression.joblib'))
    joblib.dump(rf_model, os.path.join(output_dir, 'random_forest.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))

    serialized_history = {}
    for team, games in team_history.items():
        serialized_history[team] = [
            {
                'date': g['date'].strftime('%Y-%m-%d'),
                'points': g['points'],
                'goals_for': g['goals_for'],
                'goals_against': g['goals_against'],
                'is_win': g['is_win']
            }
            for g in games
        ]
    with open(os.path.join(output_dir, 'team_history.json'), 'w', encoding='utf-8') as f:
        json.dump(serialized_history, f, indent=2)

    with open(os.path.join(output_dir, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    metrics_payload = {
        'dataset_info': {
            'total_matches': len(featured_df),
            'train_matches': len(train_df),
            'test_matches': len(test_df),
            'train_date_range': [train_start, train_end],
            'test_date_range': [test_start, test_end],
            'classes': CLASS_NAMES
        },
        'logistic_regression': lr_metrics,
        'random_forest': rf_metrics
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    print(f'Artifacts successfully saved to "{output_dir}/"')
    return metrics_payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Football Match Outcome Predictors')
    parser.add_argument('--data', default='data/matches.csv', help='Path to match CSV file')
    parser.add_argument('--output', default='models', help='Directory to save model artifacts')
    parser.add_argument('--split', type=float, default=0.75, help='Chronological train split ratio')
    args = parser.parse_args()

    train_and_save_pipeline(data_path=args.data, output_dir=args.output, train_ratio=args.split)
