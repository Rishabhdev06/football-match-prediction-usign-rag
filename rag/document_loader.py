"""
Document Ingestion & Chunking Pipeline for Football RAG
======================================================
Converts raw match CSV records, team performance summaries,
and markdown reference documents into structured, metadata-rich
text chunks with deterministic, stable IDs.
"""

import os
import glob
import re
from datetime import datetime
import pandas as pd
import numpy as np


def parse_iso_date(date_val):
    """Parses mixed date formats into an ISO YYYY-MM-DD string."""
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d'):
        try:
            return datetime.strptime(str(date_val).strip(), fmt).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
    dt = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
    if pd.isna(dt):
        return '2023-01-01'
    return dt.strftime('%Y-%m-%d')


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_data_path(rel_path):
    candidates = [
        rel_path,
        os.path.join(PROJECT_ROOT, rel_path),
        os.path.join(os.getcwd(), rel_path),
        os.path.join(PROJECT_ROOT, 'football_match_prediction_rag', rel_path),
        os.path.join(os.getcwd(), 'football_match_prediction_rag', rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(PROJECT_ROOT, rel_path)


def clean_identifier(name):
    """Converts a team or document name into a safe alphanumeric identifier."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(name).strip()).strip('_')


def load_match_chunks(csv_path=None):
    """
    Reads historical matches from CSV and converts each match into
    a searchable natural language document with verified metadata.
    """
    if csv_path is None:
        csv_path = resolve_data_path('data/matches.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Match dataset not found at {csv_path}')

    df = pd.read_csv(csv_path)
    chunks = []

    for idx, row in df.iterrows():
        home_team = str(row.get('HomeTeam', 'Unknown')).strip()
        away_team = str(row.get('AwayTeam', 'Unknown')).strip()
        raw_date = str(row.get('Date', '01/01/2023')).strip()
        iso_date = parse_iso_date(raw_date)

        # Derive season from match date
        dt = datetime.strptime(iso_date, '%Y-%m-%d')
        season = '2022-2023' if dt < datetime(2023, 7, 1) else '2023-2024'

        fthg = int(row['FTHG']) if 'FTHG' in row and not pd.isna(row['FTHG']) else 0
        ftag = int(row['FTAG']) if 'FTAG' in row and not pd.isna(row['FTAG']) else 0
        ftr = str(row.get('FTR', 'D')).strip()

        hthg = int(row['HTHG']) if 'HTHG' in row and not pd.isna(row['HTHG']) else 0
        htag = int(row['HTAG']) if 'HTAG' in row and not pd.isna(row['HTAG']) else 0
        htr = str(row.get('HTR', 'D')).strip()

        referee = str(row.get('Referee', 'Match Official')).strip()
        hs = int(row.get('HS', 0)) if not pd.isna(row.get('HS', 0)) else 0
        as_shots = int(row.get('AS', 0)) if not pd.isna(row.get('AS', 0)) else 0
        hst = int(row.get('HST', 0)) if not pd.isna(row.get('HST', 0)) else 0
        ast = int(row.get('AST', 0)) if not pd.isna(row.get('AST', 0)) else 0
        hc = int(row.get('HC', 0)) if not pd.isna(row.get('HC', 0)) else 0
        ac = int(row.get('AC', 0)) if not pd.isna(row.get('AC', 0)) else 0
        hf = int(row.get('HF', 0)) if not pd.isna(row.get('HF', 0)) else 0
        af = int(row.get('AF', 0)) if not pd.isna(row.get('AF', 0)) else 0
        hy = int(row.get('HY', 0)) if not pd.isna(row.get('HY', 0)) else 0
        ay = int(row.get('AY', 0)) if not pd.isna(row.get('AY', 0)) else 0
        hr = int(row.get('HR', 0)) if not pd.isna(row.get('HR', 0)) else 0
        ar = int(row.get('AR', 0)) if not pd.isna(row.get('AR', 0)) else 0

        # Pre-match odds
        b365h = float(row.get('B365H', 0.0)) if not pd.isna(row.get('B365H', 0.0)) else 0.0
        b365d = float(row.get('B365D', 0.0)) if not pd.isna(row.get('B365D', 0.0)) else 0.0
        b365a = float(row.get('B365A', 0.0)) if not pd.isna(row.get('B365A', 0.0)) else 0.0

        outcome_desc = 'Home Win' if ftr == 'H' else ('Away Win' if ftr == 'A' else 'Draw')
        ht_desc = 'Home Lead' if htr == 'H' else ('Away Lead' if htr == 'A' else 'Level')

        date_compact = iso_date.replace('-', '')
        stable_id = f'match_{season}_{date_compact}_{clean_identifier(home_team)}_vs_{clean_identifier(away_team)}'

        odds_text = f'Pre-match Bet365 odds: Home {b365h:.2f}, Draw {b365d:.2f}, Away {b365a:.2f}.' if b365h > 0 else 'Pre-match odds: Not recorded.'

        content = (
            f'Match Record (EPL {season}): On {raw_date} ({iso_date}), {home_team} played against {away_team} at home (Referee: {referee}). '
            f'Final Score: {home_team} {fthg} - {ftag} {away_team}. Full-Time Result: {outcome_desc}. '
            f'Half-Time Score: {hthg} - {htag} ({ht_desc}). '
            f'Match Statistics: '
            f'{home_team} (Home) had {hs} shots ({hst} on target), {hc} corners, {hf} fouls committed, {hy} yellow cards, {hr} red cards. '
            f'{away_team} (Away) had {as_shots} shots ({ast} on target), {ac} corners, {af} fouls committed, {ay} yellow cards, {ar} red cards. '
            f'{odds_text}'
        )

        metadata = {
            'chunk_id': stable_id,
            'source': 'data/matches.csv',
            'doc_type': 'match_record',
            'date': iso_date,
            'season': season,
            'home_team': home_team,
            'away_team': away_team,
            'fthg': fthg,
            'ftag': ftag,
            'ftr': ftr,
            'referee': referee,
            'row_index': idx
        }

        chunks.append({
            'id': stable_id,
            'content': content,
            'metadata': metadata
        })

    return chunks


def load_team_summary_chunks(csv_path=None):
    """
    Computes verified statistical aggregates per team and season
    to answer high-level queries directly from factual match records.
    """
    if csv_path is None:
        csv_path = resolve_data_path('data/matches.csv')
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    df['iso_date'] = df['Date'].apply(parse_iso_date)
    df['dt'] = pd.to_datetime(df['iso_date'])
    df['season'] = df['dt'].apply(lambda d: '2022-2023' if d < datetime(2023, 7, 1) else '2023-2024')

    teams = sorted(list(set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())))
    seasons = ['2022-2023', '2023-2024']

    summary_chunks = []

    for season in seasons:
        season_df = df[df['season'] == season]
        if season_df.empty:
            continue

        for team in teams:
            home_matches = season_df[season_df['HomeTeam'] == team]
            away_matches = season_df[season_df['AwayTeam'] == team]
            total_matches = len(home_matches) + len(away_matches)
            if total_matches == 0:
                continue

            h_wins = (home_matches['FTR'] == 'H').sum()
            h_draws = (home_matches['FTR'] == 'D').sum()
            h_losses = (home_matches['FTR'] == 'A').sum()
            h_gf = home_matches['FTHG'].sum()
            h_ga = home_matches['FTAG'].sum()

            a_wins = (away_matches['FTR'] == 'A').sum()
            a_draws = (away_matches['FTR'] == 'D').sum()
            a_losses = (away_matches['FTR'] == 'H').sum()
            a_gf = away_matches['FTAG'].sum()
            a_ga = away_matches['FTHG'].sum()

            total_wins = h_wins + a_wins
            total_draws = h_draws + a_draws
            total_losses = h_losses + a_losses
            total_gf = h_gf + a_gf
            total_ga = h_ga + a_ga
            total_pts = (total_wins * 3) + total_draws

            stable_id = f'summary_{season}_{clean_identifier(team)}'

            content = (
                f'Team Season Summary: In the {season} Premier League season, {team} played {total_matches} matches. '
                f'Record: {total_wins} Wins, {total_draws} Draws, {total_losses} Losses, accumulating {total_pts} points. '
                f'Goals: {total_gf} Goals Scored, {total_ga} Goals Conceded (Goal Difference: {total_gf - total_ga:+d}). '
                f'Home Record: {h_wins}W - {h_draws}D - {h_losses}L ({h_gf} GF, {h_ga} GA). '
                f'Away Record: {a_wins}W - {a_draws}D - {a_losses}L ({a_gf} GF, {a_ga} GA).'
            )

            metadata = {
                'chunk_id': stable_id,
                'source': 'data/matches.csv',
                'doc_type': 'team_summary',
                'season': season,
                'team': team,
                'date': '2023-05-30' if season == '2022-2023' else '2024-05-30'
            }

            summary_chunks.append({
                'id': stable_id,
                'content': content,
                'metadata': metadata
            })

    return summary_chunks


def load_reference_document_chunks(docs_dir=None, dataset_readme=None):
    """
    Chunks markdown reference documentation by top-level or second-level headings
    with stable deterministic IDs.
    """
    if docs_dir is None:
        docs_dir = resolve_data_path('data/reference_docs')
    if dataset_readme is None:
        dataset_readme = resolve_data_path('data/README.md')
    chunks = []
    file_paths = []
    if os.path.exists(dataset_readme):
        file_paths.append(dataset_readme)
    if os.path.exists(docs_dir):
        file_paths.extend(glob.glob(os.path.join(docs_dir, '*.md')))

    for path in file_paths:
        filename = os.path.basename(path)
        base_name = clean_identifier(os.path.splitext(filename)[0])
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split by markdown headers
        sections = re.split(r"\n(?=#{1,3}\s)", text)
        for s_idx, sec in enumerate(sections):
            cleaned_sec = sec.strip()
            if len(cleaned_sec) < 40:
                continue
            chunk_id = f'ref_{base_name}_part{s_idx+1}'
            # Extract first heading line as title
            lines = cleaned_sec.split("\n")
            title = lines[0].replace('#', '').strip()

            metadata = {
                'chunk_id': chunk_id,
                'source': filename,
                'doc_type': 'reference_document',
                'title': title,
                'date': '2024-06-01'
            }
            chunks.append({
                'id': chunk_id,
                'content': cleaned_sec,
                'metadata': metadata
            })

    return chunks


def load_all_documents():
    """Aggregates all match records, team summaries, and reference documentation."""
    all_chunks = []
    all_chunks.extend(load_match_chunks())
    all_chunks.extend(load_team_summary_chunks())
    all_chunks.extend(load_reference_document_chunks())
    return all_chunks
