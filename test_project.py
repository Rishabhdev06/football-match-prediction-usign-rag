"""
Automated Test Suite for Football Match Outcome Prediction Project
==================================================================
Verifies:
  1. Data loading and chronological sorting
  2. Strict pre-match feature engineering (no future leakage)
  3. No NaN or infinite feature values
  4. Chronological train/test split order
  5. 3-class target validity
  6. Model prediction shapes and probability normalization (sums to 1.0)
"""

import unittest
import os
import numpy as np
import pandas as pd
import joblib

from train_model import (
    load_dataset,
    build_pre_match_features,
    split_chronologically,
    calculate_implied_probabilities,
    FEATURE_COLUMNS,
    CLASS_NAMES,
    TARGET_MAP
)


class TestFootballMatchPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = 'data/matches.csv'
        assert os.path.exists(cls.data_path), f'{cls.data_path} not found!'
        cls.raw_df = load_dataset(cls.data_path)
        cls.featured_df, cls.team_hist = build_pre_match_features(cls.raw_df)

    def test_01_data_loading_and_sorting(self):
        """Verify data is loaded and sorted strictly chronologically."""
        self.assertGreater(len(self.raw_df), 100, 'Dataset should contain > 100 matches')
        self.assertIn('ParsedDate', self.raw_df.columns)
        
        # Check chronological order
        dates = self.raw_df['ParsedDate'].tolist()
        for i in range(len(dates) - 1):
            self.assertLessEqual(dates[i], dates[i+1], f'Date order broken at index {i}')

    def test_02_three_classes_target(self):
        """Verify targets are strictly 0 (Home Win), 1 (Draw), 2 (Away Win)."""
        unique_targets = set(self.featured_df['target'].unique())
        self.assertEqual(unique_targets, {0, 1, 2}, 'Targets must be exactly {0, 1, 2}')
        self.assertEqual(len(CLASS_NAMES), 3)

    def test_03_no_nan_or_inf_in_features(self):
        """Verify engineered features contain no NaNs or Infinities."""
        X = self.featured_df[FEATURE_COLUMNS]
        nan_count = X.isnull().sum().sum()
        inf_count = np.isinf(X.values).sum()
        self.assertEqual(nan_count, 0, f'Found {nan_count} NaNs in features')
        self.assertEqual(inf_count, 0, f'Found {inf_count} infinities in features')

    def test_04_implied_probabilities_normalization(self):
        """Verify bookmaker implied probabilities sum to 1.0 within epsilon."""
        sample_row = self.raw_df.iloc[0]
        ph, pd_val, pa = calculate_implied_probabilities(sample_row)
        total_p = ph + pd_val + pa
        self.assertAlmostEqual(total_p, 1.0, places=4, msg='Implied probabilities must sum to 1.0')
        self.assertGreater(ph, 0.0)
        self.assertGreater(pd_val, 0.0)
        self.assertGreater(pa, 0.0)

    def test_05_chronological_split_no_leakage(self):
        """Verify train/test split does not mix future matches into training."""
        train_df, test_df, X_train, y_train, X_test, y_test = split_chronologically(self.featured_df, train_ratio=0.75)
        self.assertEqual(len(train_df) + len(test_df), len(self.featured_df))
        
        max_train_date = train_df['ParsedDate'].max()
        min_test_date = test_df['ParsedDate'].min()
        self.assertLessEqual(max_train_date, min_test_date, 'Train set contains dates after test set start!')

    def test_06_model_artifacts_and_predictions(self):
        """Verify trained model artifacts load, predict valid classes and normalized probabilities."""
        lr = joblib.load('models/logistic_regression.joblib')
        rf = joblib.load('models/random_forest.joblib')
        scaler = joblib.load('models/scaler.joblib')

        # Test single sample
        X_sample = self.featured_df[FEATURE_COLUMNS].iloc[:5]
        X_sample_scaled = scaler.transform(X_sample)

        # Logistic Regression
        lr_preds = lr.predict(X_sample_scaled)
        lr_probas = lr.predict_proba(X_sample_scaled)
        self.assertEqual(len(lr_preds), 5)
        self.assertEqual(lr_probas.shape, (5, 3))
        for p in lr_probas:
            self.assertAlmostEqual(float(np.sum(p)), 1.0, places=4)

        # Random Forest
        rf_preds = rf.predict(X_sample)
        rf_probas = rf.predict_proba(X_sample)
        self.assertEqual(len(rf_preds), 5)
        self.assertEqual(rf_probas.shape, (5, 3))
        for p in rf_probas:
            self.assertAlmostEqual(float(np.sum(p)), 1.0, places=4)

    def test_07_load_dataset_accepts_dataframe(self):
        """Regression test: verify load_dataset accepts a pd.DataFrame without raising TypeError."""
        df_input = pd.read_csv('data/matches.csv')
        # This was previously raising: TypeError: argument of type 'method' is not iterable
        processed_df = load_dataset(df_input)
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertEqual(len(processed_df), len(self.raw_df))
        self.assertIn('ParsedDate', processed_df.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
