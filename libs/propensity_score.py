import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

from typing import List


class PropensityScoreMatcher:
    def __init__(self, df: pd.DataFrame):
        # Initialize variables for future use
        self.df = df.copy()
        self.pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('logreg', LogisticRegression(random_state=45, class_weight='balanced'))
        ])

    def transform(self, features: List[str], target: str):
        self.df = self.df.sort_values(by='label', ascending=False).rename_axis('original_idx').reset_index()
        X = self.df.loc[:, features]
        y = self.df[target]
        return X, y

    def compute_propensity_score(self, X: pd.DataFrame, y: pd.Series, kfolds: int):

        # Initialize predictions as a Pandas Series
        propensity_score = pd.Series([None] * self.df.shape[0])

        # Stratified K-Fold cross-validation
        kfold = StratifiedKFold(kfolds)
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            # Fit the pipeline and predict probabilities
            self.pipe.fit(X_train, y_train)
            y_pred = self.pipe.predict_proba(X_test)
            propensity_score.loc[test_idx] = y_pred[:, 1]
        
        return propensity_score

    def find_best_control(self, outcome: pd.Series, propensity_score: pd.Series, n_control: int=5):
        # Compute caliper as 25% of the standard deviation of propensity_score
        caliper = np.std(propensity_score) * 0.25

        # Fit nearest neighbors model using the propensity scores
        knn = NearestNeighbors(n_neighbors=n_control*4, radius=caliper)
        knn.fit(propensity_score.values.reshape(-1, 1))

        # Get distances and neighbor indices
        distances, neighbor_indexes = knn.kneighbors(propensity_score.values.reshape(-1, 1))

        # Initialize control matrix and variables
        N = outcome.astype(int).sum()
        neigh_idx = neighbor_indexes[:N]
        neigh_idx = np.where(neigh_idx < N, -1, neigh_idx)
        already_used = []
        control_obs = np.zeros((N, n_control))

        # Find the best control observations
        for i in tqdm(range(neigh_idx.shape[0])):
            j = 0
            k = 0
            while (j < n_control) and (k < neigh_idx.shape[1]):
                if (neigh_idx[i, k] > 0) and (neigh_idx[i, k] not in already_used):
                    control_obs[i, j] = neigh_idx[i, k]
                    already_used.append(neigh_idx[i, k])
                    j += 1
                k += 1
        
        return control_obs
    
    def run(self, features: List[str], target: str, outcome: str, n_control: int, kfolds: int=5):
        X, y = self.transform(features, target)
        prop_score = self.compute_propensity_score(X, y, kfolds)
        self.df['prop_score'] = prop_score
        control_obs = self.find_best_control(self.df.loc[:, outcome], prop_score, n_control)

        df_control = self.df.loc[control_obs.flatten()]
        df_treatment = self.df.query(f"{outcome}==1")
        matched_df = pd.concat([df_control, df_treatment], ignore_index=True)
        self.df = self.df.set_index('original_idx').rename_axis('').sort_index()
        return self.df, matched_df
