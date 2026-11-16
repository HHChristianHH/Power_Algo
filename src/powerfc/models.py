"""The forecasting models, from trivial to gradient boosting.

All of them share one tiny interface so the backtest can treat them alike:

    model.fit(train_table)        learn from past rows (features + price)
    model.predict(table)          return one price forecast per row

Start simple on purpose. A complex model is only worth anything if it beats
the naive ones, and by how much is the number an interviewer will ask for.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "price"


class NaiveModel:
    """Baseline: tomorrow's price equals the price `days` days ago, same hour.

    days=1 -> "same hour yesterday"; days=7 -> "same hour last week".
    Nothing is learned, so fit() does nothing.
    """

    def __init__(self, days: int):
        self.column = f"price_lag_{days}d"

    def fit(self, train: pd.DataFrame) -> "NaiveModel":
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return table[self.column].to_numpy()


class RidgePerHour:
    """Linear model: one separate regression for each of the 24 hours.

    Why one per hour? The same extra gigawatt of residual load moves the price
    far more at 18:00, when the system is tight, than at 03:00. Separate
    regressions let each hour have its own sensitivities.

    "Ridge" is ordinary least squares with a penalty on large coefficients.
    Many of our features are strongly correlated (the price lags, for
    example); the penalty stops the model from betting wildly on any one.
    RidgeCV picks the penalty strength by cross-validation on training data.
    """

    def __init__(self, features: list[str]):
        self.features = [f for f in features if f != "hour"]  # hour is handled by the split
        self.models = {}

    def fit(self, train: pd.DataFrame) -> "RidgePerHour":
        train = train.dropna(subset=[TARGET])
        for hour, rows in train.groupby("hour"):
            pipeline = make_pipeline(
                SimpleImputer(strategy="median"),   # fill the rare missing value
                StandardScaler(),                   # put all features on one scale
                RidgeCV(alphas=np.logspace(-1, 3, 9)),
            )
            self.models[hour] = pipeline.fit(rows[self.features], rows[TARGET])
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        forecast = pd.Series(np.nan, index=table.index)
        for hour, rows in table.groupby("hour"):
            forecast.loc[rows.index] = self.models[hour].predict(rows[self.features])
        return forecast.to_numpy()


class LightGBMModel:
    """Gradient-boosted decision trees.

    A decision tree splits the data with yes/no questions ("is residual load
    above 45 GW?"). Boosting builds hundreds of small trees in sequence, each
    correcting the errors of those before it. Unlike a linear model this
    captures the kinked shape of the supply curve (prices explode when the
    system gets tight, and collapse when renewables exceed demand) and
    interactions (the same wind matters more on a Sunday).

    The settings below are common defaults. They were NOT tuned on the test
    period: tuning on the data you later report on is another form of
    look-ahead.
    """

    def __init__(self, features: list[str]):
        self.features = features
        self.model = lgb.LGBMRegressor(
            n_estimators=400,        # number of trees
            learning_rate=0.05,      # how much each tree may correct
            num_leaves=31,           # size of each tree
            min_child_samples=20,    # at least 20 hours behind every leaf
            subsample=0.8,           # each tree sees 80 % of the rows ...
            subsample_freq=1,
            colsample_bytree=0.8,    # ... and 80 % of the features (less overfitting)
            random_state=0,          # reproducible results
            n_jobs=1,                # one CPU core: on data this small, coordinating
                                     # several cores costs more time than it saves
                                     # (measured: 0.5 s per fit vs 4 s with all cores)
            verbose=-1,
        )

    def fit(self, train: pd.DataFrame) -> "LightGBMModel":
        train = train.dropna(subset=[TARGET])
        self.model.fit(train[self.features], train[TARGET])
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.model.predict(table[self.features])

    def importance(self) -> pd.Series:
        """How much each feature reduced the training error, in total ("gain")."""
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.features)
