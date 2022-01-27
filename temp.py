# %% Imports
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import statistics
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# %% Load data
train = pd.read_csv(r"CSV\clean_train_nick.csv")
test = pd.read_csv(r"CSV\clean_test_nick.csv")

X = train.copy()
y = train.pop("Survived")

le = LabelEncoder()
X["Title"] = le.fit_transform(X["Title"])

# %% bleh TODO
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# %% Create XGB model

model = XGBClassifier(n_estimators=1000, learning_rate=0.01,
                      n_jobs=6, eval_metric="logloss", use_label_encoder=False)

print("Fitting the model.")
model.fit(train_X, train_y,
          early_stopping_rounds=10,
          eval_set=[(test_X, test_y)],
          verbose=True)

predictions = model.predict(test_X)
predictions.round()
print("Predictions ready.")

# %% Cross validation mean score

scores = cross_val_score(model, X, y, cv=10)

statistics.mean(scores)


# %%
