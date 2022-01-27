'''
    Titanic test 2021
'''

# %% Imports
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import statistics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.xgboost
from sklearn.preprocessing import MinMaxScaler

# %% Load data
train = pd.read_csv(r"CSV\train.csv")
test = pd.read_csv(r"CSV\test.csv")

# %% Create X DataFrame
X = pd.DataFrame()


# %% Label Encode Sex
X['Sex'] = train['Sex'].map({'female': 1, 'male': 0}).astype(int)

# %% Sibsp, Parch and FamilySize
X["Sibsp"] = train["SibSp"]
X["Parch"] = train["Parch"]
X["FamilySize"] = train["SibSp"] + train["Parch"] + 1
X["Alone"] = [1 if element == 1 else 0 for element in X["FamilySize"]]

# %% Fare
X["Fare"] = train["Fare"]


# %% Copy class
X["Pclass"] = train["Pclass"]

# %% Copy Age
X["Age"] = train["Age"]


# %% Extract titles
X['Title'] = 0
X['Title'] = train.Name.str.extract(
    '([A-Za-z]+)\.')
X['Title'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col',
                   'Rev', 'Capt', 'Sir', 'Don'], ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'], inplace=True)

# %% Fill nulls in Age according to titles
X.loc[(X.Age.isnull()) & (X.Title == 'Mr'),
      'Age'] = X.Age[X.Title == "Mr"].mean()
X.loc[(X.Age.isnull()) & (X.Title == 'Mrs'),
      'Age'] = X.Age[X.Title == "Mrs"].mean()
X.loc[(X.Age.isnull()) & (X.Title == 'Master'),
      'Age'] = X.Age[X.Title == "Master"].mean()
X.loc[(X.Age.isnull()) & (X.Title == 'Miss'),
      'Age'] = X.Age[X.Title == "Miss"].mean()
X.loc[(X.Age.isnull()) & (X.Title == 'Other'),
      'Age'] = X.Age[X.Title == "Other"].mean()

# %% Encode titles
le = LabelEncoder()
X["Title"] = le.fit_transform(X["Title"])

# %% Fix nulls in Embarked
X['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().iloc[0])

# %% Encode Embarked
le = LabelEncoder()
X["Embarked"] = le.fit_transform(train["Embarked"])

# %% Fill nulls in Fare
X['Fare'] = X['Fare'].fillna(X['Fare'].mean())

# %% Set age groups
bins = [0, 4, 12, 20, 60, 100]
labels = [0, 1, 2, 3, 4]
X["AgeGroup"] = pd.cut((train["Age"]),
                       bins=bins, labels=labels, right=False)
X["AgeGroup"] = X["AgeGroup"].cat.codes


# %% Create y DataFrame and define answers
y = pd.DataFrame()
y["Survived"] = train["Survived"]


# %% Normalize Age and Fare columns

norm = pd.DataFrame()

norm["Age"] = X.pop("Age")
norm["Fare"] = X.pop("Fare")

column_names = norm.columns
numpy_data = norm.values
numpy_data = MinMaxScaler((0, 1)).fit_transform(numpy_data)
norm = pd.DataFrame(numpy_data)
norm.columns = column_names

X["Age"] = norm["Age"]
X["Fare"] = norm["Fare"]


# %% SMOTE resample
X, y = SMOTE().fit_resample(X, y)

# %% Print the results head
print(X.head(10))

# %% Split data for validation
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)


# %% Create XGB model
mlflow.start_run()

epochs = 1000
lr = 0.01
n_jobs = 6

params = {
    "n_estimators": epochs,
    "learning_rate": lr,
    "n_jobs": n_jobs
}

model = XGBClassifier(n_estimators=epochs,
                      learning_rate=lr,
                      n_jobs=n_jobs, eval_metric="logloss",
                      use_label_encoder=False)

model.fit(train_X, train_y,
          early_stopping_rounds=10,
          eval_set=[(test_X, test_y)],
          verbose=True)

predictions = model.predict(test_X)
predictions.round()
print("Predictions ready.")

# %% Cross validation mean score

acc = statistics.mean(cross_val_score(model, X, y, cv=10))
loss = log_loss(predictions, test_y)
print("Accuracy: {:f} Loss: {:f}".format(acc, loss))


# %%
mlflow.log_metrics({"accuracy": acc, "loss": loss,
                   "best_iteration": model.best_iteration})
mlflow.log_params(params)
mlflow.xgboost.log_model(model, artifact_path="model")

# %%
mlflow.end_run()


# %%
print("Done")

# %%
