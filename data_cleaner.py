# %% Handle missing values and categorical columns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def missing_values(df: pd.DataFrame):  # Cabin 687, Embarked 2
    # Cabin 687
    df["Cabin"].fillna(0, inplace=True)

    # Embarked 2
    df["Embarked"].fillna("C", inplace=True)

    return df


def add_features(df: pd.DataFrame):
    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Alone
    df["Alone"] = [1 if element == 1 else 0 for element in df["FamilySize"]]

    # Titles
    df['Title'] = 0
    df['Title'] = df.Name.str.extract(
        '([A-Za-z]+)\.')
    df['Title'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col',
                         'Rev', 'Capt', 'Sir', 'Don'], ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'], inplace=True)

    # Age according to titles
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'),
           'Age'] = df.Age[df.Title == "Mr"].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'),
           'Age'] = df.Age[df.Title == "Mrs"].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'),
           'Age'] = df.Age[df.Title == "Master"].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'),
           'Age'] = df.Age[df.Title == "Miss"].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Other'),
           'Age'] = df.Age[df.Title == "Other"].mean()

    return df


def categorical(df: pd.DataFrame):
    # Name
    df.drop("Name", axis=1, inplace=True)

    # Sex
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # Ticket
    df.drop("Ticket", axis=1, inplace=True)

    # Cabin
    df.drop("Cabin", axis=1, inplace=True)

    # Embarked
    df.drop("Embarked", axis=1, inplace=True)

    # Title
    le = LabelEncoder()
    df["Title"] = le.fit_transform(df["Title"])

    return df


def drop_unused(df: pd.DataFrame):

   # df.drop("SibSp", axis=1, inplace=True)
   # df.drop("Parch", axis=1, inplace=True)
   # df.drop("Alone", axis=1, inplace=True)
    return df
