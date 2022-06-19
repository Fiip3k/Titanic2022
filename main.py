# %%
import data_loader
import data_cleaner
import data_preprocessor
import classifier
import analyzer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import statistics


train, test = data_loader.load_data()

train = data_cleaner.add_features(train)
train = data_cleaner.missing_values(train)
train = data_cleaner.categorical(train)
train = data_cleaner.drop_unused(train)

# TODO Data augmentation
# TODO Feature selection

train = data_preprocessor.scale(train)

X, y = data_loader.split_data(train, "Survived")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = classifier.fit_grid_search(X_train, y_train)
y_pred = classifier.get_predictions(model, X_test)

analyzer.print_report(y_test, y_pred)
print("Average score: ", statistics.mean(
    cross_val_score(model, X_train, y_train, cv=10)))
