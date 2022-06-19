# %% Analyze data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def print_report(actual, predicted):
    print(classification_report(actual, predicted))


def print_confusion_matrix(actual, predicted):
    print(confusion_matrix(actual, predicted))
