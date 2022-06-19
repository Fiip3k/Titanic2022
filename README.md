# Titanic2022
Another try at Kaggle Titanic Challenge

Files description:

- main.py - file to launch
- data_loader.py - file to handle importing Titanic data
- data_cleaner.py - handle data to be readable by xgboost + add a couple features
- data_preprocessor.py - scaling + future augmentation
- classifier.py - xgboost classifier with grid search
- analyzer.py - analyze data (SMOTE needed?)

TODO
- apply MLFlow, SMOTE from old
- feature selection
- perhaps pipelines if the code gets dirty
