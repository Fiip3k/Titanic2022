'''
    viewing data file
'''
# %% Imports
import pandas as pd
import seaborn as sns
import numpy as np

# %% Functions TODO Library


# %% Load data
train = pd.read_csv(r"CSV\train.csv")
test = pd.read_csv(r"CSV\test.csv")

# %% Correlation
X["Survived"] = train["Survived"]
corr = X.corr()

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(255, 0, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap)

# %%
