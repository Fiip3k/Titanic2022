# %% Preprocess data
from mlxtend.preprocessing import minmax_scaling
import pandas as pd


def scale(df: pd.DataFrame):
    result = minmax_scaling(df, columns=list(df.columns))
    return result

# %%
