import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import os


sns.set_style("darkgrid")
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)


def rescale(data):
    """
    doesn't make mean == 0, Values between 0 and 1
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    :param data:
    :return:
    """
    x = data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_norm = pd.DataFrame(x_scaled, columns=['filesize', 'duration', 'total_comments', 'total_plays', 'total_likes'])
    df_norm.describe()
    return df_norm


def correlation(df):
    """
    Correlation Heatmap using Pearson Correlation Coefficient
    :param df: Pandas DataFrame Object
    :return: Pandas DataFrame Object
    """
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr,
                mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()
    return True


def pair(df):
    """
    Pair plot for the columns in the Dataframe
    :param df: Pandas DataFrame Object
    :return: Pandas DataFrame Object
    """
    sns.pairplot(df)
    plt.show()
    return True


def analyze():
    """
    Main function for the analysis
    :return: Control, if the plots are rendered
    """
    data = pd.read_csv(os.getcwd().split('/exploratory_analysis') +'/data/similar-staff-picks-challenge-clips.csv')
    data = data[['filesize', 'duration', 'total_comments', 'total_plays', 'total_likes']]
    data = rescale(data)
    cov = data.cov()
    correlation(data)
    pair(data)
