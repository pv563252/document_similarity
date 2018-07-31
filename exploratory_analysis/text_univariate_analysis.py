import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os


def data_category_analysis(df):
    """
    Summary analysis the categories and Data Points per category
    :param df: Pandas DataFrame Object
    :return: True
    """
    df = df.groupby(['category_names']).count()
    df.to_csv(os.getcwd().split('/exploratory_analysis')[0] + '/data/similar-staff-picks-challenge-clips_categories.csv')
    return True


def document_length_analysis(df, title):
    """
    Summary Analysis of the length of the documents
    :param df: Pandas DataFrame Object
    :param title: String Title of the Histogram
    :return: True
    """
    lens = [len(doc) for doc in list(df)]
    # Plot.
    plt.rc('figure', figsize=(8,6))
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=2)
    # Histogram.
    plt.hist(lens, bins=20)
    plt.hold(True)
    # Average length.
    avg_len = sum(lens) / float(len(lens))
    plt.axvline(avg_len, color='#e41a1c')
    plt.hold(False)
    plt.title(title)
    plt.xlabel('Length')
    plt.text(100, 800, 'mean = %.2f' % avg_len)
    plt.show()
    return True


def language_analysis(df):
    """
    Analyze if there is a presence of other languages in the text
    :param df: Pandas Dataframe Object
    :return: True
    """
    df = df.groupby(['caption_language']).count().reset_index()
    sns.barplot(x="id", y="caption_language", data=df)
    plt.show()
    return True


def univariate_analysis():
    """
    Analysis of the text data features available
    :return: True
    """
    df = pd.read_csv(os.getcwd().split('/exploratory_analysis')[0] +
                      '/data/similar-staff-picks-challenge-clips_translation.csv')
    language_analysis(df[["caption_language", "id"]].fillna('NA'))
    document_length_analysis(df['title'].fillna(''), 'Histogram of Title Lengths.')
    data_category_analysis(df['category_names', 'id'])
    return True
