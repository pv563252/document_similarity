"""
Helper Function for Exploratory Analysis
"""


from googletrans import Translator
import pandas as pd
import os

translator = Translator()


def clean_caption():
    """
    Pre-processing for the text analysis
    :return: Text translated to English
    """
    df = pd.read_csv(os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-clips_translation.csv').fillna('NA')
    df["caption_language"] = df.apply(lambda x: detect_caption(x['caption']), axis=1)
    df["caption_en"] = df.apply(lambda x: translate_caption(x['caption']), axis=1)
    df["title_language"] = df.apply(lambda x: detect_caption(x['title']), axis=1)
    df["title_en"] = df.apply(lambda x: translate_caption(x['title']), axis=1)
    df.to_csv(os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-clips_translation.csv')
    return df


def detect_caption(caption):
    """
    Detects the language of the text
    :param caption: string
    :return: Language of the string detected, "NA" if error
    """
    try:
        return translator.detect(caption).lang
    except:
        print(caption)
        return "NA"


def translate_caption(caption):
    """
    Converts converted text to english
    :param caption: string
    :return: String translated to English, string if error
    """
    try:
        return translator.translate(caption).text
    except:
        return caption
