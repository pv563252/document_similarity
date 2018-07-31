"""
Helper Function for Modeling
"""


from utils import vimeo_api_utils
import pandas as pd
import os


def get_data():
    """
    Pre-processing function to get the cleaned dataframe with the category information
    :return: Pandas DataFrame Object
    """
    path = os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-clips.csv'
    clips = pd.read_csv(path)
    clips = pd.merge(clips, add_categories(), on='clip_id', how='left')
    clips.to_csv(os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-clips_cleaned.csv')
    return clips


def add_categories():
    """
    Map category ids to category text
    :return: Pandas DataFrame Object
    """
    path = os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-clip-categories.csv'
    clip_categories = pd.read_csv(path)
    path = os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-categories.csv'
    category_mapping = pd.read_csv(path)
    # categories['tags'] = categories.apply(lambda x: get_tags(x['clip_id']), axis=1)
    clip_categories['category_names'] = clip_categories.apply(lambda x: get_categories(x['categories'],
                                                                                       category_mapping), axis=1)
    return clip_categories


def get_tags(clip_id):
    """
    Add Additional Metadata to the Clip Data - Incomplete since Vimeo API has rate limits
    :param clip_id: Integer Clip Id of the
    :return: List of Tags
    """
    vimeo_api_utils.VimeoAPI().get_video_tags(clip_id)
    pass


def get_categories(category_list, mapping):
    """
    Map category IDs to category names.
    :param category_list: Integer List of Categories
    :param mapping: Mapping DataFrame
    :return: String of Category Names
    """
    category_list = [int(x.strip()) for x in category_list.split(',')]
    categories = mapping[mapping['category_id'].isin(category_list)]['name']
    return ",".join(categories)
