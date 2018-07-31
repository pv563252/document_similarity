"""
Helper Function for Exploratory Analysis
"""

import nltk
from nltk.corpus import stopwords
from nltk.tree import Tree
from nltk import word_tokenize, pos_tag, ne_chunk
import preprocessor as p
import pandas as pd
import re, os

stop_words = set(stopwords.words('english'))


def ie_preprocess(document):
    """
    Extracting Named Entities from the Document and Generating token lists
    :param documents: String
    :return: List of Tokens with Named Entity Mapping
    """
    document = ' '.join(re.findall(r'\b\w+\b', str(document)))
    sentences = nltk.sent_tokenize(document)  # sentence segmentation
    tokens = []
    for each in sentences:
        each = p.clean(each)
        token_list = get_chunks(each)
        token_list = [t.lower().strip() for t in token_list if t not in stopwords.words('english')]
        token_list = [t for t in token_list if t.strip is not ""]
        tokens.extend(token_list)
    return tokens


def get_chunks(text):
    """
    Get Chunks as single token after Named Entity Resolution
    :param text: String
    :return: List of Tokens
    """
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continuous_chunk.append(i[0])
    return continuous_chunk


def text_extraction():
    """
    Extract Named Entities from the text
    :return: Control if the process executes correctly
    """
    data = pd.read_csv(os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-clips_translation.csv').fillna('')
    data['text'] = data.apply(lambda x: str(x['title']) + ' ' + str(x['caption_en']) + ' ' + str(x['category_names']), axis=1)
    data['text_formatted'] = data['text'].apply(lambda x: ie_preprocess(x))
    data.to_csv(os.getcwd().split('/preprocessing')[0] + '/data/similar-staff-picks-challenge-clips_preprocessed.csv')
