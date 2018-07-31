from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessor as p
import pandas as pd
import os
import nltk, string
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.metrics import pairwise
from nltk.tree import Tree
import numpy as np
nltk.download('wordnet')


def lemmatizer(tokens):
    """
    Full morphological analysis to accurately identify the lemma for each word
    Capture more information about the language than a porter
    :param tokens: List of token identified from the document
    :return:
    """
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]


def ie_preprocess(document):
    """
    Extracting Named Entities from the Document and Generating token lists
    :param documents: String
    :return: List of Tokens with Named Entity Mapping
    """
    sentences = nltk.sent_tokenize(document)  # sentence segmentation
    tokens = []
    for each in sentences:
        each = p.clean(each)
        token_list = get_chunks(each)
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


def normalize(text):
    """
    Normalize the documents
    :param text: string
    :return: tokenized vocabulary dictionary
    """
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return lemmatizer(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def get_tf_idf(documents):
    """
    Wt,d = TFt,d log (N/DFt)
    Ability to apply L1 and L2 norm at this step.
    :param documents: list of string text
    :return: tf-idf matrix
    """
    tfidf_vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    return tfidf


def cosine_similarity(tfidf):
    """
    Function to compute cosine similarity
    :param tfidf: matrix of term frequency - inverse document frequency
    :return: dot product of the tfidf matrix
    """
    # cosine_similarities = (tfidf * tfidf.T).toarray()
    # Pair - wise cosine similaries produces the similarity between same matric if second matrix is not specified
    cosine_similarities = pairwise.cosine_similarity(tfidf)
    return cosine_similarities


def compute_pairwise_cosine_similarity(df):
    """
    Cosine Similarity for each pair of document with another.
    :return: pairwise cosine similarity matrix.
    """
    df = df.fillna('')
    df['text'] = df.apply(lambda x: str(x['title']) + ' ' + str(x['caption']) + ' ' + str(x['category_names']),
                          axis=1)
    tfidf = get_tf_idf(df['text'])
    return cosine_similarity(tfidf)


def save_champion_lists_with_ner():
    """
    Compute the pairwise cosine similarity, and save the champion lists for each result
    :return: Control, if the process executes correctly
    """
    df = pd.read_csv(os.getcwd().split('/model')[0] + '/data/similar-staff-picks-challenge-clips_cleaned.csv')
    pairwise_cosine_similarity_matrix = compute_pairwise_cosine_similarity(df)
    result = {}
    id = 0
    for each in pairwise_cosine_similarity_matrix:
        result[id] = [np.argsort(each)[-11:-2]]
        id += 1
    result_df = pd.DataFrame.from_dict(result, orient='index', columns=["similar_clips"])
    result_df['clip_id'] = df['id']
    result_df.to_csv(os.getcwd().split('/model')[0] + '/data/similar-clips-cosine_with_ner.csv')


save_champion_lists_with_ner()
