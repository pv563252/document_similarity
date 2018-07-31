from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessor as p
import pandas as pd
import os
import nltk, string
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tree import Tree
nltk.download('wordnet')


def LemTokens(tokens):
    """

    :param tokens:
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


def LemNormalize(text):
    """

    :param text:
    :return:
    """
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(ie_preprocess(text.lower().translate(remove_punct_dict)))


def get_tf_idf(documents):
    """

    :param documents:
    :return:
    """
    tfidf_vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    print(type(tfidf))
    return tfidf


def cosine_similarity(tfidf):
    """

    :param tfidf:
    :return:
    """
    cosine_similarities = (tfidf * tfidf.T).toarray()
    print(cosine_similarities)


def compute_pairwise_cosine_similarity_with_ner():
    """

    :return:
    """
    df = pd.read_csv(os.getcwd().split('/model')[0] + '/data/similar-staff-picks-challenge-clips-cleaned.csv')
    df = df.fillna(0)
    df['text'] = df.apply(lambda x: str(x['title']) + ' ' + str(x['caption']) + ' ' + str(x['category_names']),
                          axis=1)
    tfidf = get_tf_idf(df['text'])
    cosine_similarity(tfidf)


compute_pairwise_cosine_similarity_with_ner()
