from typing import List, Any, AnyStr

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob
from transformers import AutoTokenizer

import contractions


def cast_nan_str(text: Any):
    """ Cast non-string objects into empty strings

    :param text: Any string object
    :return: text
    """
    return text if type(text) is str else ''


def cast_case(text: Any, case: AnyStr = 'LOWER'):
    """ Cast the case into a string object (upper or lowercase)

    :param text: Any string object
    :param case: LOWER or UPPER
    :return: text
    """
    if case.upper() == 'LOWER':
        return text.lower()
    if case.upper() == 'UPPER':
        return text.upper()
    return text


def sub_replace(text: Any, sub: Any = None, rep: Any = None):
    """ Replace a substring with another string (sub -> rep)

    :param text: Any string object
    :param sub: Any string object
    :param rep: Any string object
    :return: text
    """
    if sub is rep:
        return text

    if type(text) is str:
        return text.replace(sub, rep)
    return [x.replace(sub, rep) for x in text]


def expand_contractions(text: Any):
    """ Expand the contractions within the text (you're -> you are)

    :param text: Any string object
    :return: text
    """
    expanded = []
    if type(text) is str:
        for token in nltk.word_tokenize(text):
            expanded.append(contractions.fix(token))
        return ' '.join(expanded)
    for token in text:
        expanded.append(contractions.fix(token))
    return expanded


def censor_username(text: Any, censor: Any = '@user'):
    """ Censor any username within the text with a placeholder

    :param text: Any string object
    :param censor: Any string object (placeholder)
    :return: text
    """
    new_text = []
    if type(text) is str:
        for token in nltk.word_tokenize(text):
            token = censor if token.startswith('@') and len(token) > 1 else token
            new_text.append(token)
        return ' '.join(new_text)

    for token in text:
        token = censor if token.startswith('@') and len(token) > 1 else token
        new_text.append(token)
    return new_text


def censor_urls(text: Any, censor: Any = 'http'):
    """ Censor any urls within the text with a placeholder

    :param text: Any string object
    :param censor: Any string object (placeholder)
    :return: text
    """
    new_text = []
    if type(text) is str:
        for token in nltk.word_tokenize(text):
            token = censor if token.startswith('http') else token
            new_text.append(token)
        return ' '.join(new_text)

    for token in text:
        token = censor if token.startswith('http') else token
        new_text.append(token)
    return new_text


# -- TextBlob -- #

def to_textblob(text: Any):
    """ Transform text into a TextBlob Object

    :param text: Any string object
    :return: TextBlob
    """
    return TextBlob(text)


def textblob_tokenize(text: Any):
    """ Tokenize text using TextBlob tokens

    :param text: Any string object
    :return: List of Strings
    """
    if isinstance(text, TextBlob):
        return text.tokens

    return TextBlob(text).tokens


def textblob_sentences(text: Any):
    """ Extract the sentences using TextBlob sentences

    :param text: Any string object
    :return: List of Strings
    """
    if isinstance(text, TextBlob):
        return text.sentences

    return TextBlob(text).sentences


# -- NLTK -- #

def nltk_tokenize(text: Any, language: AnyStr = 'english'):
    """ Tokenize text using NLTK word_tokenize

    :param text: Any string object
    :param language: The language of the text
    :return: List of String
    """
    if type(text) != str:
        return []
    return nltk.word_tokenize(
        text, language=language
    )


def nltk_porter_stem(text: Any):
    """ Stem the words from text using Porter method

    :param text: Any string object
    :return: text
    """
    stemmer = PorterStemmer()
    if type(text) is str:
        return ' '.join([stemmer.stem(x) for x in nltk.word_tokenize(text)])
    return [stemmer.stem(x) for x in text]


def nltk_lancaster_stem(text: Any):
    """ Stem the words from text using Lancaster method

    :param text: Any string object
    :return: text
    """
    stemmer = LancasterStemmer()
    if type(text) is str:
        return ' '.join([stemmer.stem(x) for x in nltk.word_tokenize(text)])
    return [stemmer.stem(x) for x in text]


def nltk_wordnet_lemma(text: Any):
    """ Lemmatize the words from text using WordNet

    :param text: Any string object
    :return: text
    """
    lemma = WordNetLemmatizer()
    if type(text) is str:
        return ' '.join([
            lemma.lemmatize(x, pos=_lemma_penn2morphy(tag))
            for x, tag in nltk.pos_tag(nltk.word_tokenize(text))
        ])
    return [
        lemma.lemmatize(x, pos=_lemma_penn2morphy(tag))
        for x, tag in nltk.pos_tag(text)
    ]


def _lemma_penn2morphy(penn_tag: AnyStr):
    try:
        morphy_tag = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
        return morphy_tag[penn_tag[:2]]
    except KeyError:
        return 'v'
    except IndexError:
        return 'v'


# -- Transformers -- #

def transformers_tokenize(
        text: Any,
        tokenizer: Any = AutoTokenizer,
        model_name: AnyStr = 'bert-base-uncased',
        returns: AnyStr = 'pt'):
    """ Tokenize text using transformers module

    :param text: Any string object
    :param tokenizer: The tokenizer model (from transformers)
    :param model_name: The pretrained model name
    :param returns: The return value (pt: PyTorch, tf: Tensorflow)
    :return: Tokenized Object
    """
    tokenizer = tokenizer.from_pretrained(model_name)
    return tokenizer(text, return_tensors=returns)
