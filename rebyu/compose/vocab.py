from typing import Any
from collections import Counter

from rebyu.util.dependency import nltk_dependency_mgt

from nltk.lm import Vocabulary


def counter_vocab(tokens: Any, cutoff: int = 2):
    """ Create a Vocabulary using the Counter class

    :param tokens: List of Tokens
    :param cutoff: Minimum occurrence to enter the Vocab
    :return: collections.Counter
    """
    vocab = Counter()
    for token in tokens:
        vocab.update(token)

    for word in list(vocab):
        if vocab[word] < cutoff:
            del vocab[word]

    return vocab


def counter_character_vocab(tokens: Any, cutoff: int = 0):
    """ Create a Character Vocabulary using the Counter class

    :param tokens: List of Tokens
    :param cutoff: Minimum occurrence to enter the Vocab
    :return: collections.Counter
    """
    vocab = Counter()
    for token in tokens:
        for char in token:
            vocab.update(char)

    for char in list(vocab):
        if vocab[char] < cutoff:
            del vocab[char]

    return vocab


def set_character_vocab(tokens: Any, sort: bool = False):
    """ Create a Character Vocabulary using a Set

    :param tokens: List of Tokens
    :param sort: Determines whether the set should be sorted
    :return: set
    """
    vocab = set()
    for token in tokens:
        for char in token:
            vocab.update(set(char))

    if sort:
        return sorted(vocab)
    return vocab


def nltk_vocab(tokens: Any, cutoff: int = 2):
    """ Create a Vocabulary using NLTK Vocabulary class

    :param tokens: List of Tokens
    :param cutoff: Minimum occurrence to enter the vocab
    :return: nltk.lm.Vocabulary
    """
    nltk_dependency_mgt(required=['punkt'])

    tokenized_soup = [x for token in tokens for x in token]
    return Vocabulary(tokenized_soup, unk_cutoff=cutoff)
