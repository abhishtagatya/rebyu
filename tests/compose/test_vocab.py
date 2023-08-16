import pytest
from typing import Any

from rebyu.compose.vocab import counter_vocab
from rebyu.compose.vocab import counter_character_vocab
from rebyu.compose.vocab import set_character_vocab
from rebyu.compose.vocab import nltk_vocab

from collections import Counter
from nltk.lm import Vocabulary


@pytest.mark.parametrize('series,cutoff,expected', [
    (['i am eating a burger'.split()], 2, Counter()),
    (['i i i i'.split(), 'i i i i'.split()], 2, Counter(i=8)),
    (['i a i a b c'.split()], 2, Counter(i=2, a=2)),
    (['bbb ccc ddd'.split()], 1, Counter(bbb=1, ccc=1, ddd=1))
])
def test_counter_vocab(series: Any, cutoff: Any, expected: Any):
    result = counter_vocab(series, cutoff)
    assert isinstance(result, Counter)
    assert result == expected


@pytest.mark.parametrize('series,cutoff,expected', [
    (['i am eating a burger'.split()], 2, Counter({'a': 3, 'i': 2, 'e': 2, 'g': 2, 'r': 2})),
    (['i i i i'.split(), 'i i i i'.split()], 0, Counter({'i': 8})),
    (['i a i a b c'.split()], 2, Counter({'i': 2, 'a': 2})),
    (['bbb ccc ddd'.split()], 1, Counter({'b': 3, 'c': 3, 'd': 3}))
])
def test_counter_character_vocab(series: Any, cutoff: Any, expected: Any):
    result = counter_character_vocab(series, cutoff)
    assert isinstance(result, Counter)
    assert result == expected


@pytest.mark.parametrize('series,expected', [
    (['i am eating a burger'.split()], set('iameatingaburger')),
    (['i i i i'.split(), 'i i i i'.split()], set('i')),
    (['i a i a b c'.split()], set('iaiabc')),
    (['bbb ccc ddd'.split()], set('bbbcccddd'))
])
def test_set_character_vocab(series: Any, expected: Any):
    result = set_character_vocab(series)
    assert isinstance(result, set)
    assert result == expected


@pytest.mark.parametrize('series,cutoff,expected', [
    (['i am eating a sandwich'.split()], 2, Vocabulary(counts='i am eating a sandwich'.split())),
    (['i i i i'.split(), 'i i i i'.split()], 2, Vocabulary(counts=Counter(i=8))),
    (['i a i a b c'.split()], 2, Vocabulary(counts=Counter(i=2, a=2, b=1, c=1))),
    (['bbb ccc ddd'.split()], 1, Vocabulary(Counter(bbb=1, ccc=1, ddd=1)))
])
def test_nltk_vocab(series: Any, cutoff: Any, expected: Any):
    result = nltk_vocab(series, cutoff)
    assert isinstance(result, Vocabulary)
    assert result.counts == expected.counts
