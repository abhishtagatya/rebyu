import pytest
from typing import Any

from rebyu.compose.pos import nltk_extract_pos_tags


def nltk_pos_tags():
    return [
        'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
        'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
        'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
        'WP$', 'WRB'
    ]


@pytest.mark.parametrize('series', [
    (['i am eating a burger'.split(), 'i am eating a sandwich'.split()]),
    (['i had had a dog'.split(), 'i had a cat'.split()]),
    (['i am eating a donut'.split()]),
    (['i watched a scary movie'.split()]),
])
def test_nltk_extract_pos_tags_tokens(series: Any):
    result = nltk_extract_pos_tags(series)
    assert len(result) == len(series)
    assert [x[0] for r in result for x in r] == [x for r in series for x in r]

    tags = [x[1] for r in result for x in r]
    all(x in nltk_pos_tags() for x in tags)


@pytest.mark.parametrize('series', [
    (['i am eating a burger', 'i am eating a sandwich']),
    (['i had had a dog', 'i had a cat']),
    (['i am eating a donut']),
    (['i watched a scary movie']),
])
def test_nltk_extract_pos_tags_text(series: Any):
    result = nltk_extract_pos_tags(series)
    assert len(result) == len(series)
    assert [x[0] for r in result for x in r] == [x for r in series for x in r.split()]

    tags = [x[1] for r in result for x in r]
    all(x in nltk_pos_tags() for x in tags)

