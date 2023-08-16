import pytest
from typing import Any

from rebyu.compose.ner import nltk_extract_ner


def nltk_ner_list():
    return [
        'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'TIME', 'MONEY',
        'PERCENT', 'FACILITY', 'GPE'  # GPE stands for Geo-Political Entity
    ]


@pytest.mark.parametrize('series', [
    (['I had lunch with Barrack Obama'.split(), 'I had lunch on the Eiffel Tower'.split()]),
    (['I just bought some Apple stock'.split(), 'I just sold my Amazon stock'.split()]),
    (['I went on vacation to Paris, France'.split()]),
    (['The total was $18'.split()]),
])
def test_nltk_extract_ner_tokens(series: Any):
    result = nltk_extract_ner(series)
    assert len(result) == len(series)

    comb_series = [x for y in series for x in y]
    entities = [x[0] for r in result for x in r]
    all(x in comb_series for x in entities)

    ners = [x[1] for r in result for x in r]
    all(x in nltk_ner_list() for x in ners)


@pytest.mark.parametrize('series', [
    (['I had lunch with Barrack Obama', 'I had lunch on the Eiffel Tower']),
    (['I just bought some Apple stock', 'I just sold my Amazon stock']),
    (['I went on vacation to Paris, France']),
    (['The total was $18']),
])
def test_nltk_extract_ner_text(series: Any):
    result = nltk_extract_ner(series)
    assert len(result) == len(series)

    comb_series = [x for y in series for x in y.split()]
    entities = [x[0] for r in result for x in r]
    all(x in comb_series for x in entities)

    ners = [x[1] for r in result for x in r]
    all(x in nltk_ner_list() for x in ners)