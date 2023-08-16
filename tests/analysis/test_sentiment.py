import pytest
from typing import Any

from rebyu.analysis.sentiment import textblob_polarity
from rebyu.analysis.sentiment import vader_polarity

from textblob import TextBlob


@pytest.mark.parametrize('series', [
    (['i just had the best coffee']),
    ([TextBlob('i just had the worst coffee')]),
    (['0123910', TextBlob('3123131')]),
    (['', '', '', ' ', ' ', ' ', '']),
    (['no opinions', TextBlob('i have some to say')])
])
def test_textblob_polarity(series: Any):
    result = textblob_polarity(series)
    assert len(result) == len(series)

    assert all(type(x) == float for x in result)
    assert all(-1 <= x <= 1 for x in result)


@pytest.mark.parametrize('series', [
    (['i just had the best coffee']),
    ([TextBlob('i just had the worst coffee')]),
    (['0123910', TextBlob('3123131')]),
    (['', '', '', ' ', ' ', ' ', '']),
    (['no opinions', TextBlob('i have some to say')])
])
def test_textblob_polarity(series: Any):
    result = vader_polarity(series)
    assert len(result) == len(series)

    assert all(type(x) == dict for x in result)
    assert all('neg' in x for x in result)
    assert all('neu' in x for x in result)
    assert all('pos' in x for x in result)
    assert all('compound' in x for x in result)

    assert all(-1 <= x['neg'] <= 1 for x in result)
    assert all(-1 <= x['neu'] <= 1 for x in result)
    assert all(-1 <= x['pos'] <= 1 for x in result)
    assert all(-1 <= x['compound'] <= 1 for x in result)
