import pytest
from typing import Any

from rebyu.preprocess.scale import linear_map
from rebyu.preprocess.scale import polarity_threshold_map


@pytest.mark.parametrize('value,category,expected', [
    (0, 2, 1),
    (0.5, 2, 1),
    (-0.5, 2, 0),
    (1, 2, 1),
    (-1, 2, 0),
    (0.5, 3, 2),
    (-0.5, 3, 0),
    (0.1, 3, 1),
    (0.9, 5, 4),
    (0.3, 4, 2)
])
def test_linear_map_unlabelled(value: Any, category: Any, expected: Any):
    result = linear_map(value, n=category)
    assert result == expected


@pytest.mark.parametrize('value,category,label,expected', [
    (0, 2, {0: 'Zero', 1: 'One'}, 'One'),
    (0.5, 2, {0: 'Zero', 1: 'One'}, 'One'),
    (-0.5, 2, {0: 'Zero', 1: 'One'}, 'Zero'),
    (1, 2, {0: 'Zero', 1: 'One'}, 'One'),
    (-1, 2, {0: 'Zero', 1: 'One'}, 'Zero'),
    (0.5, 3, {0: 'Zero', 1: 'One', 2: 'Two'}, 'Two'),
    (-0.5, 3, {0: 'Zero', 1: 'One', 2: 'Two'}, 'Zero'),
    (0.1, 3, {0: 'Zero', 1: 'One', 2: 'Two'}, 'One'),
    (0.9, 5, {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four'}, 'Four'),
    (0.3, 4, {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three'}, 'Two')
])
def test_linear_map_unlabelled(value: Any, category: Any, label: Any, expected: Any):
    result = linear_map(value, n=category, label=label)
    assert result == expected


@pytest.mark.parametrize('value,expected', [
    (0, 1),
    (0.05, 1),
    (-0.05, 1),
    (0.9, 2),
    (0.7, 2),
    (-0.9, 0),
    (-0.7, 0)
])
def test_polarity_threshold_map_unlabelled(value: Any, expected: Any):
    result = polarity_threshold_map(value, label=None)
    assert result == expected


@pytest.mark.parametrize('value,expected', [
    (0, 'Neutral'),
    (0.05, 'Neutral'),
    (-0.05, 'Neutral'),
    (0.9, 'Positive'),
    (0.7, 'Positive'),
    (-0.9, 'Negative'),
    (-0.7, 'Negative')
])
def test_polarity_threshold_map_labelled(value: Any, expected: Any):
    result = polarity_threshold_map(value)
    assert result == expected
