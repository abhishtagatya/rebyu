import pytest
from typing import Any

from rebyu.preprocess.transform import cast_nan_str
from rebyu.preprocess.transform import cast_case
from rebyu.preprocess.transform import sub_replace
from rebyu.preprocess.transform import expand_contractions
from rebyu.preprocess.transform import censor_username
from rebyu.preprocess.transform import censor_urls


@pytest.mark.parametrize('text,expected', [
    (None, ''),
    (12131, ''),
    ('Bicycle', 'Bicycle'),
    (object, ''),
])
def test_cast_nan_str(text: Any, expected: Any):
    result = cast_nan_str(text)
    assert result == expected


@pytest.mark.parametrize('text,case,expected', [
    ('I JUST ATE A SANDWICH', 'LOWER', 'i just ate a sandwich'),
    ('i just ate a sandwich', 'UPPER', 'I JUST ATE A SANDWICH'),
    ('I JUST ATE A SANDWICH', 'lower', 'i just ate a sandwich'),
    ('i just ate a sandwich', 'TITLE', 'i just ate a sandwich'),
    ('I JUST ATE A SANDWICH'.split(), 'LOWER', 'i just ate a sandwich'.split()),
    ('i just ate a sandwich'.split(), 'UPPER', 'I JUST ATE A SANDWICH'.split()),
    ('i just ate a Sandwich'.split(), 'TITLE', 'i just ate a Sandwich'.split()),
])
def test_cast_case(text: Any, case: Any, expected: Any):
    result = cast_case(text, case)
    assert result == expected


@pytest.mark.parametrize('text,sub,rep,expected', [
    ('I JUST ATE A BURGER', '', '', 'I JUST ATE A BURGER'),
    ('i just ate a sandwich', 'sandwich', 'BURGER', 'i just ate a BURGER'),
    ('I JUST ATE A SANDWICH', 'JUST', 'WANT TO', 'I WANT TO ATE A SANDWICH'),
    ('i just ate a sandwich', 'just ate', 'am eating', 'i am eating a sandwich'),
    ('I JUST ATE A BURGER'.split(), '', '', 'I JUST ATE A BURGER'.split()),
    ('i just ate a sandwich'.split(), 'sandwich', 'BURGER', 'i just ate a BURGER'.split()),
    ('I JUST ATE A SANDWICH'.split(), 'JUST', 'WANT TO', ['I', 'WANT TO', 'ATE', 'A', 'SANDWICH']),
    ('i just ate a sandwich'.split(), 'just', 'am eating', ['i', 'am eating', 'ate', 'a', 'sandwich']),
])
def test_sub_replace(text: Any, sub: Any, rep: Any, expected: Any):
    result = sub_replace(text, sub, rep)
    assert result == expected


@pytest.mark.parametrize('text,expected', [
    ("you're an animal", 'you are an animal'),
    ('ima head out yall', 'i am about to head out you all'),
    ('yall gotta go', 'you all got to go'),
    (["you're", "the", "real", "deal"], ["you", "are", "the", "real", "deal"]),
    (["ima", "be", "famous", "yall"], ["i", "am", "about", "to", "be", "famous", "you", "all"]),
])
def test_expand_contractions(text: Any, expected: Any):
    result = expand_contractions(text)
    assert result == expected


@pytest.mark.parametrize('text,censor,expected', [
    ("@mclovin i wish i had gone with mohammed", '@user', '@user i wish i had gone with mohammed'),
    ('what the @dasdiajd2das is this', '@puck', 'what the @puck is this'),
    ('@friend1 @friend2 we should start a podcast', '@user', '@user @user we should start a podcast'),
    ("@mclovin i wish i had gone with mohammed".split(), '@user', '@user i wish i had gone with mohammed'.split()),
    ('what the @dasdiajd2das is this'.split(), '@puck', 'what the @puck is this'.split()),
    ('@friend1 @friend2 we should start a podcast'.split(), '@user', '@user @user we should start a podcast'.split()),
])
def test_censor_username(text: Any, censor: Any, expected: Any):
    result = censor_username(text, censor)
    assert result == expected


@pytest.mark.parametrize('text,censor,expected', [
    ("have you tried x.com?", 'http', 'have you tried x.com?'),
    ('the website https://reddit.com looks fun', 'https', 'the website https looks fun'),
    ('https://reddit.com and https://digg.com', 'url', 'url and url'),
    ("have you tried x.com?".split(), 'http', 'have you tried x.com?'.split()),
    ('the website https://reddit.com looks fun'.split(), 'https', 'the website https looks fun'.split()),
    ('https://reddit.com and https://digg.com'.split(), 'url', 'url and url'.split()),
])
def test_censor_urls(text: Any, censor: Any, expected: Any):
    result = censor_urls(text, censor)
    assert result == expected
