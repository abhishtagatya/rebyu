import pytest
from typing import Any

from rebyu.preprocess.remove import trim_text
from rebyu.preprocess.remove import remove_numbers
from rebyu.preprocess.remove import remove_urls
from rebyu.preprocess.remove import remove_punctuations
from rebyu.preprocess.remove import remove_whitespaces
from rebyu.preprocess.remove import remove_specifics
from rebyu.preprocess.remove import remove_emojis
from rebyu.preprocess.remove import remove_stopwords


@pytest.mark.parametrize('text,length,expected', [
    ('18a18' * 200, 200, ('18a18' * 200)[:200]),
    ('trim' * 4, 4, 'trim'),
    ('', 1, ''),
    ('a'*20, 80, 'a'*20),
    ('abc'*10, 2, 'ab')
])
def test_remove_numbers(text: Any, length: Any, expected: Any):
    result = trim_text(text, length=length)
    assert len(result) == length
    assert result == expected


@pytest.mark.parametrize('text,expected', [
    ('18a18', 'a'),
    ('I was born in 1989', 'I was born in '),
    ('1923892138198391', ''),
    ('I just ate eight sandwich', 'I just ate eight sandwich')
])
def test_remove_numbers(text: Any, expected: Any):
    result = remove_numbers(text)
    assert result == expected


@pytest.mark.parametrize('text,expected', [
    ('https://google.com', ''),
    ('I just checked google', 'I just checked google'),
    ('Did you go to www.google.com', 'Did you go to '),
    ('Check out this picture pic.twitter.com/sjdiaadid', 'Check out this picture ')
])
def test_remove_urls(text: Any, expected: Any):
    result = remove_urls(text)
    assert result == expected


@pytest.mark.parametrize('text,expected', [
    ('https://google.com', 'httpsgooglecom'),
    ('I\'d like to call you again. Please!', 'Id like to call you again Please'),
    ('Did you go to www.google.com', 'Did you go to wwwgooglecom'),
    ('Never gonna give you up', 'Never gonna give you up')
])
def test_remove_punctuations(text: Any, expected: Any):
    result = remove_punctuations(text)
    assert result == expected


@pytest.mark.parametrize('text,expected', [
    ('         ', ''),
    (' Never  gonna let you down    ', 'Never  gonna let you down'),
    ('A a      b B', 'A a      b B'),
    ('Wingardium Leviosa    ', 'Wingardium Leviosa')
])
def test_remove_whitespaces(text: Any, expected: Any):
    result = remove_whitespaces(text)
    assert result == expected


@pytest.mark.parametrize('text,sub,expected', [
    ('', '', ''),
    ('Never  gonna let you down', 'Never', '  gonna let you down'),
    ('Lovevol', 'Evol', 'Lovevol'),
    ('Joseph Schwartz', 'Joseph', ' Schwartz')
])
def test_remove_specifics(text: Any, sub: Any, expected: Any):
    result = remove_specifics(text, sub)
    assert result == expected


@pytest.mark.parametrize('text,expected', [
    ('❤️❤️❤️❤️love', 'love'),
    ('good👍👍job', 'goodjob'),
    ('hello, world', 'hello, world'),
    ('𝚊𝚙𝚙𝚕𝚒𝚌𝚊𝚝𝚒𝚘𝚗', '𝚊𝚙𝚙𝚕𝚒𝚌𝚊𝚝𝚒𝚘𝚗'),
    ('𝕤𝕔𝕣𝕒𝕥𝕔𝕙✔️👊', '𝕤𝕔𝕣𝕒𝕥𝕔𝕙')
])
def test_remove_emojis(text: Any, expected: Any):
    result = remove_emojis(text)
    assert result == expected


@pytest.mark.parametrize('text,extra,expected', [
    ('I just ate a sandwich', None, 'ate sandwich'),
    ('Never gonna let you down', ['let'], 'Never gon na'),  # NLTK.word_tokenize behavior
    ('Spatula', ['spatula'], ''),
    ('Enraged Monster', None, 'Enraged Monster')
])
def test_remove_stopwords(text: Any, extra: Any, expected: Any):
    result = remove_stopwords(text, extra)
    assert result == expected
