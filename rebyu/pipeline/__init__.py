from rebyu.pipeline.base import BaseStep, BasePipeline
from rebyu.pipeline.step import (
    RebyuStep,
    PREP_CAST_NAN,
    PREP_CAST_CASE,
    PREP_REMOVE_NUMBERS,
    PREP_REMOVE_PUNCTUATIONS,
    PREP_REMOVE_WHITESPACES,
    PREP_REMOVE_SPECIFICS,
    PREP_REMOVE_STOPWORDS,
    PREP_REPLACE_WORD,
    PREP_CENSOR_USERNAME,
    PREP_CENSOR_URLS,
    PREP_EXPAND_CONTRACTIONS,
    PREP_SENTENCE_LENGTH,
    PREP_WORD_COUNT,
    COMPOSE_COUNTER_VOCAB,
    COMPOSE_COUNTER_CHARVOCAB,
    COMPOSE_SET_CHARVOCAB,

    # TextBlob
    PREP_TEXTBLOB,
    PREP_TEXTBLOB_TOKENIZE,
    PREP_TEXTBLOB_SENTENCES,
    ANALYZE_TEXTBLOB_POLARITY,

    # NLTK
    PREP_NLTK_TOKENIZE,
    COMPOSE_NLTK_VOCAB,
    ANALYZE_VADER_POLARITY,

    # Transformers
    PREP_TRANSFORMERS_TOKENIZE,
    ANALYZE_TRANSFORMERS_MODEL,
    ANALYZE_TRANSFORMERS_PIPELINE,

    ANALYZE_CARDIFF_NLP_SENTIMENT,
    ANALYZE_CARDIFF_NLP_EMOTION
)
from rebyu.pipeline.pipeline import RebyuPipeline

BLANK_PIPELINE = RebyuPipeline(
    pid='blank-pipeline',
    steps=[]
)

TEST_PIPELINE = RebyuPipeline(
    pid='standard-pipeline',
    steps=[
        PREP_CAST_NAN,
        PREP_CAST_CASE,
        PREP_EXPAND_CONTRACTIONS,
        PREP_REMOVE_NUMBERS,
        # PREP_REMOVE_PUNCTUATIONS,
        PREP_REMOVE_WHITESPACES,
        # PREP_REMOVE_SPECIFICS,
        # PREP_REMOVE_STOPWORDS,
        # PREP_WORDNET_LEMMA,
        PREP_REPLACE_WORD.add_args(sub='murder', rep='kill'),
        PREP_NLTK_TOKENIZE,
        PREP_SENTENCE_LENGTH,
        PREP_WORD_COUNT,
        COMPOSE_COUNTER_VOCAB,
        COMPOSE_SET_CHARVOCAB
    ]
)

TEXTBLOB_PIPELINE = RebyuPipeline(
    pid='textblob-pipeline',
    steps=[
        PREP_CAST_NAN,
        PREP_TEXTBLOB,
        PREP_TEXTBLOB_TOKENIZE,
        PREP_SENTENCE_LENGTH,
        COMPOSE_COUNTER_VOCAB,
        COMPOSE_COUNTER_CHARVOCAB,
        ANALYZE_TEXTBLOB_POLARITY
    ]
)

VADER_PIPELINE = RebyuPipeline(
    pid='vader-pipeline',
    steps=[
        PREP_CAST_NAN,
        PREP_NLTK_TOKENIZE,
        COMPOSE_COUNTER_VOCAB,
        COMPOSE_COUNTER_CHARVOCAB,
        ANALYZE_VADER_POLARITY
    ]
)

CARDIFF_NLP_PIPELINE = RebyuPipeline(
    pid='cardiff-nlp',
    steps=[
        PREP_CAST_NAN,
        PREP_CENSOR_USERNAME,
        PREP_CENSOR_URLS,
        ANALYZE_CARDIFF_NLP_SENTIMENT,
        ANALYZE_CARDIFF_NLP_EMOTION
    ]
)
