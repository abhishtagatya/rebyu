Overview
========

Rebyu packs quite a complete collection of analysis using different methods and modules.
The goal is primarily to simplify the process of Data Analysis and Feature Engineering using
state-of-the-art modules.

.. autoclass:: rebyu.Rebyu
   :members:
   :inherited-members:

Rebyu is the main class that encapsulates all the process of analysis and establishing an operation pipeline.

Mainly, we would focus on the three attributes within this class.

* ``data (pd.DataFrame)``: The data is stored as a Pandas DataFrame, every preprocessing operation will be appended to this DataFrame
* ``composition (Dict)``: The composition dictionary stores multiple information about the composition of the data (i.e Vocabulary, POS Tags, NER, etc.)
* ``analysis (Dict)``: The analysis dictionary stores multiple results of analysis operated on the DataFrame (i.e Sentiment, Emotion, Topic Modelling, Embedding, etc.)

.. _Pipeline:

Get Started with Pipelines
--------------------------

.. autoclass:: rebyu.pipeline.RebyuPipeline
   :members:
   :inherited-members:

Pipelines in Rebyu are a set of Steps (Functions) that operate on the data.

There are three types of Step, mainly: ``PREPROCESSING``, ``COMPOSE``, and ``ANALYZE``.

These three categories of step operate and store on the data differently.

Below is some of the prebuilt pipelines Rebyu has.

Blank Pipeline
~~~~~~~~~~~~~~

This Pipeline is Blank (default).


NLTK Pipeline
~~~~~~~~~~~~~

The NLTK Pipeline provides a pipeline to analyze the data using NLTK-based functions.

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import NLTK_PIPELINE

   rb = Rebyu(data='twitter.csv', pipeline=NLTK_PIPELINE)
   rb.run(verbose=True)

   print(rb.info())


Content:

#. :ref:`PREP_CAST_NAN` - Cast non-string data to empty string (text, text)
#. :ref:`PREP_NLTK_TOKENIZE` - Tokenize into NLTK Tokens (text, tokens)
#. :ref:`COMPOSE_NLTK_VOCAB` - Compose an NLTK Vocabulary (tokens, vocab)
#. :ref:`COMPOSE_COUNTER_CHARVOCAB` - Compose a Counter Character Vocabulary (tokens, char_vocab)
#. :ref:`COMPOSE_NLTK_POS_TAG` - Extract Part-of-Speech Tags using NLTK (tokens, pos_tags)
#. :ref:`COMPOSE_NLTK_NER` - Extract Named-Entities using NLTK (tokens, ner)
#. :ref:`ANALYZE_VADER_POLARITY` - Predict the Polarity using NLTK Vader Sentiment (text, vader_polarity)


TextBlob Pipeline
~~~~~~~~~~~~~~~~~

The TextBlob Pipeline provides a pipeline to analyze the data using TextBlob-based object.

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import TEXTBLOB_PIPELINE

   rb = Rebyu(data='twitter.csv', pipeline=TEXTBLOB_PIPELINE)
   rb.run(verbose=True)

   print(rb.info())


Content:

#. :ref:`PREP_CAST_NAN` - Cast non-string data to empty string (text, text)
#. :ref:`PREP_TEXTBLOB` - Transform into TextBlob Object (text, text)
#. :ref:`PREP_TEXTBLOB_TOKENIZE` - Tokenize into TextBlob WordList (text, tokens)
#. :ref:`COMPOSE_COUNTER_VOCAB` - Compose an Counter Word Vocabulary (tokens, vocab)
#. :ref:`COMPOSE_COUNTER_CHARVOCAB` - Compose a Counter Character Vocabulary (tokens, char_vocab)
#. :ref:`ANALYZE_TEXTBLOB_POLARITY` - Predict the Polarity using TextBlob (text, textblob_polarity)

