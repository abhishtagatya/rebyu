Step
========

Each step in a pipeline is a class-enclosed function that takes a source and a target to operate upon.

.. autoclass:: rebyu.pipeline.RebyuStep
   :members:
   :inherited-members:

RebyuStep has three categories:

* ``PREPROCESS``: Mainly used for preprocessing. Operates and stores data on ``Rebyu.data`` column.
* ``COMPOSE``: Mainly used for extracting metadata. Operates on ``Rebyu.data`` and stores it into ``Rebyu.composition``.
* ``ANALYZE``: Mainly used for analysis. Operates on ``Rebyu.data`` and stores it into ``Rebyu.analysis``.


Available Steps
---------------

.. _PREP_CAST_NAN:

Cast NAN
~~~~~~~~

Cast non-string objects into empty strings. ``source: text``, ``target: text`` (Rebyu.data)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import PREP_CAST_NAN

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)

   print(rb.info())

.. autofunction:: rebyu.preprocess.transform.cast_nan_str

.. _PREP_TEXTBLOB:

Transform TextBlob
~~~~~~~~~~~~~~~~~~

Transform Text into a TextBlob Object. ``source: text``, ``target: text`` (Rebyu.data)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import PREP_TEXTBLOB

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_TEXTBLOB)

   print(rb.info())

.. autofunction:: rebyu.preprocess.transform.to_textblob

.. _PREP_NLTK_TOKENIZE:

Tokenize using NLTK
~~~~~~~~~~~~~~~~~~~

Tokenize Text into Tokens using NLTK. ``source: text``, ``target: tokens`` (Rebyu.data)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import PREP_NLTK_TOKENIZE

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_NLTK_TOKENIZE)

   print(rb.info())

.. autofunction:: rebyu.preprocess.transform.nltk_tokenize

.. _PREP_TEXTBLOB_TOKENIZE:

Tokenize using TextBlob
~~~~~~~~~~~~~~~~~~~~~~~

Tokenize Text into Tokens using TextBlob. ``source: text``, ``target: tokens`` (Rebyu.data)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import PREP_TEXTBLOB_TOKENIZE

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)

   print(rb.info())

.. autofunction:: rebyu.preprocess.transform.textblob_tokenize

.. _COMPOSE_COUNTER_VOCAB:

Compose Counter Vocab
~~~~~~~~~~~~~~~~~~

Create a Counter Word Vocabulary. ``source: tokens``, ``target: vocab`` (Rebyu.composition)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import COMPOSE_COUNTER_VOCAB

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)
   rb.add(PREP_NLTK_TOKENIZE) # Better to Tokenize First
   rb.add(COMPOSE_COUNTER_VOCAB)

   print(rb.info())

.. autofunction:: rebyu.compose.vocab.counter_vocab

.. _COMPOSE_NLTK_VOCAB:

Compose NLTK Vocab
~~~~~~~~~~~~~~~~~~

Create an NLTK Word Vocabulary. ``source: tokens``, ``target: vocab`` (Rebyu.composition)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import COMPOSE_NLTK_VOCAB

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)
   rb.add(PREP_NLTK_TOKENIZE) # Better to Tokenize First
   rb.add(COMPOSE_NLTK_VOCAB)

   print(rb.info())

.. autofunction:: rebyu.compose.vocab.nltk_vocab

.. _COMPOSE_COUNTER_CHARVOCAB:

Compose Counter Character Vocab
~~~~~~~~~~~~~~~~~~

Create a Counter Character Vocabulary. ``source: tokens``, ``target: char_vocab`` (Rebyu.composition)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import COMPOSE_COUNTER_CHARVOCAB

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)
   rb.add(PREP_NLTK_TOKENIZE) # Better to Tokenize First
   rb.add(COMPOSE_COUNTER_CHARVOCAB)

   print(rb.info())

.. autofunction:: rebyu.compose.vocab.counter_character_vocab

.. _COMPOSE_NLTK_POS_TAG:

Extract POS using NLTK
~~~~~~~~~~~~~~~~~~

Extract Part-of-Speech Tags from Tokens using NLTK. ``source: tokens``, ``target: pos_tags`` (Rebyu.composition)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import COMPOSE_NLTK_POS_TAG

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)
   rb.add(PREP_NLTK_TOKENIZE) # Better to Tokenize First
   rb.add(COMPOSE_NLTK_POS_TAG)

   print(rb.info())

.. autofunction:: rebyu.compose.pos.nltk_extract_pos_tags

.. _COMPOSE_NLTK_NER:

Extract Named Entity using NLTK
~~~~~~~~~~~~~~~~~~

Extract Named Entity from Tokens using NLTK NE Chunks. ``source: tokens``, ``target: ner`` (Rebyu.composition)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import COMPOSE_NLTK_NER

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)
   rb.add(PREP_NLTK_TOKENIZE) # Better to Tokenize First
   rb.add(COMPOSE_NLTK_NER)

   print(rb.info())

.. autofunction:: rebyu.compose.pos.nltk_extract_ner

.. _ANALYZE_TEXTBLOB_POLARITY:

Predict Polarity using TextBlob
~~~~~~~~~~~~~~~~~~

Predict Text Polarity using TextBlob. ``source: tokens``, ``target: textblob_polarity`` (Rebyu.analysis)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import ANALYZE_TEXTBLOB_POLARITY

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)
   rb.add(PREP_NLTK_TOKENIZE) # Better to Tokenize First
   rb.add(ANALYZE_TEXTBLOB_POLARITY)

   print(rb.info())

.. autofunction:: rebyu.analysis.sentiment.textblob_polarity

.. _ANALYZE_VADER_POLARITY:

Predict Polarity using NLTK
~~~~~~~~~~~~~~~~~~

Predict Text Polarity using NLTK. ``source: tokens``, ``target: vader_polarity`` (Rebyu.analysis)

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import ANALYZE_VADER_POLARITY

   rb = Rebyu(data='twitter.csv')
   rb.add(PREP_CAST_NAN)
   rb.add(PREP_NLTK_TOKENIZE) # Better to Tokenize First
   rb.add(ANALYZE_VADER_POLARITY)

   print(rb.info())

.. autofunction:: rebyu.analysis.sentiment.vader_polarity

