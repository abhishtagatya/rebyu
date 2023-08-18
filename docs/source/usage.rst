Usage
=====

This documentation follows the basic user guide from installation to writing your first lines of code.

Installation
------------

To use Rebyu, first install it using pip:

.. code-block:: console

   (.venv) $ pip install -U rebyu


Requirements
------------

Since Rebyu uses a collection of libraries such to operate, below is the list of
requirements that comes with Rebyu.

* `Pandas <https://pandas.pydata.org>`_
* `NLTK <https://nltk.org>`_
* `TextBlob <https://textblob.readthedocs.io>`_
* `Transformers <https://huggingface.co/docs/transformers/index>`_

And a few operational requirements (util, logging, preprocessing, etc.)

* `Contractions <https://github.com/kootenpv/contractions>`_
* `Loguru <https://github.com/Delgan/loguru>`_


Quick Start
-----------

.. code-block:: python

   from rebyu import Rebyu
   from rebyu.pipeline import NLTK_PIPELINE

   rb = Rebyu(data='twitter.csv', pipeline=NLTK_PIPELINE)
   rb.run(verbose=True)

   print(rb.info())


The snippet above uses a prebuilt pipeline to do analysis using the `NLTK Library <https://nltk.org>`_.
The analysis consists of preprocessing, creating a vocabulary dictionary, and sentiment analysis all together.

To learn more about using :ref:`Pipeline`.


