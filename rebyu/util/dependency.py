from typing import List

import nltk


def nltk_dependency_mgt(required: List[str] = None):
    """ Check and Fulfill required dependencies for NLTK

    :param required: List of String (packages)
    :return:
    """
    if required is None:
        return

    for resource in required:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)
    return
