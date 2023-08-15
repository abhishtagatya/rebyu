import pytest

from rebyu.pipeline.base import BaseStep
from rebyu.pipeline.step import RebyuStep


@pytest.fixture
def rebyu_step():
    return RebyuStep(
        sid='abc-xyz',
        stype=RebyuStep.STEP_PREPROCESS,
        source='num',
        target='num',
        func=lambda x: x,
        func_args={'b': 10}
    )


def test_rebyu_step_instance(rebyu_step):
    assert isinstance(rebyu_step, BaseStep)