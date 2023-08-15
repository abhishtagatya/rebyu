import pytest

from rebyu.pipeline.base import BasePipeline
from rebyu.pipeline.pipeline import RebyuPipeline


@pytest.fixture
def rebyu_pipeline():
    return RebyuPipeline(
        pid='blank',
        steps=[]
    )


def test_rebyu_pipeline_instance(rebyu_pipeline):
    assert isinstance(rebyu_pipeline, BasePipeline)