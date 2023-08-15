import pandas as pd
import pytest
from typing import Any, List

from rebyu.pipeline.base import BaseStep
from rebyu.pipeline.base import BasePipeline


def func_a(a: int, b: int = 0, c: int = 0):
    return a + b + c


def func_b(a: int, b: int = 0, c: int = 0):
    return a - b - c


def func_c(series: Any, **kwargs):
    return sum(series) + sum(kwargs.values())


@pytest.fixture
def step_object():
    return BaseStep(
        sid='abc-xyz',
        stype=BaseStep.STEP_PREPROCESS,
        source='num',
        target='num',
        func=func_a,
        func_args={'b': 10}
    )


def test_base_step_sid(step_object):
    assert step_object.sid == 'abc-xyz'


def test_base_step_set_sid(step_object):
    step_object.set_sid(sid='xyz-abc')
    assert step_object.sid == 'xyz-abc'


def test_base_step_stype(step_object):
    assert step_object.stype == BaseStep.STEP_PREPROCESS


def test_base_step_set_stype(step_object):
    step_object.set_stype(stype=BaseStep.STEP_COMPOSE)
    assert step_object.stype == BaseStep.STEP_COMPOSE


def test_base_step_source(step_object):
    assert step_object.source == 'num'


def test_base_step_set_source(step_object):
    step_object.set_source(source='tokens')
    assert step_object.source == 'tokens'


def test_base_step_target(step_object):
    assert step_object.target == 'num'


def test_base_step_set_target(step_object):
    step_object.set_target(target='tokens')
    assert step_object.target == 'tokens'


def test_base_step_func(step_object):
    assert step_object.func == func_a


def test_base_step_set_func(step_object):
    step_object.set_func(func=func_b)
    assert step_object.func == func_b


def test_base_step_func_args(step_object):
    assert step_object.func_args == {'b': 10}


def test_base_step_add_func_args(step_object):
    step_object.add_args(b=12, c=5)
    assert step_object.func_args == {'b': 12, 'c': 5}


def test_base_step_run_preprocess(step_object):
    step_object.set_stype(stype=BaseStep.STEP_PREPROCESS)
    step_object.set_func(func=func_a)
    step_object.add_args(b=10, c=10)

    inp = pd.DataFrame([1, 2, 3], columns=['num'])
    exp = pd.DataFrame([21, 22, 23], columns=['num'])

    step_object.run(inp, {}, {})
    assert inp.equals(exp)


def test_base_step_run_compose(step_object):
    step_object.set_stype(stype=BaseStep.STEP_COMPOSE)
    step_object.set_func(func=func_c)

    data = pd.DataFrame([1, 2, 3], columns=['num'])
    inp = {}
    exp = {'num': 16}

    step_object.run(data, inp, {})
    assert inp == exp


def test_base_step_run_analysis(step_object):
    step_object.set_stype(stype=BaseStep.STEP_ANALYZE)
    step_object.set_func(func=func_c)

    data = pd.DataFrame([3, 3, 3], columns=['num'])
    inp = {}
    exp = {'num': 19}

    step_object.run(data, {}, inp)
    assert inp == exp


def test_base_step_copy(step_object):
    copy_obj = step_object.copy()
    assert step_object.sid == copy_obj.sid
    assert step_object.stype == copy_obj.stype
    assert step_object.source == copy_obj.source
    assert step_object.target == copy_obj.target
    assert step_object.func == copy_obj.func
    assert step_object.func_args == copy_obj.func_args


@pytest.fixture
def pipeline_object():
    return BasePipeline(
        pid='abc-pipeline',
        steps=[
            BaseStep(
                sid='abc',
                stype=BaseStep.STEP_PREPROCESS,
                source='num',
                target='num',
                func=func_a
            ),
            BaseStep(
                sid='def',
                stype=BaseStep.STEP_COMPOSE,
                source='num',
                target='sum',
                func=func_c
            ),
            BaseStep(
                sid='ghi',
                stype=BaseStep.STEP_ANALYZE,
                source='num',
                target='sum',
                func=func_c
            )
        ]
    )


def test_base_pipeline_pid(pipeline_object):
    assert pipeline_object.pid == 'abc-pipeline'


def test_base_pipeline_head(pipeline_object):
    assert pipeline_object.head is not None


def test_base_pipeline_tail(pipeline_object):
    assert pipeline_object.tail is not None


def test_base_pipeline_curr_idx(pipeline_object):
    assert pipeline_object.curr_idx == 0


def test_base_pipeline_curr(pipeline_object):
    assert pipeline_object.curr == pipeline_object.head


def test_base_pipeline_length(pipeline_object):
    assert pipeline_object.length == 3


def test_base_pipeline_add(pipeline_object):
    new_step = BaseStep(
        sid='jkl',
        stype=BaseStep.STEP_ANALYZE,
        source='num',
        target='sum',
        func=func_c
    )
    pipeline_object.add(new_step)

    assert pipeline_object.length == 4
    assert pipeline_object.tail.sid == new_step.sid
    assert pipeline_object.tail.func == new_step.func
    assert pipeline_object.tail.next is None


def test_base_pipeline_add_function(pipeline_object):
    pipeline_object.add_function(
        func=func_c,
        stype=BaseStep.STEP_ANALYZE,
        source='num',
        target='sum'
    )

    assert pipeline_object.length == 4
    assert pipeline_object.tail.sid == func_c.__name__
    assert pipeline_object.tail.func == func_c
    assert pipeline_object.tail.next is None


def test_base_pipeline_pop(pipeline_object):
    first_pop = pipeline_object.pop()

    assert first_pop is not None
    assert pipeline_object.length == 2
    assert pipeline_object.tail != first_pop

    second_pop = pipeline_object.pop()

    assert second_pop is not None
    assert pipeline_object.length == 1
    assert pipeline_object.tail != second_pop

    assert pipeline_object.head == pipeline_object.tail

    third_pop = pipeline_object.pop()

    assert third_pop is not None
    assert pipeline_object.length == 0
    assert pipeline_object.tail != third_pop

    assert pipeline_object.head == pipeline_object.tail

    fourth_pop = pipeline_object.pop()

    assert fourth_pop is None


def test_base_pipeline_steps_info(pipeline_object):
    steps_info = pipeline_object.steps_info()

    assert len(steps_info) == len(pipeline_object)


def test_base_pipeline_state(pipeline_object):
    data = pd.DataFrame([1, 2, 3], columns=['num'])
    comp = {}
    analysis = {}

    idx, curr = pipeline_object.state()
    assert idx == 0
    assert curr == pipeline_object.head

    next_idx = 1
    next_curr = pipeline_object.head.next

    pipeline_object.step(data, comp, analysis)

    idx, curr = pipeline_object.state()
    assert idx == next_idx
    assert curr == next_curr


def test_base_pipeline_state_and_reset(pipeline_object):
    data = pd.DataFrame([1, 2, 3], columns=['num'])
    comp = {}
    analysis = {}

    idx, curr = pipeline_object.state()
    assert idx == 0
    assert curr == pipeline_object.head

    next_idx = 1
    next_curr = pipeline_object.head.next

    pipeline_object.step(data, comp, analysis)

    idx, curr = pipeline_object.state()
    assert idx == next_idx
    assert curr == next_curr

    pipeline_object.reset()
    idx, curr = pipeline_object.state()
    assert idx == 0
    assert curr == pipeline_object.head


def test_base_pipeline_step(pipeline_object):
    data = pd.DataFrame([1, 2, 3], columns=['num'])
    exp_data = pd.DataFrame([1, 2, 3], columns=['num'])

    comp = {}
    exp_comp = {'sum': 6}

    analysis = {}
    exp_analysis = {'sum': 6}

    first_step = pipeline_object.step(data, comp, analysis)

    assert first_step is True
    assert data.equals(exp_data)

    second_step = pipeline_object.step(data, comp, analysis)

    assert second_step is True
    assert comp == exp_comp

    third_step = pipeline_object.step(data, comp, analysis)

    assert third_step is True
    assert analysis == exp_analysis

    fourth_step = pipeline_object.step(data, comp, analysis)

    assert fourth_step is False


def test_base_pipeline_run(pipeline_object):
    data = pd.DataFrame([4, 5, 6], columns=['num'])
    exp_data = pd.DataFrame([4, 5, 6], columns=['num'])

    comp = {}
    exp_comp = {'sum': 15}

    analysis = {}
    exp_analysis = {'sum': 15}

    pipeline_object.run(data, comp, analysis)

    assert data.equals(exp_data)
    assert comp == exp_comp
    assert analysis == exp_analysis


