import numpy as np
import pytest

from badminton.pose import provider_config
from badminton.profiling import StageTimer, pipeline
from test_pipeline import FakeEstimator, clip  # noqa: F401  (fixture)


def test_coreml_settings():
    assert provider_config("coreml")[0][1] == {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    assert provider_config("coreml:CPUAndNeuralEngine:NeuralNetwork")[0][1] == {
        "ModelFormat": "NeuralNetwork", "MLComputeUnits": "CPUAndNeuralEngine"}
    with pytest.raises(ValueError):
        provider_config("coreml:GPUOnly")


def test_stage_timer_wrap():
    class Tool:
        def preprocess(self, x):
            return x + 1

    t, timer = Tool(), StageTimer()
    timer.wrap(t, "preprocess", "pre")
    assert t.preprocess(1) == 2 and t.preprocess(2) == 3
    (name, per_frame, per_call, calls), = timer.table(frames=2)
    assert name == "pre" and calls == 2


def test_pipeline_breakdown(clip, camera):
    video, calib = clip
    timer, n, wall = pipeline(video, FakeEstimator(camera), calib, frames=10)
    stages = {r[0] for r in timer.table(n)}
    assert n == 10 and wall > 0
    assert {"读帧解码", "挑选球员", "画骨架和球场线", "写入编码"} <= stages
