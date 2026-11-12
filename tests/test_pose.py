import numpy as np
import pytest

from badminton.pose import PoseEstimator


class _Fail(Exception):
    pass


def _estimator_with_detector(det):
    est = object.__new__(PoseEstimator)  # skip model download / session creation
    est.det = det
    return est


def test_empty_nms_on_coreml_means_no_people():
    def det(img):
        raise _Fail("Non-zero status code returned while running CoreML_1 node. Input (_1464) has a "
                    "dynamic shape ({-1}) but the runtime shape ({0}) has zero elements. "
                    "This is not supported by the CoreML EP.")

    boxes = _estimator_with_detector(det)._detect(np.zeros((8, 8, 3), np.uint8))
    assert boxes.shape == (0, 4)


def test_other_detector_errors_are_not_swallowed():
    def det(img):
        raise _Fail("CoreML: something else broke")

    with pytest.raises(_Fail):
        _estimator_with_detector(det)._detect(np.zeros((8, 8, 3), np.uint8))
