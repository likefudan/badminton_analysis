import numpy as np

from badminton.coreml_native import NativePose


class FakeModel:
    def __init__(self, batch, calls):
        self.batch, self.calls = batch, calls

    def predict(self, feed):
        x = feed["input"]
        assert x.shape[0] == self.batch
        self.calls.append(self.batch)
        # encode each crop's first pixel into the output so the order can be checked
        sx = np.zeros((self.batch, 17, 8), np.float32)
        sx[:, :, 0] = x[:, 0, 0, 0][:, None]
        return {"simcc_x": sx, "simcc_y": np.zeros((self.batch, 17, 8), np.float32)}


class FakeRTMPose:
    model_input_size = (4, 4)

    def preprocess(self, image, bbox):
        return np.full((4, 4, 3), bbox[0], np.float32), np.zeros(2), np.ones(2)

    def postprocess(self, outputs, center, scale):
        sx, _ = outputs
        return sx[:, :, 0][..., None].repeat(2, -1), np.ones((1, 17))


def make(calls):
    p = object.__new__(NativePose)
    p.rtmpose = FakeRTMPose()
    p.model_input_size = (4, 4)
    p._models = {b: FakeModel(b, calls) for b in (1, 2, 4)}
    return p


def test_people_batched_into_smallest_model_in_order():
    for n, expected in [(1, [1]), (2, [2]), (3, [4]), (5, [4, 1]), (6, [4, 2])]:
        calls = []
        kpts, scores = make(calls)(np.zeros((10, 10, 3)), [[i, 0, 1, 1] for i in range(n)])
        assert calls == expected
        assert kpts.shape == (n, 17, 2)
        np.testing.assert_array_equal(kpts[:, 0, 0], np.arange(n))
