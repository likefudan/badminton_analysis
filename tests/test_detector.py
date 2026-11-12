import numpy as np

from badminton.detector import nms, person_boxes, raw_output_names


def test_nms_keeps_best_of_overlapping_and_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10.5]], float)
    scores = np.array([0.8, 0.9, 0.7, 0.5])
    assert nms(boxes, scores, 0.65).tolist() == [1, 2]


def test_nms_iou_threshold_is_inclusive_like_onnx():
    # IoU exactly 0.5: kept at thr 0.5 (suppress only when IoU > thr)
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], float)
    assert len(nms(boxes, np.array([0.9, 0.8]), 0.5)) == 2
    assert len(nms(boxes, np.array([0.9, 0.8]), 0.49)) == 1


def test_person_boxes_thresholds_rescales_and_caps():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 600, (300, 2))
    boxes = np.hstack([xy, xy + 20])
    scores = rng.uniform(0, 1, 300)
    out = person_boxes(boxes, scores, ratio=0.5, score_thr=0.3, max_det=10)
    assert len(out) == 10
    best = boxes[np.argmax(scores)] / 0.5
    np.testing.assert_allclose(out[0], best)
    assert person_boxes(boxes, scores * 0, 1.0).shape == (0, 4)


def test_raw_outputs_found_through_gathers():
    from onnx import TensorProto, helper

    vi = lambda n, s: helper.make_tensor_value_info(n, TensorProto.FLOAT, s)  # noqa: E731
    nodes = [
        helper.make_node("Identity", ["x"], ["boxes"]),
        helper.make_node("Identity", ["x"], ["scores"]),
        helper.make_node("Gather", ["boxes", "idx"], ["b2"], axis=1),
        helper.make_node("Gather", ["scores", "idx"], ["s2"], axis=1),
        helper.make_node("Transpose", ["s2"], ["s3"], perm=[0, 2, 1]),
        helper.make_node("NonMaxSuppression", ["b2", "s3"], ["sel"]),
    ]
    g = helper.make_graph(nodes, "g", [vi("x", [1, 8, 4])], [helper.make_tensor_value_info("sel", TensorProto.INT64, None)],
                          [helper.make_tensor("idx", TensorProto.INT64, [2], [0, 1])])
    assert raw_output_names(helper.make_model(g)) == ("boxes", "scores")
