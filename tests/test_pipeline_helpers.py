"""Tests for pipeline helpers that need no model or video."""

import numpy as np

from crowd_heatmap.heatmap import HeatmapAccumulator
from crowd_heatmap.pipeline import extract_detections, save_heatmap_image


class _Tensor:
    """Minimal stand-in for a torch tensor."""

    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def int(self):
        return _Tensor(self.data.astype(int))


class _Boxes:
    """Stand-in for ultralytics Boxes."""

    def __init__(self, xyxy, ids, conf):
        self.xyxy = _Tensor(xyxy)
        self.id = None if ids is None else _Tensor(ids)
        self.conf = _Tensor(conf)
        self._n = len(xyxy)

    def __len__(self):
        return self._n


class _Result:
    """Stand-in for ultralytics Results."""

    def __init__(self, boxes):
        self.boxes = boxes


def test_extract_detections_converts_to_plain_python():
    boxes = _Boxes([[10.9, 20.2, 50.5, 90.7]], [3.0], [0.8])
    det = extract_detections(_Result(boxes))
    assert det[0][:2] == ((10, 20, 50, 90), 3)
    assert isinstance(det[0][1], int) and abs(det[0][2] - 0.8) < 1e-6


def test_extract_detections_without_ids_or_boxes():
    assert extract_detections(_Result(None)) == []
    assert extract_detections(_Result(_Boxes([[0, 0, 1, 1]], None, [0.5]))) == []
    assert extract_detections(_Result(_Boxes(np.zeros((0, 4)), [], []))) == []


def test_save_heatmap_image_writes_png(tmp_path):
    heat = HeatmapAccumulator(30, 20, radius=2, blur=3)
    heat.update([(10, 10)])
    target = tmp_path / "nested" / "h.png"
    assert save_heatmap_image(heat, str(target)) is True
    assert target.stat().st_size > 0


def test_save_heatmap_image_bad_extension_does_not_raise(tmp_path):
    heat = HeatmapAccumulator(30, 20, radius=2, blur=3)
    assert save_heatmap_image(heat, str(tmp_path / "heat.unknownext")) is False
