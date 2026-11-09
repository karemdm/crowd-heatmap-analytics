# Weights

The project uses Ultralytics YOLO26 detection models trained on COCO. Only class 0 (`person`) is tracked.

| File | Size | In this repo | Use |
|------|------|--------------|-----|
| `yolo26x.pt` | ~114 MB | No (see below) | Default (`--model yolo26x.pt`). Highest capacity, slowest. |
| `yolo26m.pt` | ~43 MB | Yes, `weights/yolo26m.pt` | Lighter option: `--model weights/yolo26m.pt`. |

## Getting `yolo26x.pt`

`yolo26x.pt` exceeds GitHub's 100 MB per-file limit, so it is not committed. Two options:

1. **Automatic**: the default `--model yolo26x.pt` is an official Ultralytics model name. If the file is not found, Ultralytics downloads it on first run into the current working directory (internet access required).
2. **Manual**: download it from the assets of this repository's GitHub release and place it anywhere, then pass the path, e.g. `--model weights/yolo26x.pt`.

`yolo26m.pt` can also be fetched automatically by passing the bare name `--model yolo26m.pt`.

## Source and license

- Source: [Ultralytics](https://github.com/ultralytics/ultralytics), official pretrained COCO weights.
- License: **AGPL-3.0** (Ultralytics). Commercial use outside AGPL terms requires an Ultralytics Enterprise License.
- The weights are redistributed unmodified.

## Using other models

Any Ultralytics detection model whose class 0 is `person` works, e.g. other YOLO26 sizes (`yolo26n.pt`, `yolo26s.pt`, `yolo26l.pt`) or older families (`yolo11x.pt`, `yolov8x.pt`). Models with a different class mapping would need a change to `Config.classes`.
