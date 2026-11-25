# How it works

This page describes the processing done for every frame, the meaning of each parameter, and how to tune them.

## 1. Detection and tracking

Each frame goes to `YOLO.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.10, classes=[0], imgsz=1920)`.

- `classes=[0]` keeps only the COCO `person` class.
- `persist=True` keeps tracker state between calls, so IDs carry over from frame to frame.
- ByteTrack associates high-confidence detections first, then tries to match low-confidence ones to existing tracks. With the stock `bytetrack.yaml` (`track_high_thresh: 0.25`, `track_low_thresh: 0.1`, `new_track_thresh: 0.25`) the low `conf` helps keep an already tracked person alive through weak detections, but a new track only starts from a detection scoring at least 0.25.
- Detections that the tracker has not yet confirmed (no ID) are ignored for that frame.

For every tracked box `(x1, y1, x2, y2)`, the **foot point** is

```
p = ( floor((x1 + x2) / 2), y2 )
```

The foot point approximates where the person touches the ground. It is used for both the trail and the heatmap, which makes the map show floor occupancy rather than where bodies appear in the image.

## 2. Trajectories

`TrajectoryStore` keeps the last `trail` (default 30) foot points for every track ID in a bounded deque. The points are drawn as an open yellow polyline, with the green box and an `ID: n (conf)` label on top. The trail of an ID that has received no point for 300 consecutive frames (`Config.trail_max_age`) is dropped, so memory does not grow with the number of people who have left the scene; ByteTrack does not revive IDs lost for longer than its buffer (30 frames by default), so drawn trails are unaffected.

## 3. Heatmap accumulation

Per frame `t`:

1. Start from an all-zero float mask `M_t` the size of the frame.
2. For every foot point, draw a filled circle of radius `r` (default 18 px) with value `I` (default 3.5).
3. Smooth: `B_t = GaussianBlur(M_t, k x k)`, `k = 35`, sigma derived from `k` by OpenCV.
4. Accumulate with exponential decay:

```
H_t = d_f * H_{t-1} + B_t        d_f = d ** (30 / fps),  d = 0.995 at 30 FPS
```

`--decay` is the per-frame factor at a 30 FPS reference. The factor actually applied per processed frame is `d_f = d ** (30 / fps)`, with `fps` the real processing rate of the video, so the memory is defined in seconds and does not change with the frame rate. At 30 FPS `d_f = d`.

Implementation note: the mask is allocated once and only the bounding box of the circles, grown by `k // 2 + 1` pixels, is drawn, blurred and added. Outside that box the blurred mask is zero, so the result equals the full-frame formula up to float rounding; frames without detections only apply the decay.

Properties of this update:

- An event `n` frames old has weight `d_f^n`. Half-life in seconds: `ln(0.5) / ln(d) / 30`, about 4.6 s for `d = 0.995` at any frame rate (138 frames at 30 FPS, 69 at 15 FPS, 276 at 60 FPS).
- A person standing still at one spot drives the value there towards `B / (1 - d)`, i.e. 200x the single-frame value.
- Circles are drawn (set), not added: two people overlapping in the same frame produce the same mask value as one. Density builds up only across frames.
- `H_t` is linear in `I`. Since the display uses min-max normalization (below), any positive `I` gives the same rendered heatmap and the same `--save-heatmap` image; it only scales the raw accumulator values. The CLI rejects `I <= 0` (zero would give an empty map, a negative value an inverted one).

## 4. Visualization

1. **Normalize**: `N_t = 255 * (H_t - min H_t) / (max H_t - min H_t)` (OpenCV `NORM_MINMAX`), rounded to the nearest integer and saturated to `uint8` (values stay in 0-255, no wrap-around); accumulator values below `1e-20` count as zero. If the accumulator is all zeros, the result is all zeros.
2. **Colorize**: `C_t = applyColorMap(N_t, COLORMAP_JET)`; blue = little or no recent presence, red = the hottest spot in the current accumulator.
3. **Blend**: `V_t = alpha * frame + (1 - alpha) * C_t` with `alpha = 0.6`.
4. **HUD**: a black rectangle blended at 60 % opacity in the top-left corner with three lines: current occupancy (tracks in this frame), peak occupancy (max over all frames so far), total unique people (distinct track IDs so far).
5. **Compose**: `[tracking | heatmap]` horizontally, so the output is `2W x H`.

Because normalization happens every frame, colors are relative: the hottest pixel is always red, even when activity is low overall.

## 5. Preview scaling

The preview is resized by

```
ratio = min(1.0, int(0.90 * screen_w) / (2W), int(0.85 * screen_h) / H)
```

so it fits within 90 % of the screen width and 85 % of the screen height and is never upscaled. The recorded video always keeps full resolution. If the screen size cannot be read (no display), the ratio is 1.0.

## Parameter guide

| Parameter | Effect when increased | Effect when decreased |
|-----------|-----------------------|-----------------------|
| `conf` | Fewer false positives, may lose small or distant people | More people kept, more false boxes |
| `imgsz` | Better detection of small people, slower, more GPU memory | Faster, small people missed |
| `decay` | Longer memory (closer to a cumulative map) | Map follows recent motion only |
| `intensity` | No visible change (min-max normalization cancels a uniform scale); raw accumulator values grow | (same) |
| `radius` | Wider footprint per person | More precise, spottier map |
| `blur` | Smoother, more diffuse map | Sharper blobs |
| `alpha` | More of the video visible | More of the colormap visible |
| `trail` | Longer trails | Shorter trails |

## Tuning tips

- **Long-term occupancy** (where people spend time over the whole video): raise `--decay` towards `0.999` or higher. With `0.999` the half-life is about 23 s (693 frames at 30 FPS).
- **Live flow** (where people are right now): lower `--decay` to `0.98`-`0.99`.
- **Other resolutions**: scale `--radius` and `--blur` with the image width (see [Resolution scaling](#resolution-scaling)).
- **Slow hardware**: use `--model weights/yolo26m.pt` and `--imgsz 1280` or `960`; small, distant people may be lost.
- **Too many spurious boxes**: raise `--conf` to `0.25`-`0.35`.
- **Frequent ID switches** (inflating "Total Unique People"): try `--tracker botsort.yaml`, or tune the Ultralytics tracker YAML (e.g. `track_buffer`) by copying it and passing its path to `--tracker`.
- **Export for reports**: `--save-heatmap path.png` stores the final accumulator colorized at input resolution; combine with a high `--decay` for a whole-video summary.

## Resolution scaling

`--radius` (default 18) and `--blur` (default 35) are measured in pixels and are not scaled automatically. The defaults are tuned for input about 1920 px wide. For a different width `W`, pass values scaled by `W / 1920`:

| Input width | `--radius` | `--blur` (odd) |
|---|---|---|
| 1280 | 12 | 23 |
| 1920 (default) | 18 | 35 |
| 2560 | 24 | 47 |
| 3840 (4K) | 36 | 71 |

```bash
crowd-heatmap -i data/street_4k.mp4 --radius 36 --blur 71
```

Round `--blur` to the nearest odd integer. The same applies to the HUD text, which is drawn at a fixed pixel size. `--decay` is not affected: it is defined per second of video (see above).
