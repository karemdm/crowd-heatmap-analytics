# Results

Annotated videos produced by this project.

- `abbey_road_heatmap.mp4`: Pedestrian crossing: tracking (left) and accumulated heatmap with occupancy statistics (right).
- `street_2_heatmap.mp4`: Street scene: tracking and heatmap side by side.
- `street_3_heatmap.mp4`: Street scene: tracking and heatmap side by side.

## Source footage

- `abbey_road_heatmap.mp4`: made from the sample clip [`samples/abbey_road_20s.mp4`](../samples/README.md), cut from the public Abbey Road crossing live webcam on YouTube (https://www.youtube.com/watch?v=zMCea32gpmg). It was produced with the current code (decay defined per second) and the YOLO26x model:

  ```bash
  crowd-heatmap -i samples/abbey_road_20s.mp4 -o outputs/abbey_road_proc.mp4 --model yolo26x.pt --save-heatmap outputs/abbey_road_heat.png --no-display
  ```

- `street_2_heatmap.mp4`, `street_3_heatmap.mp4`: made from public street recordings; source link pending.

All videos here are the tool's side-by-side output re-encoded for the repository (H.264, 1920 px wide, 15 FPS). `docs/assets/demo.gif` is also made from the tool's output on the same kind of public footage.

People appear in public spaces; the outputs are shown only to illustrate the method.
If you appear in a video and want it removed, please open an issue.
