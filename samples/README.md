# Samples

| File | Content |
|---|---|
| `abbey_road_20s.mp4` | 20 s of the Abbey Road zebra crossing in London at night (1920×1080, 30 FPS, no audio): pedestrians crossing and standing on the pavement. |
| `abbey_road_day_20s.mp4` | 20 s of the same crossing by day, with more people (1920×1080, 30 FPS, no audio). |
| `shinjuku_20s.mp4` | 20 s of a scramble crossing in Shinjuku, Tokyo, at night while the pedestrian light is green: a dense crowd crossing in every direction (1920×1080, 30 FPS, no audio). |

## Usage

```bash
crowd-heatmap -i samples/abbey_road_20s.mp4 -o outputs/abbey_road_proc.mp4 --no-display
```

## Source

Captured from public live webcams on YouTube:

- `abbey_road_20s.mp4`, `abbey_road_day_20s.mp4`: Abbey Road crossing live webcam on YouTube: https://www.youtube.com/watch?v=zMCea32gpmg
- `shinjuku_20s.mp4`: Tokyo Shinjuku crossing live webcam on YouTube: https://www.youtube.com/watch?v=6dp-bvQ7RWo

The daytime Abbey Road and Shinjuku clips are cut from the longer clips of the [directional-line-crossing-counter](https://github.com/karemdm/directional-line-crossing-counter) samples (`abbey_road_3min.mp4` from 150 s, `shinjuku_3min.mp4` from 64 s), choosing the 20 s with the most people.

People appear in a public space. The clips are included only to demonstrate the tool; if you appear in one and want it
removed, please open an issue.
