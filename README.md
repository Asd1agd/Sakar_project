# Perception Pipeline for Mobile Robotics: Tracking and Spatial Reasoning

A real-time multi-object tracking and spatial reasoning system for mobile robots operating in warehouse-like environments. The pipeline processes raw YOLO detections from a camera stream and produces stable object tracks, depth estimates, and event-level insights.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
  - [Detection Ingestion](#1-detection-ingestion)
  - [Multi-Object Tracking: GridClusterCOM](#2-multi-object-tracking-gridclustercom)
  - [Spatial Reasoning (Depth Estimation)](#3-spatial-reasoning-depth-estimation)
  - [Event-Level Intelligence](#4-event-level-intelligence)
- [Algorithm Deep Dive](#algorithm-deep-dive)
- [Parameters](#parameters)
- [Handling Detection Imperfections](#handling-detection-imperfections)
- [Failure Cases & Mitigations](#failure-cases--mitigations)
- [Engineering Trade-offs](#engineering-trade-offs)
- [Potential Improvements](#potential-improvements)
- [Output](#output)

---

## Overview

This system transforms noisy, frame-by-frame object detections into reliable, structured intelligence:

- **Stable tracks** — objects are assigned persistent IDs across frames
- **Depth estimates** — distance inferred from bounding box area using a pinhole camera model
- **Event insights** — virtual zone crossing counts, closest object detection, track lifecycle management

The core tracker, `GridClusterCOM`, is a deterministic, online clustering algorithm using a spatio-temporal grid and centroid-based association. It is designed to be explainable, robust to real-world sensor noise, and suitable for real-time operation.

---

## Features

- ✅ Real-time multi-object tracking with stable IDs
- ✅ Handles missed detections, jitter, duplicates, and false positives
- ✅ Depth estimation from bounding box area (no extra sensors required)
- ✅ Virtual zone crossing counter (unique objects only)
- ✅ Closest object detection per frame (highlighted visually)
- ✅ Track lifecycle management (active/lost states)
- ✅ Grid-based O(1) nearest-neighbor search — scales with scene complexity
- ✅ Fully deterministic — reproducible outputs, no randomness
- ✅ Works with real YOLO detections or simulated data

---

## Project Structure

```
PythonProject9/
├── closest_cluster.py          # GridClusterCOM tracker implementation
├── sakar_basic.py              # Main pipeline script (YOLO + tracking + visualization)
├── main.py                     # Entry point / alternate runner
├── best (3).pt                 # Custom-trained YOLO model weights
├── yolov8n.pt                  # YOLOv8 nano baseline weights
├── output_video.mp4            # Annotated output video
└── *.mp4                       # Input video files
```

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install opencv-python numpy ultralytics
```

No additional configuration is needed. The YOLO model weights (`best (3).pt`) must be present in the project directory.

---

## Usage

### Run on real video

```bash
cd PythonProject9
python sakar_basic.py
```

Press `q` to quit. The annotated output is saved as `output_video.mp4`.

### Run with simulated detections

To use synthetic data instead of YOLO, replace the detection loop in `sakar_basic.py` with a generator that yields randomized bounding boxes with controlled noise. The tracker API (`add_point`) is identical — no other changes are needed.

---

## System Architecture

The pipeline consists of four logical stages:

```
Camera Stream
     │
     ▼
┌─────────────────────┐
│  Detection Ingestion │  ← YOLO bounding boxes, class labels, confidence scores
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  GridClusterCOM     │  ← Multi-object tracking: stable IDs via spatio-temporal grid
│  Tracker            │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Spatial Reasoning  │  ← Depth estimation from bounding box area
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Event Intelligence │  ← Zone counting, closest object, track status
└─────────────────────┘
     │
     ▼
Annotated Output Video
```

---

### 1. Detection Ingestion

A pre-trained YOLO model (`best (3).pt`) processes each video frame and returns:
- Bounding boxes `(x1, y1, x2, y2)`
- Class labels and confidence scores

Real-world imperfections handled by the pipeline include:

| Imperfection | Description |
|---|---|
| Missed detections | YOLO may fail to detect an object in some frames |
| Duplicate detections | Multiple boxes for the same physical object |
| Bounding box jitter | Slight coordinate variation between consecutive frames |
| False positives | Detections of non-existent objects |

---

### 2. Multi-Object Tracking: GridClusterCOM

Implemented in `closest_cluster.py`. A deterministic, online clustering algorithm that maintains a set of tracks and associates new detections to existing ones using spatial proximity.

#### Track Representation

Each track stores:
- A **leader** (centroid): the geometric mean of recent detection positions
- A **bounded buffer** of the last `reduction_threshold` points (default: 4)
- Metadata: frame history, cluster ID

#### Association Logic

For each new detection at `(cx, cy)` in frame `f`:

1. Compute the grid cell key `(gx, gy, f)` using `cell_size = com_radius / √2`
2. Generate candidate cells: the 3×3 spatial neighbourhood across the last `frame_threshold` frames
3. Retrieve candidate cluster indices from `cell_to_cluster` dictionary
4. Compute Euclidean distance to each candidate cluster's leader
5. Assign to the nearest cluster with distance `≤ com_radius`; otherwise create a new cluster

This guarantees that any point within `com_radius` of an existing leader is always found (the cell-size choice is exact), while limiting candidates to a constant-size set regardless of total track count.

#### Track Lifecycle

| Stage | Trigger |
|---|---|
| **Creation** | No existing cluster within `com_radius` |
| **Update** | Detection associated; leader recomputed from buffer |
| **Active** | Seen in `max(4×FPS, 10)` frames (tracked in main script) |
| **Deletion** | No new point received for `frame_threshold` frames |

---

### 3. Spatial Reasoning (Depth Estimation)

Depth is estimated using an inverse-square pinhole camera model:

```
depth = d_ref × √(A_ref / area)
```

Where:
- `area = (x2 - x1) × (y2 - y1)` — bounding box pixel area
- `A_ref` — reference area at known distance `d_ref` (calibrated manually)
- Default: `calibration_area_for_1m = 10000 × 1.5`

**Assumptions:**
- Constant physical object size across all detected objects
- Object is fronto-parallel and fully visible
- No additional sensors required (camera only)

**Limitations:**
- Sensitive to bounding box jitter (directly propagates to depth noise)
- Same scale used for all object classes
- Provides relative depth (useful for ranking) rather than absolute ground truth

---

### 4. Event-Level Intelligence

#### Closest Object Detection
- Each frame, all active tracks are compared by estimated depth
- The closest track is highlighted with a **red bounding box** and labelled `"closest"`
- Runs in O(N) per frame

#### Virtual Zone Crossing Counter
- A fixed rectangular zone is drawn on the frame (cyan rectangle)
- Each time a tracked object's centroid enters the zone for the first time, `virtual_box_passed_count` is incremented
- Each object is counted only once (unique crossing semantics)

#### Track Status
Tracks are labelled `"Active"` in the main script once they have been consistently observed for `max(4×FPS, 10)` frames. This suppresses transient false positive tracks from appearing as reliable detections.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `com_radius` | — | Maximum centroid-to-detection distance for association |
| `cell_size` | `com_radius / √2` | Grid cell size (auto-derived) |
| `frame_threshold` | 80 | Frames to keep a track alive without new detections (~2–3s at 30fps) |
| `reduction_threshold` | 4 | Buffer size for centroid averaging |

---

## Handling Detection Imperfections

| Imperfection | Mitigation |
|---|---|
| **Missed detections** | Tracks persist for `frame_threshold` frames; object re-acquired on reappearance |
| **Duplicate detections** | Both detections fall near the same leader, updating the same track; centroid shifts minimally |
| **Bounding box jitter** | Centroid averaged over last `reduction_threshold` points — random noise is smoothed out |
| **False positives** | Creates a short-lived isolated track; receives no further updates and is deleted after `frame_threshold` frames |

---

## Failure Cases & Mitigations

| Failure Case | How the Design Handles It |
|---|---|
| Temporary occlusion | Track kept alive for `frame_threshold` frames; re-matched on reappearance |
| Crossing objects | May merge into one track if they come within `com_radius` — known limitation of centroid-only association |
| Object leaving frame | Track stops updating; removed after `frame_threshold` frames |
| Frequent false positives | Short-lived tracks suppressed by the "Active" threshold in main script |

---

## Engineering Trade-offs

### Fixed Buffer vs. Incremental Mean

Two centroid update strategies are supported:

| Mode | Behaviour | Trade-off |
|---|---|---|
| **Unbounded buffer** (`reduction_threshold=None`) | Stores all points; uses incremental mean | Exact centroid history, but memory grows indefinitely |
| **Bounded buffer** (`reduction_threshold=N`) | Stores last N points; recomputes centroid from them | Bounded memory; centroid is more responsive to recent motion |

The default bounded buffer of size 4 balances memory efficiency and adaptability — the centroid reacts quickly to object movement while smoothing out short-term jitter.

### Grid Search vs. Global Search

The spatio-temporal grid limits candidates per detection to at most `9 × frame_threshold` cells. In practice far fewer cells are occupied, making association effectively O(1) rather than O(total tracks). This is essential for real-time performance in dense scenes.

---

## Potential Improvements

1. **Motion prediction** — Kalman filters or constant-velocity models for better occlusion handling and crossing objects
2. **Appearance features** — Color histograms or lightweight embeddings to disambiguate spatially close objects
3. **Adaptive `com_radius`** — Scale radius by estimated depth (distant objects appear smaller and move less in image space)
4. **Ground plane projection** — Use camera calibration to project bounding box feet onto a 3D ground plane for accurate positioning
5. **Structured output** — Write per-frame tracking results to JSON/CSV for offline analysis or downstream planning
6. **Automatic parameter tuning** — Derive `frame_threshold` from detected frame rate automatically

---

## Output

The pipeline produces:
- **Annotated video** (`output_video.mp4`) with:
  - Bounding boxes per tracked object with persistent IDs
  - Depth estimate labels
  - Red box + `"closest"` label on nearest object
  - Cyan virtual zone rectangle
  - Running zone crossing counter
  - `"Active"` / `"Lost"` track status labels
