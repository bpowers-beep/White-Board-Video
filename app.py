"""
Powers Video Creation App — Streamlit App  ·  Version 4.0: Professional Vector-Style Path Drawing
Converts sketch images + a text script into a whiteboard-style MP4 video
with an AI voiceover perfectly synced to the drawing animations.


Animation engine: Three-Phase Professional Vector-Sketch Engine
  Phase 1 — Outline Phase (first 50 % of scene duration):
      Edge-detection / pencil-sketch filter extracts the main high-contrast
      structural lines.  cv2.findContours builds the exact pixel-coordinate
      paths of every stroke.


      NEW v4.0 — Text Center-Line Logic:
          Detected strokes are skeletonized (morphological thinning) so the
          hand executes fluid single-stroke movements along the true center-line
          of each letter/line rather than tracing the outer contour border.


      NEW v4.0 — Hierarchical Distance Sorting:
          Contours are grouped into spatial clusters (DBSCAN-style bounding-box
          proximity).  The hand completes every stroke inside one structural
          cluster (e.g. a flower, the cat's face) before jumping to the next
          nearest cluster — no mid-element teleporting.


      NEW v4.0 — Natural Pen Lift:
          When the hand must travel between disconnected clusters it briefly
          lifts (the hand graphic fades to ~40 % opacity and accelerates) to
          mimic realistic human sketching cadence.


  Phase 2 — Shading & Color Phase (last 50 % of scene duration):
      The original full-detail artwork (soft pencil shading, paper textures,
      gorgeous colors) smoothly fades and bleeds into the outline drawing.


  Hold — the final fully-rendered frame is held completely still for the last
      second of the scene so the viewer can admire the finished piece.


Compatible with moviepy 1.x AND 2.x.
"""

import asyncio
import base64
import html
import json
import os
import tempfile
import textwrap
import time
import wave
import struct

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Powers Video Creation App",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── moviepy: support both 1.x (moviepy.editor) and 2.x (moviepy) ─────────────
try:
    from moviepy import AudioFileClip, ImageSequenceClip, concatenate_videoclips
    _MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import AudioFileClip, ImageSequenceClip, concatenate_videoclips
    _MOVIEPY_V2 = False


def _clip_set_duration(clip, duration):
    if _MOVIEPY_V2:
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def _clip_set_audio(video_clip, audio_clip):
    if _MOVIEPY_V2:
        return video_clip.with_audio(audio_clip)
    return video_clip.set_audio(audio_clip)


def _clip_set_position(clip, position):
    if _MOVIEPY_V2:
        return clip.with_position(position)
    return clip.set_position(position)


# ── Helper: ensure an event loop exists (needed for edge-tts on Windows) ─────
def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


# ══════════════════════════════════════════════════════════════════════════════
#  1.  CANVAS PREPARATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_canvas_image(pil_image: Image.Image, canvas_size: tuple) -> np.ndarray:
    """
    Resize and centre the original image (RGB, full detail) onto a white canvas.
    Returns an RGB uint8 array of shape (canvas_h, canvas_w, 3).
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    h_orig, w_orig = img_rgb.shape[:2]

    scale = min(canvas_size[0] / w_orig, canvas_size[1] / h_orig) * 0.88
    new_w = max(1, int(w_orig * scale))
    new_h = max(1, int(h_orig * scale))
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    x_off = (canvas_size[0] - new_w) // 2
    y_off = (canvas_size[1] - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img_resized

    return canvas  # RGB


# ══════════════════════════════════════════════════════════════════════════════
#  2.  PHASE 1 — OUTLINE EXTRACTION  (pencil-sketch / edge-detection)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_outline_canvas(canvas_img_rgb: np.ndarray):
    """
    Extract structural lines from a full-color or B&W canvas using
    LAB color-space edge detection.

    Colored strokes on white backgrounds (e.g. red "C A T") are detected just
    as reliably as dark ink because all three LAB channels are edge-detected
    independently and merged: L catches luminance edges, a/b catch color edges.
    """
    # LAB color-aware edges
    lab = cv2.cvtColor(canvas_img_rgb, cv2.COLOR_RGB2Lab)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_sm = cv2.bilateralFilter(l_chan, d=9, sigmaColor=75, sigmaSpace=75)
    a_sm = cv2.bilateralFilter(a_chan, d=9, sigmaColor=75, sigmaSpace=75)
    b_sm = cv2.bilateralFilter(b_chan, d=9, sigmaColor=75, sigmaSpace=75)
    edges_l    = cv2.Canny(l_sm, threshold1=25, threshold2=75)
    edges_a    = cv2.Canny(a_sm, threshold1=15, threshold2=45)
    edges_b    = cv2.Canny(b_sm, threshold1=15, threshold2=45)
    # Grayscale Canny as safety net
    gray   = cv2.cvtColor(canvas_img_rgb, cv2.COLOR_RGB2GRAY)
    gray_sm = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges_gray = cv2.Canny(gray_sm, threshold1=25, threshold2=75)
    # Merge all channels
    edges = cv2.bitwise_or(edges_l,    edges_a)
    edges = cv2.bitwise_or(edges,      edges_b)
    edges = cv2.bitwise_or(edges,      edges_gray)
    # Slight dilation for hand-drawn body
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    # White-background outline
    outline_gray = cv2.bitwise_not(edges_dilated)
    pencil_hint  = cv2.normalize(gray_sm, None, 180, 255, cv2.NORM_MINMAX)
    blended      = cv2.addWeighted(outline_gray, 0.90, pencil_hint, 0.10, 0)
    outline_rgb  = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)
    return outline_rgb, edges_dilated


def _build_reveal_order(edge_binary: np.ndarray) -> np.ndarray:
    """
    Build a 2-D float32 reveal-order map (0=first, 1=last) from the edge mask.

    Uses spatial left-to-right, top-to-bottom position instead of pixel
    darkness. This ensures the contour drawing path and the pixel reveal sweep
    stay in sync: the hand draws left-side strokes first, then moves right,
    exactly like a human writing on a whiteboard.
    Small organic noise prevents a hard wipe-line look.
    """
    H, W = edge_binary.shape
    xs = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))
    ys = np.repeat(np.linspace(0.0, 1.0, H, dtype=np.float32)[:, np.newaxis], W, axis=1)
    spatial = xs * 0.80 + ys * 0.20
    noise   = np.random.default_rng(seed=42).random((H, W)).astype(np.float32)
    reveal_order = spatial * 0.88 + noise * 0.12
    reveal_order = cv2.GaussianBlur(reveal_order, (21, 21), 0)
    lo, hi = reveal_order.min(), reveal_order.max()
    if hi > lo:
        reveal_order = (reveal_order - lo) / (hi - lo)
    return reveal_order.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  STROKE EXTRACTION  (v5.0 — LAB color-aware, skeletonised center-lines)
# ══════════════════════════════════════════════════════════════════════════════

def _skeletonize_edges(edge_binary: np.ndarray) -> np.ndarray:
    """
    Reduce thick stroke regions to single-pixel center-lines via iterative
    morphological thinning (Zhang-Suen style).  Works on a uint8 mask where
    255 = stroke pixel.
    """
    img = (edge_binary > 0).astype(np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, close_k, iterations=1)

    skeleton = np.zeros_like(img)
    element  = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp   = cv2.dilate(eroded, element)
        temp   = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skeleton * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  CHARACTER CLUSTERING  (v5.0 — baseline-aware left-to-right text sort)
# ══════════════════════════════════════════════════════════════════════════════

_CONTOUR_MIN_ARC_PX    = 4.0   # noise gate: discard shorter contours
_CLUSTER_PROXIMITY_FRAC = 0.06  # fraction of canvas diagonal for proximity merge
_PEN_LIFT_DIST_SQ_FRAC  = 0.002 # pen-lift threshold (fraction of diag²)
_PEN_LIFT_OPACITY        = 0.35  # hand opacity during pen-lift travel


def _bounding_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def _contour_center(contour):
    x, y, w, h = _bounding_box(contour)
    return x + w // 2, y + h // 2


def _group_contours_into_clusters(contours: list, canvas_shape: tuple) -> list:
    """
    Group contours into character/word clusters optimised for horizontal text.

    Strategy (v5.0):
      1. Compute a 'text-baseline row' for each contour: the vertical band its
         bounding box occupies.  Two contours on the same baseline row that are
         also close horizontally are merged first.
      2. Then do a standard proximity BFS for anything still unmerged.
      3. Final clusters are sorted left-to-right by their leftmost x coordinate,
         then top-to-bottom.  This ensures "C A T S" is drawn C → A → T → S.
    """
    H, W = canvas_shape
    diag      = np.sqrt(H ** 2 + W ** 2)
    threshold = _CLUSTER_PROXIMITY_FRAC * diag
    # For text, allow wider horizontal reach than vertical
    h_threshold = threshold * 1.8
    v_threshold = threshold * 0.9

    n = len(contours)
    assigned = [-1] * n

    def _box(i):
        x, y, w, h = _bounding_box(contours[i])
        return x, y, x + w, y + h   # x0,y0,x1,y1

    cluster_id = 0
    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = cluster_id
        frontier = [i]
        while frontier:
            cur = frontier.pop()
            cx0, cy0, cx1, cy1 = _box(cur)
            ccx = (cx0 + cx1) / 2
            ccy = (cy0 + cy1) / 2
            for j in range(n):
                if assigned[j] != -1:
                    continue
                jx0, jy0, jx1, jy1 = _box(j)
                jcx = (jx0 + jx1) / 2
                jcy = (jy0 + jy1) / 2
                dx = abs(ccx - jcx)
                dy = abs(ccy - jcy)
                # Relaxed horizontal tolerance for text on the same baseline
                same_baseline = dy < v_threshold and dx < h_threshold
                # Standard radial proximity
                near = np.sqrt(dx**2 + dy**2) <= threshold
                if same_baseline or near:
                    assigned[j] = cluster_id
                    frontier.append(j)
        cluster_id += 1

    clusters: list = [[] for _ in range(cluster_id)]
    for i, cid in enumerate(assigned):
        clusters[cid].append(contours[i])

    # Sort clusters: primary = top-to-bottom baseline, secondary = left-to-right
    def _cluster_sort_key(cl):
        xs = [_bounding_box(c)[0] for c in cl]
        ys = [_bounding_box(c)[1] for c in cl]
        return (min(ys) // 60, min(xs))   # 60-px vertical bands = same row

    clusters.sort(key=_cluster_sort_key)
    return clusters


def _cluster_left_x(cluster: list) -> int:
    return min(_bounding_box(c)[0] for c in cluster)


def _cluster_centroid(cluster: list) -> tuple:
    xs = [_contour_center(c)[0] for c in cluster]
    ys = [_contour_center(c)[1] for c in cluster]
    return int(np.mean(xs)), int(np.mean(ys))


def _sort_contours_within_cluster(cluster: list, start_pt: tuple) -> list:
    """
    Nearest-neighbour chain of contours within one cluster starting from
    start_pt.  Each contour can be reversed to minimise travel.
    """
    def _pt(c, idx):
        return int(c[idx][0][0]), int(c[idx][0][1])

    remaining = list(range(len(cluster)))
    ordered   = []
    cur = start_pt

    while remaining:
        best_i, best_d, best_flip = None, float('inf'), False
        ex, ey = cur
        for i in remaining:
            sx, sy = _pt(cluster[i],  0)
            ex2, ey2 = _pt(cluster[i], -1)
            ds = (sx-ex)**2 + (sy-ey)**2
            de = (ex2-ex)**2 + (ey2-ey)**2
            if ds <= de:
                d, flip = ds, False
            else:
                d, flip = de, True
            if d < best_d:
                best_d, best_i, best_flip = d, i, flip
        chosen = cluster[best_i]
        if best_flip:
            chosen = chosen[::-1]
        ordered.append(chosen)
        remaining.remove(best_i)
        cur = _pt(ordered[-1], -1)
    return ordered


# ══════════════════════════════════════════════════════════════════════════════
#  5.  MASTER DRAWING PATH  (v5.0 — velocity-based frame allocation)
# ══════════════════════════════════════════════════════════════════════════════

def _build_drawing_waypoints(edge_binary: np.ndarray) -> list:
    """
    Build the complete ordered list of (x, y, is_lift) drawing waypoints.

    v5.0 pipeline:
      1. Skeletonise edge mask → single-pixel center-lines.
      2. FindContours on skeleton (CHAIN_APPROX_NONE = every pixel).
      3. Noise-gate: drop contours shorter than _CONTOUR_MIN_ARC_PX.
      4. Group into clusters with baseline-aware text clustering.
      5. Clusters sorted left-to-right / top-to-bottom.
      6. Within each cluster, chain contours nearest-neighbour.
      7. Insert pen-lift waypoints (is_lift=True) between clusters.

    Returns:
        List of (x, y, is_lift) — spatial order only, no time values yet.
        Timing is assigned later by _allocate_frames_by_distance().
    """
    H, W = edge_binary.shape

    skeleton = _skeletonize_edges(edge_binary)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        # Fallback: horizontal sweep
        return [(int(W * t), H // 2, False) for t in np.linspace(0, 1, 200)]

    contours = [c for c in contours if cv2.arcLength(c, closed=False) >= _CONTOUR_MIN_ARC_PX]
    if not contours:
        return [(int(W * t), H // 2, False) for t in np.linspace(0, 1, 200)]

    clusters = _group_contours_into_clusters(contours, (H, W))

    pen_lift_sq = _PEN_LIFT_DIST_SQ_FRAC * (W**2 + H**2)

    def _pt(c, idx):
        return int(c[idx][0][0]), int(c[idx][0][1])

    waypoints: list = []
    cur_end = None

    for cluster in clusters:
        if not cluster:
            continue

        # Entry point: leftmost start of any contour in this cluster
        entry_candidates = [_pt(c, 0) for c in cluster] + [_pt(c, -1) for c in cluster]
        if cur_end is None:
            # First cluster: start at topmost-leftmost entry
            start_pt = min(entry_candidates, key=lambda p: (p[1], p[0]))
        else:
            # Nearest entry to current pen position
            ex, ey = cur_end
            start_pt = min(entry_candidates, key=lambda p: (p[0]-ex)**2 + (p[1]-ey)**2)

        # Pen-lift travel segment between clusters
        if cur_end is not None:
            ex, ey = cur_end
            jump_sq = (start_pt[0]-ex)**2 + (start_pt[1]-ey)**2
            if jump_sq > pen_lift_sq:
                n_lift = max(3, int(np.sqrt(jump_sq) / 6))
                for k in range(1, n_lift + 1):
                    alpha = k / n_lift
                    lx = int(ex + (start_pt[0]-ex) * alpha)
                    ly = int(ey + (start_pt[1]-ey) * alpha)
                    waypoints.append((lx, ly, True))

        # Draw all strokes in this cluster
        ordered = _sort_contours_within_cluster(cluster, start_pt)
        for contour in ordered:
            for pt in contour:
                waypoints.append((int(pt[0][0]), int(pt[0][1]), False))
        cur_end = _pt(ordered[-1], -1)

    return waypoints


def _allocate_frames_by_distance(waypoints: list, num_frames: int) -> list:
    """
    Assign a frame index to every waypoint proportionally to cumulative arc
    length (velocity-based allocation).

    This gives each waypoint a time t ∈ [0,1] such that the hand moves at a
    visually uniform speed: long straight strokes get more frames than dense
    curly sections, so the marker appears to travel at a steady pace.

    Pen-lift segments travel at _PEN_LIFT_SPEED_MULT × normal speed so the
    hand snaps between characters quickly without consuming too many frames.

    Returns:
        List of (x, y, t, is_lift) sorted by t ∈ [0,1].
    """
    PEN_LIFT_SPEED = 5.0   # pen lifts travel this many times faster

    if not waypoints:
        return []

    # Compute cumulative arc-length, weighting lift segments as faster
    n = len(waypoints)
    arc = [0.0] * n
    for i in range(1, n):
        x0, y0, _ = waypoints[i-1]
        x1, y1, lift = waypoints[i]
        dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        weight = 1.0 / PEN_LIFT_SPEED if lift else 1.0
        arc[i] = arc[i-1] + dist * weight

    total = arc[-1]
    if total == 0:
        total = 1.0

    tagged = []
    for i, (x, y, is_lift) in enumerate(waypoints):
        t = arc[i] / total
        tagged.append((x, y, t, is_lift))

    # Thin to ≤ 6000 waypoints; always keep lift waypoints
    if len(tagged) > 6000:
        lift_pts = [p for p in tagged if p[3]]
        draw_pts = [p for p in tagged if not p[3]]
        step = max(1, len(draw_pts) // max(1, 6000 - len(lift_pts)))
        draw_pts = draw_pts[::step]
        combined = sorted(lift_pts + draw_pts, key=lambda p: p[2])
        # Re-normalise t after thinning
        if combined:
            t_max = combined[-1][2]
            tagged = [(x, y, tt / max(t_max, 1e-9), lft)
                      for x, y, tt, lft in combined]

    return tagged


def _build_contour_drawing_path(edge_binary: np.ndarray,
                                _unused_reveal_order=None) -> list:
    """
    Public entry point (signature kept for compatibility).
    Returns (x, y, t, is_lift) list with velocity-based t values.
    """
    waypoints = _build_drawing_waypoints(edge_binary)
    return _allocate_frames_by_distance(waypoints, num_frames=6000)


def _get_hand_tip_for_frame(
    tagged_path: list,
    t: float,
    prev_tip: tuple,
    smoothing: float = 0.30,
) -> tuple:
    """
    Binary-search the tagged path for the waypoint at time t, then apply
    a light EMA for smooth motion.  Returns (sx, sy, is_lift).
    """
    if not tagged_path:
        return prev_tip[0], prev_tip[1], False

    lo, hi = 0, len(tagged_path) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if tagged_path[mid][2] < t:
            lo = mid + 1
        else:
            hi = mid

    idx = max(0, min(lo, len(tagged_path) - 1))
    raw_x, raw_y, _, is_lift = tagged_path[idx]

    # Pen lifts: reduced smoothing so the hand snaps across quickly
    eff = smoothing * 0.20 if is_lift else smoothing
    sx = int(prev_tip[0] * (1 - eff) + raw_x * eff)
    sy = int(prev_tip[1] * (1 - eff) + raw_y * eff)
    return sx, sy, is_lift


# ══════════════════════════════════════════════════════════════════════════════
#  6.  HAND / MARKER GRAPHIC RENDERER  (v5.0 — unchanged interface)
# ══════════════════════════════════════════════════════════════════════════════

# ── Load the custom hand asset once at module level ───────────────────────────
_HAND_ASSET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Gemini_Generated_Image_uwy2f4uwy2f4uwy2-removebg-preview.png",
)
_HAND_TIP_X_SRC = 338
_HAND_TIP_Y_SRC = 180
_hand_asset_cache: dict = {}


def _get_hand_asset(target_height: int):
    if target_height in _hand_asset_cache:
        return _hand_asset_cache[target_height]
    if not os.path.exists(_HAND_ASSET_PATH):
        _hand_asset_cache[target_height] = None
        return None
    options = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS
    pil = Image.open(_HAND_ASSET_PATH).convert("RGBA")
    src_w, src_h = pil.size
    scale = target_height / src_h
    new_w = max(1, int(src_w * scale))
    pil_r = pil.resize((new_w, target_height), options)
    rgba  = np.array(pil_r)
    tip_x = int(_HAND_TIP_X_SRC * scale)
    tip_y = int(_HAND_TIP_Y_SRC * scale)
    result = (rgba, tip_x, tip_y)
    _hand_asset_cache[target_height] = result
    return result


def _draw_hand_marker(
    frame: np.ndarray,
    tip_x: int,
    tip_y: int,
    opacity: float = 1.0,
) -> np.ndarray:
    """
    Composite the hand PNG onto frame so the marker nib lands at (tip_x, tip_y).
    Falls back to a circle dot if the asset is missing.
    """
    H, W = frame.shape[:2]
    hand_h = max(60, int(H * 0.22))
    asset  = _get_hand_asset(hand_h)

    if asset is None:
        out = frame.copy()
        cv2.circle(out, (tip_x, tip_y), 6, (30, 30, 30), -1, cv2.LINE_AA)
        return out

    rgba, anchor_x, anchor_y = asset
    hh, hw = rgba.shape[:2]

    dst_x0 = tip_x - anchor_x;  dst_y0 = tip_y - anchor_y
    dst_x1 = dst_x0 + hw;        dst_y1 = dst_y0 + hh

    src_x0 = max(0, -dst_x0);   src_y0 = max(0, -dst_y0)
    src_x1 = hw - max(0, dst_x1 - W)
    src_y1 = hh - max(0, dst_y1 - H)

    cdx0 = max(0, dst_x0);  cdy0 = max(0, dst_y0)
    cdx1 = min(W, dst_x1);  cdy1 = min(H, dst_y1)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return frame

    patch = rgba[src_y0:src_y1, src_x0:src_x1]
    alpha = (patch[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    out   = frame.copy()
    roi   = out[cdy0:cdy1, cdx0:cdx1].astype(np.float32)
    blended = patch[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    out[cdy0:cdy1, cdx0:cdx1] = blended.astype(np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  7.  PHASE 1 — MASTER-SLAVE DRAW REVEAL  (v5.0)
# ══════════════════════════════════════════════════════════════════════════════

def _phase1_outline_frames(
    outline_canvas: np.ndarray,
    _unused_reveal_order,
    edge_binary: np.ndarray,
    num_frames: int,
) -> list:
    """
    Phase 1 — Master-Slave Draw Reveal (v5.0).

    The hand path is the MASTER timeline.  Nothing is revealed on the canvas
    until the marker nib physically touches that coordinate.

    Rendering:
      - Maintain a growing polyline list of all (x,y) points visited so far.
      - Each frame: paint a white canvas, then call cv2.polylines() with all
        accumulated drawing segments up to the current index.
      - Pen-lift segments are NOT drawn to the canvas (is_lift=True).
      - The stroke color is sampled from the full-color outline_canvas at
        the tip position so colored text is drawn in its own color.
      - Hand opacity drops to _PEN_LIFT_OPACITY during lifts.
    """
    H, W = edge_binary.shape[:2] if len(edge_binary.shape) == 2 else edge_binary.shape[:2]

    tagged_path = _build_contour_drawing_path(edge_binary)

    if not tagged_path:
        white = np.full((H, W, 3), 255, dtype=np.uint8)
        return [white] * num_frames

    # Pre-sample stroke color at every waypoint from the outline canvas
    # outline_canvas is white-bg with dark lines; for color we fall back to
    # a near-black marker color when the pixel is light (background).
    MARKER_COLOR = (15, 15, 15)  # default dark marker

    def _sample_color(x, y):
        cy = min(max(int(y), 0), outline_canvas.shape[0]-1)
        cx = min(max(int(x), 0), outline_canvas.shape[1]-1)
        r, g, b = outline_canvas[cy, cx]
        # If pixel is near-white (background), use marker color
        if r > 200 and g > 200 and b > 200:
            return MARKER_COLOR
        return (int(b), int(g), int(r))  # RGB→BGR for cv2

    # Build per-waypoint draw segments ahead of time
    # Each segment: list of (x,y) pts forming one continuous pen-down stroke
    segments: list = []   # list of np arrays, one per continuous stroke
    current_seg: list = []

    for x, y, t, is_lift in tagged_path:
        if is_lift:
            if current_seg:
                segments.append(np.array(current_seg, dtype=np.int32))
                current_seg = []
        else:
            current_seg.append([x, y])
    if current_seg:
        segments.append(np.array(current_seg, dtype=np.int32))

    # Map each frame index to a waypoint index
    n_pts = len(tagged_path)
    # t values are in [0,1]; map frame f to waypoint index
    draw_pts_only = [(i, x, y, t) for i, (x, y, t, lft) in enumerate(tagged_path) if not lft]

    white_base = np.full((H, W, 3), 255, dtype=np.uint8)

    # We'll accumulate segments progressively
    # For each frame f, reveal all segments up to t=(f+1)/num_frames
    frames = []
    prev_tip = (tagged_path[0][0], tagged_path[0][1])

    # Precompute segment membership: which t value does each segment end at?
    # We'll draw segments incrementally per frame
    seg_idx_for_t = []  # for each draw_pt, which segment does it belong to?
    seg_pt_idx = []     # and which point index within that segment?
    s_i, p_i = 0, 0
    for x, y, t, is_lift in tagged_path:
        if not is_lift:
            seg_idx_for_t.append((s_i, p_i))
            if s_i < len(segments) and p_i >= len(segments[s_i]) - 1:
                s_i += 1
                p_i = 0
            else:
                p_i += 1
        else:
            seg_idx_for_t.append(None)

    # Line thickness: scale with canvas size
    thickness = max(2, int(min(H, W) * 0.004))

    for f in range(num_frames):
        t_frame = (f + 1) / num_frames

        # Binary search: find last waypoint index with t <= t_frame
        lo, hi = 0, n_pts - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tagged_path[mid][2] <= t_frame:
                lo = mid
            else:
                hi = mid - 1
        frontier_idx = lo

        # Build canvas: white + all completed segments up to frontier
        canvas = white_base.copy()

        # Collect all complete segments and the partial current segment
        completed_segs = []
        partial_seg    = None
        pts_drawn = 0
        for i, (x, y, t, is_lift) in enumerate(tagged_path[:frontier_idx+1]):
            if is_lift:
                if partial_seg and len(partial_seg) > 1:
                    completed_segs.append(np.array(partial_seg, dtype=np.int32))
                partial_seg = []
                continue
            if partial_seg is None:
                partial_seg = []
            partial_seg.append([x, y])

        if partial_seg and len(partial_seg) > 1:
            # The last partial segment is the "current" one being drawn
            completed_segs.append(np.array(partial_seg, dtype=np.int32))

        # Draw all accumulated strokes
        for seg in completed_segs:
            # Sample color from midpoint of this segment
            mid_pt = seg[len(seg)//2]
            color = _sample_color(mid_pt[0], mid_pt[1])
            cv2.polylines(canvas, [seg.reshape(-1,1,2)], False,
                          color, thickness, cv2.LINE_AA)

        # Hand tip position
        tip_x, tip_y, is_lift = _get_hand_tip_for_frame(
            tagged_path, t_frame, prev_tip, smoothing=0.35)
        prev_tip = (tip_x, tip_y)

        tip_x = max(0, min(W-1, tip_x))
        tip_y = max(0, min(H-1, tip_y))

        hand_opacity = _PEN_LIFT_OPACITY if is_lift else 1.0
        frame = _draw_hand_marker(canvas, tip_x, tip_y, opacity=hand_opacity)
        frames.append(frame)

    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  8.  PHASE 2 — COLOR FILL WITH HAND  (v5.0)
# ══════════════════════════════════════════════════════════════════════════════

def _phase2_color_frames(
    outline_canvas: np.ndarray,
    full_canvas: np.ndarray,
    num_frames: int,
    tagged_path: list = None,
    reveal_order: np.ndarray = None,
) -> list:
    """
    Phase 2 — Color Fill Phase (v5.0).

    The marker re-traces the drawing path (same left-to-right stroke order)
    while the original image colors bleed in behind it — as if the hand is
    now coloring in what it just outlined.  The hand stays fully visible.
    """
    H, W = outline_canvas.shape[:2]

    # Organic bleed offset so color seeps in unevenly (ink-into-paper feel)
    rng = np.random.default_rng(seed=7)
    bleed = rng.random((H, W)).astype(np.float32)
    bleed = cv2.GaussianBlur(bleed, (31, 31), 0)
    lo, hi = bleed.min(), bleed.max()
    if hi > lo:
        bleed = (bleed - lo) / (hi - lo)

    outline_f = outline_canvas.astype(np.float32)
    full_f    = full_canvas.astype(np.float32)

    # Starting tip: last draw position from phase 1
    prev_tip = (W // 2, H // 2)
    if tagged_path:
        draw_pts = [(x,y) for x,y,_t,lft in tagged_path if not lft]
        if draw_pts:
            prev_tip = draw_pts[-1]

    frames = []
    for f in range(num_frames):
        gt = (f + 1) / num_frames

        # Color fill: globally advancing alpha with per-pixel organic offset
        pixel_alpha = np.clip((gt - bleed * 0.50) / 0.50, 0.0, 1.0).astype(np.float32)
        alpha3 = pixel_alpha[:, :, np.newaxis]
        frame  = (full_f * alpha3 + outline_f * (1.0 - alpha3)).astype(np.uint8)

        # Hand retrace: follow the same path at the color frontier
        if tagged_path:
            tip_x, tip_y, is_lift = _get_hand_tip_for_frame(
                tagged_path, gt, prev_tip, smoothing=0.30)
            prev_tip = (tip_x, tip_y)
            tip_x = max(0, min(W-1, tip_x))
            tip_y = max(0, min(H-1, tip_y))
            frame = _draw_hand_marker(frame, tip_x, tip_y, opacity=0.90)

        frames.append(frame)

    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  9.  IMAGE CONVERSION INTERFACE ENTRYPOINT  (v5.0)
# ══════════════════════════════════════════════════════════════════════════════

def image_to_frames(
    pil_image: Image.Image,
    canvas_size=(1280, 720),
    num_frames: int = 120,
    fps: int = 24,
    hold_seconds: float = 1.0,
) -> list:
    """
    Convert a single PIL image into a list of whiteboard animation frames.

    Phase 1 (50 % of frames): Master-slave draw reveal — the hand traces
        each stroke from the skeleton path, and lines appear ONLY where the
        nib is currently touching.  Velocity-based frame allocation keeps the
        marker moving at a visually uniform speed.
    Phase 2 (50 % of frames): Color fill — hand re-traces while original
        colors bleed in organically behind it.
    Hold (hold_seconds × fps): final frame frozen.
    """
    full_canvas = _prepare_canvas_image(pil_image, canvas_size)
    outline_canvas, edge_binary = _extract_outline_canvas(full_canvas)

    # Build reveal order (spatial, for phase 2 bleed direction compatibility)
    reveal_order = _build_reveal_order(edge_binary)

    phase1_frames = max(1, num_frames // 2)
    phase2_frames = max(1, num_frames - phase1_frames)

    p1 = _phase1_outline_frames(
        outline_canvas, reveal_order, edge_binary, phase1_frames)

    # Reuse the tagged path so phase 2 hand follows the same route
    tagged_path = _build_contour_drawing_path(edge_binary)
    p2 = _phase2_color_frames(
        outline_canvas, full_canvas, phase2_frames,
        tagged_path=tagged_path, reveal_order=reveal_order)

    hold_count  = max(1, int(round(hold_seconds * fps)))
    hold_frames = [full_canvas.copy()] * hold_count

    return p1 + p2 + hold_frames

# ══════════════════════════════════════════════════════════════════════════════
#  10. VOICEOVER GENERATION (edge-tts)
# ══════════════════════════════════════════════════════════════════════════════

async def _generate_tts_async(text: str, output_path: str, voice: str, metadata_path: str | None = None) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    if metadata_path is not None:
        await communicate.save(output_path, metadata_path)
    else:
        await communicate.save(output_path)


def generate_voiceover(text: str, output_path: str, voice: str = "en-US-JennyNeural") -> None:
    """Generate an MP3 voiceover using edge-tts."""
    loop = get_or_create_event_loop()
    loop.run_until_complete(_generate_tts_async(text, output_path, voice, metadata_path=None))


def generate_voiceover_with_timestamps(
    text: str,
    output_path: str,
    metadata_path: str,
    voice: str = "en-US-JennyNeural",
) -> None:
    """Generate a voiceover and accompanying edge-tts metadata word timings."""
    loop = get_or_create_event_loop()
    loop.run_until_complete(_generate_tts_async(text, output_path, voice, metadata_path=metadata_path))


def _parse_edge_tts_word_metadata(metadata_path: str) -> list[dict]:
    events = []
    if not os.path.exists(metadata_path):
        return events

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "WordBoundary":
                continue
            offset_ticks = obj.get("offset", 0)
            duration_ticks = obj.get("duration", 0)
            events.append(
                {
                    "text": obj.get("text", "").strip(),
                    "start": offset_ticks / 10_000_000,
                    "duration": duration_ticks / 10_000_000,
                }
            )
    return events


def _build_caption_segments(
    word_events: list[dict],
    max_chars: int = 40,
    max_duration: float = 4.0,
    max_words: int = 8,
) -> list[dict]:
    segments = []
    current = None

    for word in word_events:
        if not word["text"]:
            continue

        word_text = word["text"].strip()
        if not word_text:
            continue

        if current is None:
            current = {
                "text": word_text,
                "start": word["start"],
                "end": word["start"] + max(word["duration"], 0.1),
                "word_count": 1,
            }
            continue

        next_text = f"{current['text']} {word_text}"
        next_duration = (word["start"] + max(word["duration"], 0.1)) - current["start"]

        if (
            len(next_text) > max_chars
            or next_duration > max_duration
            or current["word_count"] >= max_words
        ):
            segments.append({
                "text": current["text"],
                "start": current["start"],
                "end": current["end"],
            })
            current = {
                "text": word_text,
                "start": word["start"],
                "end": word["start"] + max(word["duration"], 0.1),
                "word_count": 1,
            }
        else:
            current["text"] = next_text
            current["end"] = word["start"] + max(word["duration"], 0.1)
            current["word_count"] += 1

    if current is not None:
        segments.append({
            "text": current["text"],
            "start": current["start"],
            "end": current["end"],
        })

    return segments


def _build_caption_segments_from_original(
    word_events: list[dict],
    original_text: str,
    max_chars: int = 40,
    max_duration: float = 4.0,
    max_words: int = 8,
) -> list[dict]:
    """
    Reconstruct caption segments using the original scene `original_text` to
    preserve punctuation. Maps original word tokens to edge-tts word events by
    order and keeps the timing from `word_events`.
    """
    if not word_events:
        return []

    import re

    # Tokenize original text into words and punctuation tokens
    tokens = re.findall(r"\w+|[^\w\s]+", original_text)
    orig_words = []
    puncts = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.match(r"\w+", tok):
            word = tok
            punct = ""
            if i + 1 < len(tokens) and not re.match(r"\w+", tokens[i + 1]):
                punct = tokens[i + 1]
                i += 1
            orig_words.append(word)
            puncts.append(punct)
        i += 1

    # Rebuild word events but replace text with the original word + trailing punctuation
    new_events = []
    for idx, ev in enumerate(word_events):
        text = ev.get("text", "").strip()
        if idx < len(orig_words):
            text_repl = orig_words[idx] + (puncts[idx] if puncts[idx] else "")
        else:
            text_repl = text
        new_events.append({
            "text": text_repl,
            "start": ev.get("start", 0),
            "duration": ev.get("duration", 0),
        })

    # Delegate to the existing segmenting logic
    return _build_caption_segments(new_events, max_chars=max_chars, max_duration=max_duration, max_words=max_words)


def _offset_caption_segments(caption_segments: list[dict], offset: float) -> list[dict]:
    return [
        {
            "text": segment["text"],
            "start": segment["start"] + offset,
            "end": segment["end"] + offset,
        }
        for segment in caption_segments
    ]


def _render_caption_image(text: str, canvas_size: tuple, font_size: int = 38) -> np.ndarray:
    width, height = canvas_size
    max_width = int(width * 0.88)
    lines = textwrap.wrap(text, width=28)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    measure_image = Image.new("RGBA", (1, 1))
    measure_draw = ImageDraw.Draw(measure_image)

    def _text_size(line: str) -> tuple[int, int]:
        if hasattr(font, "getbbox"):
            x0, y0, x1, y1 = font.getbbox(line)
            return x1 - x0, y1 - y0
        return measure_draw.textsize(line, font=font)

    line_sizes = [_text_size(line) for line in lines]
    line_heights = [size[1] for size in line_sizes]
    text_height = sum(line_heights) + (len(lines) - 1) * 8
    text_width = max((size[0] for size in line_sizes), default=0)
    padding = 18
    img_h = text_height + padding * 2
    img_w = min(max_width, text_width + padding * 2)

    image = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Background highlight box
    box_color = (0, 0, 0, 180)
    draw.rounded_rectangle((0, 0, img_w, img_h), radius=18, fill=box_color)

    y = padding
    for line, (line_w, line_h) in zip(lines, line_sizes):
        x = (img_w - line_w) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h + 8

    return np.array(image)


def _overlay_caption_clips(video_clip, caption_segments, canvas_size):
    try:
        if _MOVIEPY_V2:
            from moviepy import CompositeVideoClip, ImageClip
        else:
            from moviepy.editor import CompositeVideoClip, ImageClip
    except Exception:
        return video_clip

    caption_clips = []
    for segment in caption_segments:
        duration = max(0.1, segment["end"] - segment["start"])
        caption_image = _render_caption_image(segment["text"], canvas_size)
        caption = _clip_set_duration(ImageClip(caption_image), duration)
        caption = _clip_set_position(caption, ("center", "bottom"))
        if hasattr(caption, "set_start"):
            caption = caption.set_start(segment["start"])
        else:
            caption = caption.with_start(segment["start"])
        caption_clips.append(caption)

    if not caption_clips:
        return video_clip

    composite = CompositeVideoClip([video_clip, *caption_clips], size=canvas_size)
    return composite


def get_audio_duration(audio_path: str) -> float:
    """Return the duration of an audio file in seconds using moviepy."""
    clip = AudioFileClip(audio_path)
    duration = clip.duration
    clip.close()
    return duration


def get_scene_duration(
    text: str | None = None,
    custom_audio_path: str | None = None,
    custom_audio_file=None,
    wpm: float = 140.0,
    min_seconds: float = 1.0,
) -> float:
    """Return the duration for a scene.

    If a custom audio path or uploaded custom audio file is provided, return the
    actual duration of that audio. Otherwise estimate duration from word count.
    """
    if custom_audio_path is not None:
        return get_audio_duration(custom_audio_path)

    if custom_audio_file is not None:
        if hasattr(custom_audio_file, "seek"):
            custom_audio_file.seek(0)
        suffix = os.path.splitext(getattr(custom_audio_file, "name", ""))[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(custom_audio_file.read())
            tmp_path = tmp.name
        try:
            return get_audio_duration(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if not text:
        return min_seconds

    import re

    words = re.findall(r"\w+", text)
    duration_seconds = max(min_seconds, (len(words) / wpm) * 60.0)
    return duration_seconds


# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _generate_voice_preview(voice_id: str) -> bytes:
    """Generate a short MP3 preview for a voice. Cached per session."""
    preview_text = "Hello! I could be a great voice for your whiteboard video."
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        generate_voiceover(preview_text, tmp_path, voice=voice_id)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


#  11. BACKGROUND MUSIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "music")


def get_available_music_tracks() -> dict:
    """Scan the assets/music directory for audio tracks and return a name -> path map."""
    tracks = {}
    if not os.path.exists(_MUSIC_DIR):
        try:
            os.makedirs(_MUSIC_DIR, exist_ok=True)
        except Exception:
            return tracks

    supported = (".mp3", ".wav", ".m4a", ".aac", ".ogg")
    try:
        fnames = os.listdir(_MUSIC_DIR)
    except Exception:
        return tracks

    for fname in fnames:
        if os.path.splitext(fname)[1].lower() in supported:
            full_path = os.path.join(_MUSIC_DIR, fname)
            # Strip extension, replace underscores/hyphens with spaces, title-case
            stem = os.path.splitext(fname)[0]
            title = stem.replace("_", " ").replace("-", " ").title()
            tracks[title] = full_path
    return tracks


@st.cache_data(show_spinner=False)
def _create_audio_preview_data(track_path: str, preview_seconds: float = 5.0) -> tuple[str, str]:
    """Generate a short base64-encoded MP3 preview for the given music file."""
    clip = AudioFileClip(track_path)
    duration = min(preview_seconds, clip.duration)
    # moviepy 2.x uses .subclipped(); 1.x uses .subclip()
    if hasattr(clip, "subclipped"):
        preview_clip = clip.subclipped(0, duration)
    else:
        preview_clip = clip.subclip(0, duration)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        temp_preview_path = tmp_file.name

    try:
        preview_clip.write_audiofile(
            temp_preview_path,
            logger=None,
            verbose=False,
            bitrate="96k",
        )
        with open(temp_preview_path, "rb") as f:
            encoded_preview = base64.b64encode(f.read()).decode("ascii")
    finally:
        preview_clip.close()
        clip.close()
        if os.path.exists(temp_preview_path):
            os.remove(temp_preview_path)

    return encoded_preview, "audio/mpeg"


def build_music_selector_html(music_tracks: dict, current_selection: str = "None") -> str:
    """
    Render a unified music selector widget that combines track selection with
    hover-to-preview audio playback.

    Hovering a row auto-plays a short clip; moving away stops it.
    Clicking a row selects that track (highlighted in blue) and also
    writes the choice into a hidden <input> so the Streamlit component
    value can be read back via st.session_state after a re-run.

    The widget communicates the user's click back to Streamlit by
    setting window.parent.__streamlit_music_selection and dispatching a
    custom event, which the sidebar code reads from st.session_state via
    a small JS→Python bridge using st.components.v1.html + a key query.
    """
    preview_map = {}
    for title, path in sorted(music_tracks.items()):
        try:
            preview_b64, mime_type = _create_audio_preview_data(path)
            preview_map[title] = {"data": preview_b64, "mime": mime_type}
        except Exception:
            continue

    if not preview_map:
        return "<div style='color:#888;font-size:0.9rem;'>No music tracks found in assets/music/.</div>"

    tracks_json = json.dumps(preview_map)
    all_titles_json = json.dumps(["None"] + list(preview_map.keys()))
    current_json = json.dumps(current_selection)

    return f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; background: transparent; }}

  #music-selector {{
    border: 1px solid #d1d8e0;
    border-radius: 8px;
    overflow: hidden;
    background: #fff;
  }}

  /* ── "None" row at the top ── */
  #none-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid #e8ecf0;
    cursor: pointer;
    transition: background 0.15s;
    font-size: 0.88rem;
    color: #555;
  }}
  #none-row:hover {{ background: #f5f7fa; }}
  #none-row.selected {{ background: #dbeafe; font-weight: 600; color: #1d4ed8; }}

  /* ── Track rows ── */
  .track-row {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid #e8ecf0;
    cursor: pointer;
    transition: background 0.15s;
    position: relative;
  }}
  .track-row:last-child {{ border-bottom: none; }}
  .track-row:hover {{ background: #eff6ff; }}
  .track-row.selected {{ background: #dbeafe; }}
  .track-row.playing {{ background: #e0f2fe; }}

  .track-icon {{ font-size: 1rem; flex-shrink: 0; width: 1.25rem; text-align: center; }}
  .track-name {{ flex: 1; font-size: 0.88rem; color: #1e293b; }}
  .track-row.selected .track-name {{ font-weight: 600; color: #1d4ed8; }}

  /* Animated sound-wave bars shown while hovering/playing */
  .sound-wave {{
    display: none;
    align-items: flex-end;
    gap: 2px;
    height: 14px;
    flex-shrink: 0;
  }}
  .track-row.playing .sound-wave {{ display: flex; }}
  .sound-wave span {{
    display: block;
    width: 3px;
    border-radius: 2px;
    background: #3b82f6;
    animation: bounce 0.6s ease infinite alternate;
  }}
  .sound-wave span:nth-child(1) {{ height: 4px;  animation-delay: 0s;    }}
  .sound-wave span:nth-child(2) {{ height: 10px; animation-delay: 0.1s;  }}
  .sound-wave span:nth-child(3) {{ height: 7px;  animation-delay: 0.2s;  }}
  .sound-wave span:nth-child(4) {{ height: 12px; animation-delay: 0.05s; }}
  .sound-wave span:nth-child(5) {{ height: 5px;  animation-delay: 0.15s; }}
  @keyframes bounce {{
    from {{ transform: scaleY(0.4); }}
    to   {{ transform: scaleY(1.0); }}
  }}

  /* Hover tooltip */
  .hover-tip {{
    display: none;
    position: absolute;
    right: 0.6rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.72rem;
    color: #64748b;
    pointer-events: none;
  }}
  .track-row:hover:not(.playing) .hover-tip {{ display: block; }}

  /* Mini waveform progress bar while previewing */
  .track-row.playing::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    height: 2px;
    background: #3b82f6;
    animation: progress-bar 5s linear forwards;
  }}
  @keyframes progress-bar {{ from {{ width: 0%; }} to {{ width: 100%; }} }}

  /* Caption */
  #caption {{
    font-size: 0.78rem;
    color: #64748b;
    padding: 0.4rem 0.75rem 0.5rem;
    border-top: 1px solid #e8ecf0;
    background: #f8fafc;
    text-align: center;
  }}

  /* Hidden value output */
  #selected-value {{ display: none; }}
</style>

<div id="music-selector">
  <!-- None option -->
  <div id="none-row" onclick="selectTrack('None', this)">
    <span style="font-size:1rem;">🚫</span>
    <span style="flex:1;font-size:0.88rem;">None</span>
  </div>

  <!-- Track rows injected by JS -->
  <div id="track-list"></div>

  <div id="caption">🎵 Hover to preview · Click to select</div>
</div>

<audio id="preview-player" preload="none"></audio>
<input id="selected-value" type="text" value={current_json} />

<script>
const previewTracks = {tracks_json};
const allTitles    = {all_titles_json};
const player       = document.getElementById('preview-player');
const trackList    = document.getElementById('track-list');
const noneRow      = document.getElementById('none-row');
const hiddenInput  = document.getElementById('selected-value');

let currentlyPlaying = null;   // title of the row whose audio is playing
let selectedTitle    = {current_json};
let hoverTimer       = null;   // delay before starting preview

// ── Build track rows ─────────────────────────────────────────────────────────
Object.keys(previewTracks).forEach(title => {{
  const row = document.createElement('div');
  row.className = 'track-row' + (title === selectedTitle ? ' selected' : '');
  row.dataset.title = title;
  row.innerHTML = `
    <span class="track-icon">🎵</span>
    <span class="track-name">${{escHtml(title)}}</span>
    <div class="sound-wave">
      <span></span><span></span><span></span><span></span><span></span>
    </div>
    <span class="hover-tip">hover to preview</span>
  `;
  row.addEventListener('mouseenter', () => onHoverEnter(title, row));
  row.addEventListener('mouseleave', () => onHoverLeave(row));
  row.addEventListener('click',      () => selectTrack(title, row));
  trackList.appendChild(row);
}});

// Mark "None" selected if it's the current value
if (selectedTitle === 'None') noneRow.classList.add('selected');

// ── Hover: play after a 300 ms delay so fast mouse-overs don't trigger audio ─
function onHoverEnter(title, row) {{
  hoverTimer = setTimeout(() => playPreview(title, row), 300);
}}

function onHoverLeave(row) {{
  clearTimeout(hoverTimer);
  stopPreview(row);
}}

function playPreview(title, row) {{
  const track = previewTracks[title];
  if (!track) return;

  // Stop any currently playing row
  if (currentlyPlaying && currentlyPlaying !== title) {{
    const prev = trackList.querySelector(`[data-title="${{currentlyPlaying}}"]`);
    if (prev) prev.classList.remove('playing');
  }}

  const src = `data:${{track.mime}};base64,${{track.data}}`;
  if (player.dataset.src !== src) {{
    player.pause();
    player.src = src;
    player.dataset.src = src;
    player.load();
  }}
  player.currentTime = 0;
  player.play().catch(() => {{}});

  row.classList.add('playing');
  currentlyPlaying = title;
}}

function stopPreview(row) {{
  clearTimeout(hoverTimer);
  player.pause();
  player.currentTime = 0;
  row.classList.remove('playing');
  if (currentlyPlaying === row.dataset.title) currentlyPlaying = null;
}}

// ── Click: select a track ────────────────────────────────────────────────────
function selectTrack(title, clickedRow) {{
  selectedTitle = title;
  hiddenInput.value = title;

  // Update visual selection
  document.querySelectorAll('.track-row').forEach(r => r.classList.remove('selected'));
  noneRow.classList.remove('selected');
  clickedRow.classList.add('selected');

  // Notify Streamlit via query param trick: update URL hash so the parent
  // can read it. Also push to window.name as a simple cross-frame channel.
  try {{
    window.parent.postMessage({{ type: 'music_selection', value: title }}, '*');
  }} catch(e) {{}}

  // Flash caption to confirm
  const cap = document.getElementById('caption');
  cap.textContent = title === 'None' ? '🚫 No music selected' : `✅ Selected: ${{title}}`;
  setTimeout(() => {{ cap.textContent = '🎵 Hover to preview · Click to select'; }}, 2000);
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}
</script>
"""


def build_music_hover_script(music_tracks: dict) -> str:
    """
    Returns a <script> tag (safe for st.markdown unsafe_allow_html=True) that:
      1. Bakes all audio previews as base64 data URIs into JS.
      2. Uses a MutationObserver to watch for Streamlit's dropdown <li> elements
         (class "option" inside a [data-baseweb="menu"]).
      3. Attaches mouseenter → play preview, mouseleave → stop on each option
         whose text matches a known track title.
    This works because st.markdown injects content into the same DOM as the
    Streamlit app itself (unlike components.html which is a sandboxed iframe).
    """
    preview_map = {}
    for title, path in sorted(music_tracks.items()):
        try:
            preview_b64, mime_type = _create_audio_preview_data(path)
            preview_map[title] = {"data": preview_b64, "mime": mime_type}
        except Exception:
            continue

    if not preview_map:
        return ""

    tracks_json = json.dumps(preview_map)

    return f"""
<script>
(function() {{
  const TRACKS = {tracks_json};
  let player = null;
  let hoverTimer = null;

  function getOrCreatePlayer() {{
    if (!player) {{
      player = document.createElement('audio');
      player.id = '__wb_music_preview__';
      player.preload = 'none';
      document.body.appendChild(player);
    }}
    return player;
  }}

  function playPreview(title) {{
    const track = TRACKS[title];
    if (!track) return;
    const p = getOrCreatePlayer();
    const src = `data:${{track.mime}};base64,${{track.data}}`;
    if (p.dataset.src !== src) {{
      p.pause();
      p.src = src;
      p.dataset.src = src;
      p.load();
    }}
    p.currentTime = 0;
    p.play().catch(() => {{}});
  }}

  function stopPreview() {{
    if (player) {{ player.pause(); player.currentTime = 0; }}
  }}

  function attachHoverToOption(el) {{
    if (el.dataset.wbBound) return;
    el.dataset.wbBound = '1';
    const title = el.innerText.trim();
    if (!TRACKS[title]) return;   // skip "None" and unknown options
    el.addEventListener('mouseenter', () => {{
      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(() => playPreview(title), 250);
    }});
    el.addEventListener('mouseleave', () => {{
      clearTimeout(hoverTimer);
      stopPreview();
    }});
    // Visual hint
    el.title = '\u25b6\ufe0f Hover to preview';
  }}

  // MutationObserver: fires whenever Streamlit mounts the dropdown menu
  const observer = new MutationObserver((mutations) => {{
    for (const m of mutations) {{
      for (const node of m.addedNodes) {{
        if (node.nodeType !== 1) continue;
        // Streamlit renders dropdown items as <li role="option"> inside
        // a div[data-baseweb="menu"]
        const menu = node.matches('[data-baseweb="menu"]')
          ? node
          : node.querySelector('[data-baseweb="menu"]');
        if (menu) {{
          menu.querySelectorAll('li').forEach(attachHoverToOption);
        }}
      }}
    }}
  }});

  observer.observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
"""



def _mix_background_music(
    voiceover_clip,
    music_path: str,
    volume_pct: float,
    final_duration: float,
):
    """
    Load a background music track, adjust its volume, loop it to cover `final_duration` seconds,
    trim it precisely to `final_duration`, then composite it underneath the voiceover clip.
    Args:
        voiceover_clip : moviepy AudioFileClip for the primary narration.
        music_path     : absolute path to the selected music file.
        volume_pct     : volume as a percentage (0–50). Converted to a 0–0.5 multiplier.
        final_duration  : exact duration (seconds) of the finished video.
    Returns:
        A CompositeAudioClip with music underneath the voiceover.
    """
    try:
        from moviepy import CompositeAudioClip
    except ImportError:
        from moviepy.editor import CompositeAudioClip

    volume_factor = volume_pct / 100.0  # e.g. 10% -> 0.10
    music_raw = AudioFileClip(music_path)

    # ── Loop the music if it is shorter than the video ────────────────────────
    if music_raw.duration < final_duration:
        loops_needed = int(np.ceil(final_duration / music_raw.duration))
        try:
            # moviepy 2.x
            music_looped = music_raw.with_effects(
                [__import__("moviepy", fromlist=["audio"]).audio.fx.AudioLoop(nloops=loops_needed)]
            )
        except Exception:
            # Fallback: manually concatenate copies
            try:
                from moviepy import concatenate_audioclips
            except ImportError:
                from moviepy.editor import concatenate_audioclips
            music_looped = concatenate_audioclips([music_raw] * loops_needed)
    else:
        music_looped = music_raw

    # ── Trim precisely to the video's end frame ───────────────────────────────
    music_trimmed = _clip_set_duration(music_looped, final_duration)

    # ── Apply volume safely across all MoviePy versions ────────────────────────
    try:
        if hasattr(music_trimmed, "with_volume_scaled"):
            music_ducked = music_trimmed.with_volume_scaled(volume_factor)
        elif hasattr(music_trimmed, "multiply_volume"):
            music_ducked = music_trimmed.multiply_volume(volume_factor)
        else:
            music_ducked = music_trimmed.volumex(volume_factor)
    except Exception:
        try:
            from moviepy.audio.fx.volumex import volumex
            music_ducked = music_trimmed.fx(volumex, volume_factor)
        except Exception:
            # Emergency safety catch: keep audio going un-ducked if everything else crashes
            music_ducked = music_trimmed

    # ── Mix: music underneath the voiceover ──────────────────────────────────
    mixed = CompositeAudioClip([music_ducked, voiceover_clip])
    return mixed


# ══════════════════════════════════════════════════════════════════════════════
#  12. FINAL COMPOSITION (moviepy)
# ══════════════════════════════════════════════════════════════════════════════

def compose_video(
    ordered_images,
    audio_path: str,
    output_path: str,
    canvas_size=(1280, 720),
    fps: int = 24,
    hold_seconds: float = 1.0,
    progress_callback=None,
    per_scene_durations: list | None = None,
    caption_segments: list[dict] | None = None,
    music_path: str | None = None,
    music_volume_pct: float = 10.0,
    custom_audio_path: str | None = None,
) -> None:
    """
    1. Calculate total audio duration.
    2. Split time proportionally across all source images.
    3. Generate whiteboards-sketch animations for each image.
    4. Mix down background ambient tracks (looped/trimmed to match
       exact video length) underneath the voiceover.

    If a custom uploaded audio file is provided, it will be loaded directly
    to preserve its physical duration and sync with the rendered frames.
    """
    n_images = len(ordered_images)
    # If caller provides per-scene durations, use them. Otherwise split audio evenly.
    audio_source_path = custom_audio_path if custom_audio_path else audio_path
    if isinstance(per_scene_durations, (list, tuple)) and len(per_scene_durations) == n_images:
        scene_durations = list(per_scene_durations)
        audio_duration = sum(scene_durations)
    else:
        audio_duration = get_audio_duration(audio_source_path)
        scene_durations = [audio_duration / max(1, n_images)] * n_images

    clips = []
    for idx, pil_img in enumerate(ordered_images):
        if progress_callback:
            progress_callback(idx, n_images)

        time_per_image = scene_durations[idx]
        total_frames = max(30, int(time_per_image * fps))
        # Subtract hold frames from the animation budget so the scene
        # still fits exactly within time_per_image seconds.
        hold_count = max(1, int(round(hold_seconds * fps)))
        anim_frames = max(10, total_frames - hold_count)

        rgb_frames = image_to_frames(
            pil_img,
            canvas_size=canvas_size,
            num_frames=anim_frames,
            fps=fps,
            hold_seconds=hold_seconds,
        )
        clip = ImageSequenceClip(rgb_frames, fps=fps)
        clip = _clip_set_duration(clip, time_per_image)
        clips.append(clip)

    # Concatenate all scene clips
    video = concatenate_videoclips(clips, method="compose")

    # Attach voiceover — use custom uploaded audio when available.
    audio_source_path = custom_audio_path if custom_audio_path else audio_path
    voiceover = AudioFileClip(audio_source_path)
    final_duration = min(video.duration, voiceover.duration)

    video = _clip_set_duration(video, final_duration)
    voiceover = _clip_set_duration(voiceover, final_duration)

    if caption_segments:
        video = _overlay_caption_clips(video, caption_segments, canvas_size)

    # ── Optional: mix background music underneath the voiceover ───────────────
    if music_path:
        final_audio = _mix_background_music(
            voiceover_clip=voiceover,
            music_path=music_path,
            volume_pct=music_volume_pct,
            final_duration=final_duration,
        )
    else:
        final_audio = voiceover

    final = _clip_set_audio(video, final_audio)

    # Write output MP4
    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=output_path + ".temp_audio.m4a",
        remove_temp=True,
        logger=None,
    )

    final.close()
    voiceover.close()
    video.close()


# ══════════════════════════════════════════════════════════════════════════════
#  12. STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/whiteboard.png", width=80)
        st.title("Powers Video Creation App")
        st.markdown("**Version 4.0 — Professional Vector-Style Path Drawing**")
        st.markdown("---")

        st.markdown("### Settings")
        canvas_w = st.number_input("Canvas Width (px)", value=1280, step=64, min_value=320)
        canvas_h = st.number_input("Canvas Height (px)", value=720, step=64, min_value=240)
        fps_val = st.slider("Frames per Second (FPS)", min_value=12, max_value=60, value=24, step=1)
        hold_sec = st.slider("Freeze Final Frame (seconds)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

        st.markdown("---")
        st.markdown("### 🎵 Background Music")

        music_tracks = get_available_music_tracks()

        # Initialize session state for music selection
        if "selected_music_path" not in st.session_state:
            st.session_state.selected_music_path = None

        if music_tracks:
            st.write("**Listen & Choose a Track:**")
            for track_name, track_path in music_tracks.items():
                col1, col2, col3 = st.columns([1.2, 2.5, 3])
                with col1:
                    is_selected = st.session_state.selected_music_path == track_path
                    btn_label = "✅ Selected" if is_selected else "Select"
                    btn_type = "primary" if is_selected else "secondary"
                    if st.button(btn_label, key=f"select_{track_name}", type=btn_type, use_container_width=True):
                        st.session_state.selected_music_path = track_path
                        st.rerun()
                with col2:
                    st.markdown(f"<div style='margin-top:8px'><b>{track_name}</b></div>", unsafe_allow_html=True)
                with col3:
                    st.audio(track_path)

            # Show current selection and clear button
            current_name = next((n for n, p in music_tracks.items() if p == st.session_state.selected_music_path), "None")
            st.info(f"🎵 **Selected:** {current_name}")
            if st.session_state.selected_music_path is not None:
                if st.button("🚫 Clear Music", use_container_width=True):
                    st.session_state.selected_music_path = None
                    st.rerun()
        else:
            st.caption("No tracks found in assets/music/ — add .mp3/.wav files there to enable background music.")

        selected_music_path = st.session_state.selected_music_path

        music_volume_pct = 10.0
        if selected_music_path is not None:
            music_volume_pct = st.slider(
                "Music Volume (%)",
                min_value=0,
                max_value=50,
                value=10,
                step=1,
                help=(
                    "Auto-ducking level — keeps the music quietly in the "
                    "background so it never drowns out the narrator. "
                    "10 % is a good starting point."
                ),
            )

        st.markdown("---")
        st.markdown("### 🎤 Narration Source")
        voice_over_source = st.radio(
            "Voice Over Source",
            ["AI Voice Narrator", "Upload Custom Audio Clip (MP4/WAV)"],
            index=0,
            key="voice_over_source",
        )

        custom_audio_file = None
        if voice_over_source == "Upload Custom Audio Clip (MP4/WAV)":
            custom_audio_file = st.file_uploader(
                "Upload your custom narration",
                type=["mp3", "wav"],
                key="custom_audio_file",
                help=(
                    "Upload a recorded narration file here. "
                    "This audio will be used instead of the AI voiceover."
                ),
            )
            if custom_audio_file is not None:
                st.success(f"Custom narration ready: {custom_audio_file.name}")
            else:
                st.info("Upload an MP3 or WAV narration clip to enable custom audio.")
        else:
            st.info("Use the AI voice options in Step 2 to choose your narrator.")

        st.markdown("---")
        st.markdown("#### 🎨 How it works")
        st.markdown(
            "**Phase 1 — Outline (50 %):**\n\n"
            "• **Skeleton Center-Line:** Strokes are morphologically thinned to "
            "their single-pixel spine so the hand traces the true center of each "
            "letter and line — fluid single-stroke movements.\n\n"
            "• **Hierarchical Cluster Sort:** Contours are grouped into structural "
            "elements (a face, a flower cluster, etc.). The hand completes one "
            "entire element before jumping to the next nearest cluster.\n\n"
            "• **Natural Pen Lift:** Inter-cluster jumps trigger a pen-lift — the "
            "hand fades to 40 % opacity and accelerates across empty space, "
            "mimicking realistic human sketching cadence.\n\n"
            "**Phase 2 — Color (50 %):** The full artwork — shading, textures, "
            "colors — bleeds organically into the outline.\n\n"
            "**Hold:** Final frame freezes for the last second."
        )
        st.markdown("---")
        st.caption("Powered by OpenCV · edge-tts · MoviePy")

    canvas_size = (int(canvas_w), int(canvas_h))

    # ── Main area ─────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;'>🎨 Powers Video Creation App</h1>"
        "<p style='text-align:center; color:grey;'>"
        "<b>Version 4.0 — Professional Vector-Style Path Drawing</b><br>"
        "Skeleton Center-Line · Hierarchical Cluster Sort · Natural Pen Lift · "
        "Outline → Color bleed → Final hold · Audio perfectly synced"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Step 1: Upload images ─────────────────────────────────────────────────
    st.header("Step 1 — Upload Sketch Images")
    uploaded_files = st.file_uploader(
        "Upload one or more sketch images (PNG, JPG, BMP, WEBP)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
        help=(
            "Upload the artwork you want to animate. "
            "Phase 1 will trace every stroke center-line."
        ),
    )

    ordered_images = []
    ordered_names = []

    if uploaded_files:
        st.markdown("#### Arrange Image Rendering Order")
        cols_per_row = 4
        order_inputs = {}

        rows = [
            uploaded_files[i: i + cols_per_row]
            for i in range(0, len(uploaded_files), cols_per_row)
        ]

        for row in rows:
            cols = st.columns(cols_per_row)
            for col, uf in zip(cols, row):
                with col:
                    img = Image.open(uf)
                    st.image(img, caption=uf.name, use_container_width=True)
                    order_val = st.number_input(
                        f"Order for **{uf.name}**",
                        min_value=1,
                        max_value=len(uploaded_files),
                        value=uploaded_files.index(uf) + 1,
                        step=1,
                        key=f"order_{uf.name}",
                    )
                    order_inputs[uf.name] = order_val

        sorted_files = sorted(
            uploaded_files,
            key=lambda f: (order_inputs[f.name], f.name),
        )

        for uf in sorted_files:
            ordered_images.append(Image.open(uf))
            ordered_names.append(uf.name)

        st.success(
            "**Final sequence:** " + " → ".join(f"`{n}`" for n in ordered_names)
        )

    st.markdown("---")

    # ── Step 2: Script / Voiceover ────────────────────────────────────────────
    st.header("Step 2 — Write Your Script")
    voice_over_source = st.session_state.get("voice_over_source", "AI Voice Narrator")

    script_help = (
        "This text will be converted to an AI voiceover using edge-tts."
        if voice_over_source == "AI Voice Narrator"
        else "Optional: use this to document your narration when uploading custom audio."
    )
    script_text = st.text_area(
        "Video narration script",
        height=180,
        placeholder=(
            "Directions: User input script for each scene in this format.\n"
            "1: Text for scene 1\n"
            "2: Text for scene 2\n"
            "3: Text for scene 3\n\n"
            "Use one line per scene. Add text until all scenes have text."
        ),
        help=script_help,
    )

    selected_voice = "en-US-JennyNeural"
    selected_voice_label = st.session_state.get("selected_voice_label", None)

    if voice_over_source == "AI Voice Narrator":
        voice_options = {
            # ── United States ──────────────────────────────────────────────────
            "🇺🇸 Jenny   (US Female · Friendly)":       "en-US-JennyNeural",
            "🇺🇸 Aria    (US Female · Natural)":        "en-US-AriaNeural",
            "🇺🇸 Ana     (US Female · Warm)":           "en-US-AnaNeural",
            "🇺🇸 Michelle(US Female · Professional)":   "en-US-MichelleNeural",
            "🇺🇸 Guy     (US Male   · Confident)":      "en-US-GuyNeural",
            "🇺🇸 Davis   (US Male   · Authoritative)":  "en-US-DavisNeural",
            "🇺🇸 Tony    (US Male   · Energetic)":      "en-US-TonyNeural",
            "🇺🇸 Jason   (US Male   · Deep)":           "en-US-JasonNeural",
            # ── United Kingdom ─────────────────────────────────────────────────
            "🇬🇧 Sonia   (UK Female · Clear)":          "en-GB-SoniaNeural",
            "🇬🇧 Libby   (UK Female · Expressive)":     "en-GB-LibbyNeural",
            "🇬🇧 Maisie  (UK Female · Young)":          "en-GB-MaisieNeural",
            "🇬🇧 Ryan    (UK Male   · Conversational)": "en-GB-RyanNeural",
            "🇬🇧 Thomas  (UK Male   · Distinguished)":  "en-GB-ThomasNeural",
            # ── Australia ──────────────────────────────────────────────────────
            "🇦🇺 Natasha (AU Female · Warm)":           "en-AU-NatashaNeural",
            "🇦🇺 William (AU Male   · Casual)":         "en-AU-WilliamNeural",
            # ── Canada ─────────────────────────────────────────────────────────
            "🇨🇦 Clara   (CA Female · Polished)":       "en-CA-ClaraNeural",
            "🇨🇦 Liam    (CA Male   · Smooth)":         "en-CA-LiamNeural",
            # ── Ireland ────────────────────────────────────────────────────────
            "🇮🇪 Emily   (IE Female · Lyrical)":        "en-IE-EmilyNeural",
            "🇮🇪 Connor  (IE Male   · Relaxed)":        "en-IE-ConnorNeural",
        }

        st.write("**Choose a Voice** — click Preview to hear a sample:")

        if "voice_preview_id" not in st.session_state:
            st.session_state.voice_preview_id = None
        if selected_voice_label is None:
            selected_voice_label = list(voice_options.keys())[0]
            st.session_state.selected_voice_label = selected_voice_label

        prev_flag = None
        for label, voice_id in voice_options.items():
            flag = label[:4].strip()
            if flag != prev_flag:
                st.markdown(
                    f"<div style='font-size:0.78rem;color:#888;margin:6px 0 2px;padding-left:4px;'>{flag}</div>",
                    unsafe_allow_html=True,
                )
                prev_flag = flag

            is_selected = st.session_state.selected_voice_label == label
            col_prev, col_name, col_sel = st.columns([1.1, 3.8, 1.3])

            with col_prev:
                if st.button("▶ Preview", key=f"prev_{voice_id}", use_container_width=True):
                    st.session_state.voice_preview_id = voice_id
                    st.rerun()

            with col_name:
                weight = "700" if is_selected else "400"
                color  = "#1d4ed8" if is_selected else "#1e293b"
                st.markdown(
                    f"<div style='margin-top:6px;font-size:0.87rem;font-weight:{weight};color:{color};'>{label}</div>",
                    unsafe_allow_html=True,
                )

            with col_sel:
                btn_label = "✅ Selected" if is_selected else "Select"
                btn_type  = "primary"   if is_selected else "secondary"
                if st.button(btn_label, key=f"sel_{voice_id}", type=btn_type, use_container_width=True):
                    st.session_state.selected_voice_label = label
                    st.session_state.voice_preview_id = None
                    st.rerun()

            if st.session_state.voice_preview_id == voice_id:
                try:
                    name_short = label.split("(")[0].strip()
                    with st.spinner(f"Generating preview for {name_short}…"):
                        audio_bytes = _generate_voice_preview(voice_id)
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.warning(f"Preview failed: {e}")

        selected_voice_label = st.session_state.selected_voice_label
        selected_voice = voice_options.get(selected_voice_label, "en-US-JennyNeural")
        st.info(f"🎙️ **Active voice:** {selected_voice_label}")
        st.markdown("---")
    else:
        st.info("🎙️ **Using uploaded custom narration**")
        if not st.session_state.get("custom_audio_file"):
            st.warning("Upload a custom audio clip in the sidebar to enable this mode.")
        st.markdown("---")

    # ── Step 3: Compile Video ─────────────────────────────────────────────────
    st.header("Step 3 — Render Whiteboard Animation")

    if st.button("🚀 Generate Video", type="primary", use_container_width=True):
        if not ordered_images:
            st.error("Please upload at least one image in Step 1.")
            return

        voice_over_source = st.session_state.get("voice_over_source", "AI Voice Narrator")
        custom_audio_file = st.session_state.get("custom_audio_file")

        if voice_over_source == "AI Voice Narrator":
            if not script_text.strip():
                st.error("Please provide a narration script in Step 2.")
                return
        else:
            if custom_audio_file is None:
                st.error("Please upload a custom audio clip in the sidebar.")
                return

        try:
            status = st.status("Initializing Whiteboard Studio...", expanded=True)

            with status:
                voice_over_source = st.session_state.get("voice_over_source", "AI Voice Narrator")
                custom_audio_file = st.session_state.get("custom_audio_file")
                temp_dir = tempfile.mkdtemp(prefix="wbv_")
                output_path = os.path.join(temp_dir, "whiteboard_output.mp4")

                if voice_over_source == "Upload Custom Audio Clip (MP4/WAV)":
                    st.write("🎙️ Using uploaded custom narration audio...")
                    audio_path = os.path.join(temp_dir, f"custom_narration{os.path.splitext(custom_audio_file.name)[1]}")
                    custom_audio_file.seek(0)
                    with open(audio_path, "wb") as f:
                        f.write(custom_audio_file.read())
                    exact_audio_duration = get_scene_duration(custom_audio_path=audio_path)
                    per_scene_durations = [exact_audio_duration / max(1, len(ordered_images))] * len(ordered_images)
                    scene_caption_segments = []
                else:
                    st.write("🎙️ Synthesizing professional AI voiceover via edge-tts per scene...")
                    audio_path = os.path.join(temp_dir, "voiceover_combined.mp3")
                    scene_caption_segments = []

                    # Parse numbered script into per-scene text + pause (default 2s)
                    def _parse_script(script: str, default_pause: float = 2.0):
                        entries = {}
                        for line in script.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            # Support formats like: "1: This is text" or "1 - Text"
                            import re
                            m = re.match(r"^(\d+)\s*[:\-)]+\s*(.+)$", line)
                            if not m:
                                # Try split on first ':'
                                parts = line.split(":", 1)
                                if len(parts) == 2 and parts[0].strip().isdigit():
                                    idx = int(parts[0].strip())
                                    txt = parts[1].strip()
                                else:
                                    continue
                            else:
                                idx = int(m.group(1))
                                txt = m.group(2).strip()

                            # Optional inline pause syntax: append "|| pause=2.5"
                            pause = default_pause
                            if "||" in txt:
                                txt, tail = txt.split("||", 1)
                                txt = txt.strip()
                                tail = tail.strip()
                                pm = re.search(r"pause\s*=\s*([0-9.]+)", tail)
                                if pm:
                                    try:
                                        pause = float(pm.group(1))
                                    except Exception:
                                        pass

                            entries[idx] = {"text": txt, "pause": pause}
                        return entries

                    def _write_silence_wav(seconds: float, path: str, framerate: int = 22050):
                        n_channels = 1
                        sampwidth = 2
                        n_frames = int(framerate * seconds)
                        with wave.open(path, "w") as wf:
                            wf.setnchannels(n_channels)
                            wf.setsampwidth(sampwidth)
                            wf.setframerate(framerate)
                            silent_frame = struct.pack('<h', 0)
                            wf.writeframes(silent_frame * n_frames)

                    parsed = _parse_script(script_text or "")

                    # Build per-scene audio clips (TTS + silence) and durations
                    audio_clips = []
                    per_scene_durations = []

                    try:
                        try:
                            from moviepy import concatenate_audioclips
                        except Exception:
                            from moviepy.editor import concatenate_audioclips

                        for i in range(1, len(ordered_images) + 1):
                            entry = parsed.get(i, {"text": "", "pause": 2.0})
                            scene_audio_path = os.path.join(temp_dir, f"voice_scene_{i}.mp3")
                            scene_metadata_path = os.path.join(temp_dir, f"voice_scene_{i}.json")
                            if entry["text"].strip():
                                generate_voiceover_with_timestamps(
                                    entry["text"],
                                    scene_audio_path,
                                    scene_metadata_path,
                                    voice=selected_voice,
                                )
                                clip = AudioFileClip(scene_audio_path)
                                audio_clips.append(clip)
                                scene_duration = clip.duration
                                per_scene_durations.append(scene_duration)

                                word_events = _parse_edge_tts_word_metadata(scene_metadata_path)
                                segments = _build_caption_segments_from_original(word_events, entry["text"])
                                scene_caption_segments.extend(_offset_caption_segments(segments, sum(per_scene_durations[:-1])))
                            else:
                                scene_duration = get_scene_duration(text=entry["text"])
                                per_scene_durations.append(scene_duration)

                            pause_seconds = float(entry.get("pause", 2.0))
                            if pause_seconds > 0.001:
                                silence_path = os.path.join(temp_dir, f"silence_{i}.wav")
                                _write_silence_wav(pause_seconds, silence_path)
                                sclip = AudioFileClip(silence_path)
                                audio_clips.append(sclip)
                                per_scene_durations[-1] += sclip.duration

                        if audio_clips:
                            combined = concatenate_audioclips(audio_clips)
                            combined.write_audiofile(audio_path, logger=None, bitrate="128k")
                            combined.close()
                        else:
                            _write_silence_wav(1.0, audio_path.replace('.mp3', '.wav'))
                            audio_path = audio_path.replace('.mp3', '.wav')
                    finally:
                        for c in audio_clips:
                            try:
                                c.close()
                            except Exception:
                                pass

                st.write("🎞️ Rendering high-fidelity sketch paths and cluster matrices...")
                progress_bar = st.progress(0, text="Processing frames...")

                def update_prog(idx, total):
                    pct = int((idx / total) * 100)
                    progress_bar.progress(pct, text=f"Compiling Scene {idx + 1} of {total}...")

                compose_video(
                    ordered_images=ordered_images,
                    audio_path=audio_path,
                    output_path=output_path,
                    canvas_size=canvas_size,
                    fps=fps_val,
                    hold_seconds=hold_sec,
                    progress_callback=update_prog,
                    caption_segments=scene_caption_segments,
                    music_path=selected_music_path,
                    music_volume_pct=music_volume_pct,
                    per_scene_durations=per_scene_durations if 'per_scene_durations' in locals() else None,
                    custom_audio_path=audio_path if voice_over_source == "Upload Custom Audio Clip (MP4/WAV)" else None,
                )

            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                progress_bar.progress(100, text="Done! 🎉")
                status.success("✅  Your Professional Vector-Sketch video is ready!")
                st.balloons()

                st.download_button(
                    label="⬇️  Download Whiteboard Video (MP4)",
                    data=video_bytes,
                    file_name="whiteboard_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

                st.subheader("Preview")
                st.video(video_bytes)
            else:
                st.error("❌  Video file was not created. Check the terminal for errors.")

        except ImportError as e:
            st.error(
                f"❌  Missing dependency: **{e}**\n\n"
                "Run the install command again and restart the app."
            )
        except Exception as e:
            st.error(f"❌  An error occurred:\n\n```\n{e}\\n```")
            raise


if __name__ == "__main__":
    main()