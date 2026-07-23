"""
Plotting for qbench: the scatter chart (ppl / kld vs bpw) and the KLD spread chart.

The theme and the leader-label layout engine (collision-relaxed point labels with leader lines)
are absorbed from compare_q_plot.py, which this module supersedes.
"""

import math
import textwrap

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------------------------
# Theme

def _set_theme(dark):
    if dark:
        sns.set_theme(
            style = "darkgrid",
            context = "talk",
            rc = {
                "figure.facecolor": "#15171c",
                "axes.facecolor": "#1f2329",
                "axes.edgecolor": "#4b515c",
                "axes.labelcolor": "#e6e8eb",
                "axes.titlecolor": "#f1f3f5",
                "grid.color": "#303238",
                "text.color": "#e6e8eb",
                "xtick.color": "#c8ccd2",
                "ytick.color": "#c8ccd2",
                "legend.facecolor": "#252a31",
                "legend.edgecolor": "#4b515c",
            },
        )
    else:
        sns.set_theme(style = "whitegrid", context = "talk")


def _text_colors(dark):
    return {
        "value": "#f1f3f5" if dark else "#33373d",
        "sep": "#8e949d" if dark else "#9aa0a8",
        "muted": "#b9bec6" if dark else "#5f6670",
        "tick": "#8e949d" if dark else "#5f6670",
        "floor": "#8e949d" if dark else "#7a828c",
    }


def make_palette(groups):
    """Fixed hue per group name; reserved slots for common formats, remainder in stable order"""
    cols = sns.color_palette("tab10", n_colors = 10)
    fixed_cols = {"AWQ": 0, "EXL3": 1, "GGUF": 2}
    unused = [i for i in range(len(cols)) if i not in fixed_cols.values()]
    palette = {}
    for g in sorted(groups):
        if g in fixed_cols:
            palette[g] = cols[fixed_cols[g]]
    for g in sorted(groups):
        if g not in fixed_cols:
            palette[g] = cols[unused.pop(0)]
    return palette


# ---------------------------------------------------------------------------------------------
# Leader-label layout engine (from compare_q_plot.py)

def _split_label(label):
    label = label.split("[")[0].strip()
    parts = label.split(maxsplit = 1)
    group = parts[0] if parts else "Other"
    point_label = parts[1] if len(parts) > 1 else group
    return group, point_label


def _make_box(center, width, height):
    return mtransforms.Bbox.from_bounds(center[0] - width / 2, center[1] - height / 2, width, height)


def _overlap_area(a, b):
    ix = min(a.x1, b.x1) - max(a.x0, b.x0)
    iy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def contains(p, q, r):
        return (
            min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6 and
            min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) <= 1e-6 and contains(a, c, b):
        return True
    if abs(o2) <= 1e-6 and contains(a, d, b):
        return True
    if abs(o3) <= 1e-6 and contains(c, a, d):
        return True
    if abs(o4) <= 1e-6 and contains(c, b, d):
        return True
    return False


def _fit_center_curve(rows, ax):
    xs = np.array([r["x"] for r in rows], dtype = np.float64)
    ys = np.array([r["y"] for r in rows], dtype = np.float64)
    x_span = max(float(xs.max() - xs.min()), 1e-9)
    y_floor = max(float(ys[ys > 0].min()) * 0.1 if np.any(ys > 0) else 1e-9, 1e-9)
    safe_ys = np.maximum(ys, y_floor)

    try:
        slope, intercept = np.polyfit(xs, np.log(safe_ys), 1)
    except np.linalg.LinAlgError:
        slope, intercept = 0.0, math.log(float(np.median(safe_ys)))

    def curve_y(x):
        return max(math.exp(intercept + slope * x), y_floor)

    def curve_px(x):
        return ax.transData.transform((x, curve_y(x)))

    def normal_at(x):
        x0 = x - x_span * 0.01
        x1 = x + x_span * 0.01
        a = curve_px(x0)
        b = curve_px(x1)
        tangent = b - a
        length = max(float(np.linalg.norm(tangent)), 1e-9)
        normal = np.array([-tangent[1], tangent[0]], dtype = np.float64) / length
        return normal, curve_px(x)

    return curve_y, normal_at


def _line_obstacles(line_df, ax):
    obstacles = []
    for _, group_df in line_df.groupby("group", sort = False):
        points = [
            ax.transData.transform((row.x, row.y))
            for row in group_df.itertuples()
        ]
        for a, b in zip(points, points[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            distance = max((dx * dx + dy * dy) ** 0.5, 1.0)
            steps = max(int(distance // 24), 1)
            for step in range(steps + 1):
                t = step / steps
                x = a[0] + dx * t
                y = a[1] + dy * t
                obstacles.append(_make_box((x, y), 18, 18))
    return obstacles


def _line_segments(line_df, ax):
    segments = []
    for _, group_df in line_df.groupby("group", sort = False):
        points = [
            ax.transData.transform((row.x, row.y))
            for row in group_df.itertuples()
        ]
        segments += list(zip(points, points[1:]))
    return segments


def _segment_intersects_box(a, b, box):
    if box.contains(*a) or box.contains(*b):
        return True
    corners = [
        np.array((box.x0, box.y0), dtype = np.float64),
        np.array((box.x1, box.y0), dtype = np.float64),
        np.array((box.x1, box.y1), dtype = np.float64),
        np.array((box.x0, box.y1), dtype = np.float64),
    ]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(_segments_cross(a, b, c, d) for c, d in edges)


def _touches_endpoint(a, segment):
    return (
        np.linalg.norm(a - segment[0]) < 1e-4 or
        np.linalg.norm(a - segment[1]) < 1e-4
    )


def _score_layout(centers, sizes, anchors, initial_centers, obstacles, line_segments):
    boxes = [
        _make_box(center, width, height)
        for center, (width, height) in zip(centers, sizes)
    ]
    score = 0.0

    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            score += _overlap_area(box, other) * 1000.0
        for obstacle in obstacles:
            score += _overlap_area(box, obstacle) * 500.0

    for center, anchor, initial in zip(centers, anchors, initial_centers):
        leader_length = float(np.linalg.norm(center - anchor))
        score += leader_length * 0.22
        score += leader_length * leader_length * 0.0008

    for i, (a, size_a) in enumerate(zip(centers, sizes)):
        for b, size_b in zip(centers[i + 1:], sizes[i + 1:]):
            distance = max(float(np.linalg.norm(a - b)), 1e-6)
            preferred = max(size_a[0], size_b[0]) * 2.0
            if distance < preferred:
                score += (preferred - distance) ** 2 * 0.025

    leaders = list(zip(anchors, centers))
    for i, (a0, a1) in enumerate(leaders):
        if np.linalg.norm(a1 - a0) < 22:
            continue
        for segment in line_segments:
            if _touches_endpoint(a0, segment):
                continue
            if _segments_cross(a0, a1, segment[0], segment[1]):
                score += 9000.0
        for b0, b1 in leaders[i + 1:]:
            if np.linalg.norm(b1 - b0) < 22:
                continue
            if _segments_cross(a0, a1, b0, b1):
                score += 8000.0
        for j, box in enumerate(boxes):
            if i != j and _segment_intersects_box(a0, a1, box):
                score += 6000.0

    return score


def _leader_conflict_count(idx, center, centers, sizes, anchors, line_segments):
    leader = (anchors[idx], center)
    boxes = [
        _make_box(c, width, height)
        for c, (width, height) in zip(centers, sizes)
    ]
    conflicts = 0

    for segment in line_segments:
        if _touches_endpoint(leader[0], segment):
            continue
        if _segments_cross(leader[0], leader[1], segment[0], segment[1]):
            conflicts += 1

    for other_idx, other_center in enumerate(centers):
        if other_idx == idx:
            continue
        if _segments_cross(leader[0], leader[1], anchors[other_idx], other_center):
            conflicts += 1
        if _segment_intersects_box(leader[0], leader[1], boxes[other_idx]):
            conflicts += 1
        if boxes[idx].overlaps(boxes[other_idx]):
            conflicts += 1

    return conflicts


def _leader_has_conflict(idx, center, centers, sizes, anchors, line_segments):
    return _leader_conflict_count(idx, center, centers, sizes, anchors, line_segments) > 0


def _repair_leader_conflicts(centers, sizes, anchors, axes_box, line_segments, attempt, stage):
    centers = [np.array(center, dtype = np.float64) for center in centers]
    moved = False

    for idx in range(len(centers)):
        if not _leader_has_conflict(idx, centers[idx], centers, sizes, anchors, line_segments):
            continue

        rng = np.random.default_rng(1009 + attempt * 97 + stage * 193 + idx * 389)
        width, height = sizes[idx]
        best_center = centers[idx]
        best_score = float("inf")

        for trial in range(50):
            radius = 28.0 + trial * 2.8
            angle = rng.uniform(0.0, math.tau)
            candidate = anchors[idx] + np.array((math.cos(angle), math.sin(angle))) * radius
            candidate[0] = min(max(candidate[0], axes_box.x0 + width / 2), axes_box.x1 - width / 2)
            candidate[1] = min(max(candidate[1], axes_box.y0 + height / 2), axes_box.y1 - height / 2)

            trial_centers = list(centers)
            trial_centers[idx] = candidate
            conflicts = _leader_conflict_count(idx, candidate, trial_centers, sizes, anchors, line_segments)
            score = conflicts * 10000.0 + float(np.linalg.norm(candidate - anchors[idx]))
            if conflicts == 0:
                centers[idx] = candidate
                moved = True
                break
            if score < best_score:
                best_score = score
                best_center = candidate
        else:
            centers[idx] = best_center
            moved = True

    return centers, moved


def _staged_layout(initial, sizes, anchors, axes_box, obstacles, line_segments, attempt):
    centers = initial
    for stage in range(3):
        centers = _relax_layout(centers, sizes, anchors, initial, axes_box, obstacles)
        centers, repaired = _repair_leader_conflicts(
            centers,
            sizes,
            anchors,
            axes_box,
            line_segments,
            attempt,
            stage,
        )
        if not repaired:
            break
    return centers


def _relax_layout(centers, sizes, anchors, initial_centers, axes_box, obstacles):
    centers = [np.array(center, dtype = np.float64) for center in centers]
    obstacle_specs = [
        (
            obstacle.x0 + obstacle.width / 2,
            obstacle.y0 + obstacle.height / 2,
            obstacle.width,
            obstacle.height,
            obstacle,
        )
        for obstacle in obstacles
    ]

    for iteration in range(240):
        max_delta = 0.0
        boxes = [
            _make_box(center, width, height)
            for center, (width, height) in zip(centers, sizes)
        ]

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                ix = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
                iy = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
                if ix <= 0 or iy <= 0:
                    continue
                if ix < iy:
                    step = ix / 2 + 1.5
                    direction = -1 if centers[i][0] <= centers[j][0] else 1
                    centers[i][0] += direction * step
                    centers[j][0] -= direction * step
                else:
                    step = iy / 2 + 1.5
                    direction = -1 if centers[i][1] <= centers[j][1] else 1
                    centers[i][1] += direction * step
                    centers[j][1] -= direction * step
                max_delta = max(max_delta, step)

        boxes = [
            _make_box(center, width, height)
            for center, (width, height) in zip(centers, sizes)
        ]
        for i, box in enumerate(boxes):
            for ox, oy, ow, oh, obstacle in obstacle_specs:
                if abs(centers[i][0] - ox) > (sizes[i][0] + ow) / 2:
                    continue
                if abs(centers[i][1] - oy) > (sizes[i][1] + oh) / 2:
                    continue
                if not box.overlaps(obstacle):
                    continue
                ix = min(box.x1, obstacle.x1) - max(box.x0, obstacle.x0)
                iy = min(box.y1, obstacle.y1) - max(box.y0, obstacle.y0)
                if ix <= 0 or iy <= 0:
                    continue
                dx = centers[i][0] - ox
                dy = centers[i][1] - oy
                if ix < iy:
                    push = (ix + 2.0) if dx >= 0 else -(ix + 2.0)
                    centers[i][0] += push
                    max_delta = max(max_delta, abs(push))
                else:
                    push = (iy + 2.0) if dy >= 0 else -(iy + 2.0)
                    centers[i][1] += push
                    max_delta = max(max_delta, abs(push))

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                delta = centers[i] - centers[j]
                distance = max(float(np.linalg.norm(delta)), 1e-6)
                preferred = max(sizes[i][0], sizes[j][0]) * 1.5
                if distance >= preferred:
                    continue
                push = delta / distance * min((preferred - distance) * 0.012, 1.2)
                centers[i] += push
                centers[j] -= push
                max_delta = max(max_delta, float(np.linalg.norm(push)))

        for i, ((width, height), initial) in enumerate(zip(sizes, initial_centers)):
            pull = (initial - centers[i]) * 0.001
            leader_pull = anchors[i] - centers[i]
            leader_distance = max(float(np.linalg.norm(leader_pull)), 1e-6)
            pull += leader_pull / leader_distance * min(leader_distance * 0.006, 1.1)
            centers[i] += pull
            max_delta = max(max_delta, float(np.linalg.norm(pull)))
            centers[i][0] = min(max(centers[i][0], axes_box.x0 + width / 2), axes_box.x1 - width / 2)
            centers[i][1] = min(max(centers[i][1], axes_box.y0 + height / 2), axes_box.y1 - height / 2)

        if iteration > 30 and max_delta < 0.05:
            break

    return centers


def _initial_label_centers(rows, anchors, sizes, ax, attempt):
    _, normal_at = _fit_center_curve(rows, ax)
    centers = []
    ordered = sorted(range(len(rows)), key = lambda i: (rows[i]["x"], rows[i]["y"], rows[i]["group"], rows[i]["point_label"]))

    for rank, idx in enumerate(ordered):
        row = rows[idx]
        anchor = anchors[idx]
        width, height = sizes[idx]
        normal, curve = normal_at(row["x"])
        side = 1.0 if float(np.dot(anchor - curve, normal)) >= 0 else -1.0

        tangent = np.array([normal[1], -normal[0]], dtype = np.float64)
        base_distance = max(42.0, min(92.0, max(width, height) * 0.85))
        center = anchor + normal * side * base_distance

        if attempt:
            shake = 10.0 + attempt * 5.0
            center += tangent * math.sin((rank + 1) * (attempt + 2.3)) * shake
            center += normal * side * math.cos((rank + 2.5) * (attempt + 1.7)) * shake * 0.55
        if attempt == 3 and rank % 5 == 0:
            center = anchor - normal * side * (base_distance * 0.75)

        centers.append((idx, center))

    centers.sort(key = lambda x: x[0])
    return [center for _, center in centers]


def _layout_labels(fig, ax, rows, anchors, sizes, line_df, extra_obstacles = None):
    """Run the staged layout over abstract label boxes; returns final pixel centers"""
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer).padded(-8)
    obstacles = _line_obstacles(line_df, ax)
    obstacles += [_make_box(anchor, 30, 30) for anchor in anchors]
    if extra_obstacles:
        obstacles += extra_obstacles
    line_segments = _line_segments(line_df, ax)

    best_score = float("inf")
    best_centers = None
    for attempt in range(3):
        initial = _initial_label_centers(rows, anchors, sizes, ax, attempt)
        centers = _staged_layout(initial, sizes, anchors, axes_box, obstacles, line_segments, attempt)
        score = _score_layout(centers, sizes, anchors, initial, obstacles, line_segments)
        if score < best_score:
            best_score = score
            best_centers = centers
    return best_centers


def _draw_leader(ax, anchor, center, color):
    distance = float(np.linalg.norm(np.asarray(center) - np.asarray(anchor)))
    if distance > 22:
        x0, y0 = ax.transData.inverted().transform(anchor)
        x1, y1 = ax.transData.inverted().transform(center)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color = color,
            alpha = 0.5,
            linewidth = 0.7,
            zorder = 4,
        )


def _add_point_labels(fig, ax, rows, line_df, dark, palette):
    """Scatter-chart labels: point name above, value below (compare_q_plot behavior)"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    anchors = [ax.transData.transform((r["x"], r["y"])) for r in rows]
    labels = []

    for r in rows:
        label_text = ax.text(
            r["x"],
            r["y"],
            r["point_label"],
            color = palette[r["group"]],
            fontsize = 8.5,
            fontweight = "bold",
            ha = "center",
            va = "bottom",
            bbox = {
                "boxstyle": "round,pad=0.25",
                "facecolor": ax.get_facecolor(),
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder = 5,
        )
        score_text = ax.text(
            r["x"],
            r["y"],
            f"{r['y']:.3f}",
            fontsize = 8.5,
            ha = "center",
            va = "top",
            zorder = 6,
        )
        labels.append((label_text, score_text, r))

    fig.canvas.draw()
    sizes = []
    label_sizes = []
    for label_text, score_text, _ in labels:
        label_box = label_text.get_window_extent(renderer)
        score_box = score_text.get_window_extent(renderer)
        box = mtransforms.Bbox.union([label_box, score_box]).padded(2)
        sizes.append((box.width, box.height))
        label_sizes.append((label_box.height, score_box.height))

    best_centers = _layout_labels(fig, ax, rows, anchors, sizes, line_df)

    for (label_text, score_text, row), center, anchor, (label_height, score_height) in zip(labels, best_centers, anchors, label_sizes):
        split_y = center[1] + (score_height - label_height) / 2
        split_pos = ax.transData.inverted().transform((center[0], split_y))
        label_text.set_position(split_pos)
        score_text.set_position(split_pos)
        _draw_leader(ax, anchor, center, palette[row["group"]])


def _add_spread_point_labels(fig, ax, rows, line_df, dark, palette, extra_obstacles):
    """
    Spread-chart labels: point name (group color, bold) above a value row, which is either a
    single median value or "median | excess-over-floor" with a dimmed separator. Laid out by the
    same collision-relaxed engine as the scatter labels.
    """
    colors = _text_colors(dark)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    anchors = [ax.transData.transform((r["x"], r["y"])) for r in rows]

    pill = {
        "boxstyle": "round,pad=0.25",
        "facecolor": ax.get_facecolor(),
        "edgecolor": "none",
        "alpha": 0.72,
    }

    labels = []
    for r in rows:
        name = ax.text(
            r["x"], r["y"], r["point_label"],
            color = palette[r["group"]], fontsize = 9, fontweight = "bold",
            ha = "center", va = "bottom",
            bbox = dict(pill),
            zorder = 5,
        )
        if r.get("value2") is not None:
            # One pill behind the whole value row: a transparent-glyph text with the combined
            # string carries the background and auto-sizes to it; the colored parts render on
            # top without their own boxes
            bg = ax.text(
                r["x"], r["y"], f"{r['value']} | {r['value2']}",
                color = (0, 0, 0, 0), fontsize = 9, ha = "center", va = "top",
                bbox = dict(pill), zorder = 5,
            )
            parts = [
                ax.text(r["x"], r["y"], r["value"], color = colors["value"], fontsize = 9,
                        ha = "right", va = "top", zorder = 6),
                ax.text(r["x"], r["y"], "|", color = colors["sep"], fontsize = 9,
                        ha = "center", va = "top", zorder = 6),
                ax.text(r["x"], r["y"], r["value2"], color = colors["value"], fontsize = 9,
                        ha = "left", va = "top", zorder = 6),
            ]
        else:
            bg = None
            parts = [
                ax.text(r["x"], r["y"], r["value"], color = colors["value"], fontsize = 9,
                        ha = "center", va = "top", bbox = dict(pill), zorder = 6),
            ]
        labels.append((name, bg, parts, r))

    fig.canvas.draw()
    sizes = []
    metas = []
    for name, bg, parts, _ in labels:
        name_box = name.get_window_extent(renderer)
        row_box = bg.get_window_extent(renderer) if bg is not None else parts[0].get_window_extent(renderer)
        row_w = row_box.width
        row_h = row_box.height
        w = max(name_box.width, row_w) + 4
        h = name_box.height + row_h + 2
        sizes.append((w, h))
        metas.append((name_box.height, row_h))

    best_centers = _layout_labels(fig, ax, rows, anchors, sizes, line_df, extra_obstacles)

    inv = ax.transData.inverted()
    for (name, bg, parts, row), center, anchor, (name_h, row_h) in zip(labels, best_centers, anchors, metas):
        split_y = center[1] + (row_h - name_h) / 2
        name.set_position(inv.transform((center[0], split_y)))
        if bg is not None:
            bg.set_position(inv.transform((center[0], split_y)))
        if len(parts) == 3:
            parts[0].set_position(inv.transform((center[0] - 4, split_y)))
            parts[1].set_position(inv.transform((center[0], split_y)))
            parts[2].set_position(inv.transform((center[0] + 4, split_y)))
        else:
            parts[0].set_position(inv.transform((center[0], split_y)))
        _draw_leader(ax, anchor, center, palette[row["group"]])


# ---------------------------------------------------------------------------------------------
# Scatter chart (ppl or kld vs bpw / vram)

def plot_scatter(results, args, ref_line = None):
    """
    Scatter of one metric vs storage cost, grouped by format, with leader labels. ref_line, if
    given, is drawn as a dotted horizontal line with a left-aligned label instead of a data
    point (keeps a distant reference from compressing the interesting region).
    """
    x_key = "vram_gb" if args.vram else "layer_bpw"
    y_key = "kld" if args.kld else "ppl"
    x_label = (
        r"quantized weight size $|W_q|$ / GiB (excl. embeddings, incl. output head)" if args.vram else
        r"bits per weight (excl. embeddings and output head)"
    )
    y_label = (
        r"KL divergence, $D_{\mathrm{KL}}(p_{\mathrm{FP}} \parallel p_{\mathrm{quant}})$" if args.kld else
        r"perplexity"
    )

    rows = []
    for r in results:
        if y_key not in r:
            continue
        x_ = r[x_key]
        y_ = r[y_key]
        if x_ > args.max_x or y_ > args.max_y:
            continue
        group, point_label = _split_label(r["label"])
        rows.append(
            {
                "group": group,
                "point_label": point_label,
                "label": r["label"].split("[")[0].strip(),
                "x": x_,
                "y": y_,
            }
        )

    if not rows:
        print("No plottable results after applying axis/mask limits.")
        return

    _set_theme(args.dark)
    plt.rcParams["figure.figsize"] = (14, 10)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left = 0.08, right = 0.96, top = 0.89, bottom = 0.10)

    df = pd.DataFrame(rows)
    groups = sorted(df["group"].unique())
    palette = make_palette(groups)

    group_counts = df["group"].value_counts()
    line_df = df[df["group"].map(group_counts) > 1].sort_values(["group", "x", "y", "point_label"])

    if not line_df.empty:
        sns.lineplot(
            data = line_df,
            x = "x",
            y = "y",
            hue = "group",
            palette = palette,
            hue_order = groups,
            linewidth = 1.8,
            linestyle = ":",
            estimator = None,
            sort = False,
            ax = ax,
            legend = False,
        )

    sns.scatterplot(
        data = df,
        x = "x",
        y = "y",
        hue = "group",
        palette = palette,
        hue_order = groups,
        s = 86,
        edgecolor = "white",
        linewidth = 0.8,
        ax = ax,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color = palette[group],
            linestyle = ":",
            linewidth = 1.8,
            marker = "o",
            markersize = 7,
            markerfacecolor = palette[group],
            markeredgecolor = "white",
            markeredgewidth = 0.8,
            label = group,
        )
        for group in groups
    ]
    ax.legend(
        handles = handles,
        loc = "upper right",
        bbox_to_anchor = (0.98, 0.98),
        frameon = False,
        fontsize = 14,
        handlelength = 1.8,
        handletextpad = 0.7,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    if args.kld:
        ax.yaxis.label.set_verticalalignment("bottom")
    colors = _text_colors(args.dark)
    ax.tick_params(axis = "both", which = "major", labelsize = 13, colors = colors["tick"])
    subtitle = getattr(args, "subtitle", None)
    if subtitle:
        ax.set_title(args.title, pad = 42)
        ax.text(
            0.5,
            1.025,
            subtitle,
            transform = ax.transAxes,
            ha = "center",
            va = "bottom",
            fontsize = 13,
            color = colors["muted"],
        )
    else:
        ax.set_title(args.title, pad = 22)
    ax.margins(x = 0.08, y = 0.12)
    sns.despine(ax = ax, left = True, bottom = True)

    # Reference as a horizontal line, extending the y range to include it if needed
    if ref_line is not None:
        y0, y1 = ax.get_ylim()
        pad = (y1 - y0) * 0.06
        if not (y0 + pad <= ref_line["value"] <= y1 - pad):
            ax.set_ylim(min(y0, ref_line["value"] - pad), max(y1, ref_line["value"] + pad))
        x0, x1 = ax.get_xlim()
        ax.axhline(ref_line["value"], color = colors["floor"], linewidth = 2.2, linestyle = ":", zorder = 1)
        ax.annotate(
            f"{ref_line['label']}: {ref_line['value']:.4f}",
            (x0, ref_line["value"]),
            textcoords = "offset points", xytext = (6, 5),
            ha = "left", fontsize = 11, color = colors["floor"], zorder = 5,
        )
        ax.set_xlim(x0, x1)

    _add_point_labels(fig, ax, rows, line_df, args.dark, palette)

    try:
        import mplcursors
        point_collection = next(
            c for c in reversed(ax.collections)
            if len(c.get_offsets()) == len(rows)
        )
        cursor = mplcursors.cursor(point_collection, hover = True)

        @cursor.connect("add")
        def on_add(sel):
            point = rows[sel.index]
            sel.annotation.set_text(
                f"{point['label']}\n{x_label}: {point['x']:.3f}\n{y_label}: {point['y']:.4f}"
            )
    except (ImportError, StopIteration):
        pass

    if args.plot_file:
        fig.savefig(args.plot_file, dpi = 160)
        plt.close(fig)
    else:
        plt.show()


# Backwards-compatible entry point matching compare_q_plot.plot()
def plot(results, args):
    plot_scatter(results, args)


# ---------------------------------------------------------------------------------------------
# KLD spread chart

def default_spread_caption(floor) -> str:
    cap = (
        "Per-token KL divergence between each quantized model and the unquantized reference. The solid line follows the median token, with a shaded "
        "p25–p75 band; the dotted line is the mean, running above the median when divergence is heavy-tailed, "
        "concentrating in tokens where the reference itself is undecided."
    )
    if floor is not None:
        cap += (
            " Gray dotted lines mark the reference's self-noise floor: the divergence the bf16 model shows against itself when perturbed at the scale "
            "of its own rounding error. The right-hand scale expresses divergence as a multiple of that floor."
        )
    return cap


def plot_kld_spread(results: list, title: str, subtitle: str, dark: bool, plot_file: str, caption: str | bool = True, vram: bool = False):
    """
    KLD vs bpw with the per-token spread: solid median line with a shaded p25-p75 band per group
    (log scale - the spread covers orders of magnitude), the mean as a faint dotted line, and
    the reference's self-noise floor as dotted lines. A quant is at the floor where its band
    sinks below the floor line.
    """
    _set_theme(dark)
    plt.rcParams["figure.figsize"] = (14, 10.6)
    fig, ax = plt.subplots()
    # Extra right margin for the twin-axis (x floor) label
    fig.subplots_adjust(left = 0.09, right = 0.93, top = 0.895, bottom = 0.10 if caption is False else 0.185)
    colors = _text_colors(dark)

    groups = {}
    floor = None
    for r in results:
        if "kld" not in r:
            continue
        if r["group"] == "noise_floor":
            floor = r
            continue
        groups.setdefault(r["group"], []).append(r)

    palette = make_palette(groups.keys())

    # x-span across all groups, for sizing single-point columns
    x_key = "vram_gb" if vram else "bpw_layer"
    all_x = [r[x_key] for rs in groups.values() for r in rs]
    x_span = max(max(all_x) - min(all_x), 1e-6)

    label_rows = []
    line_records = []
    for g in sorted(groups):
        rs = sorted(groups[g], key = lambda r: r[x_key])
        xs = [r[x_key] for r in rs]
        color = palette[g]
        # p10/p90 are measured and cached too, but only the interquartile band is drawn to keep
        # the chart readable once several groups overlap. A single-point group gets a thin
        # column instead of a (zero-width) band
        if len(rs) == 1:
            halfw = x_span * 0.012
            xs_band = [xs[0] - halfw, xs[0] + halfw]
            band_lo = [rs[0]["kld_p25"]] * 2
            band_hi = [rs[0]["kld_p75"]] * 2
        else:
            xs_band = xs
            band_lo = [r["kld_p25"] for r in rs]
            band_hi = [r["kld_p75"] for r in rs]
        ax.fill_between(
            xs_band, band_lo, band_hi,
            color = color, alpha = 0.16, linewidth = 0, zorder = 2,
        )
        ax.plot(
            xs, [r["kld_median"] for r in rs],
            color = color, linewidth = 2.2, marker = "o", markersize = 8,
            markeredgecolor = "white", markeredgewidth = 0.8,
            label = f"{g} (median, p25–p75)", zorder = 4,
        )
        ax.plot(
            xs, [r["kld"] for r in rs],
            color = color, linewidth = 1.0, linestyle = ":", marker = "D", markersize = 4.5,
            alpha = 0.65, label = f"{g} (mean)", zorder = 3,
        )
        for r in rs:
            line_records.append({"group": f"{g}-median", "x": r[x_key], "y": r["kld_median"]})
            line_records.append({"group": f"{g}-mean", "x": r[x_key], "y": r["kld"]})
            row = {
                "group": g,
                "point_label": r["label"],
                "x": r[x_key],
                "y": r["kld_median"],
                "value": f"{r['kld_median']:.4f}",
                "value2": f"{r['kld_median'] - floor['kld_median']:.4f}" if floor is not None else None,
            }
            label_rows.append(row)

    extra_obstacles = []
    if floor is not None:
        x0, x1 = ax.get_xlim()
        # Same faint pattern as the group mean lines for the floor mean, a stronger dotted line
        # for the floor median; labels sit at the left edge where the sloping data leaves room
        ax.axhline(floor["kld_median"], color = colors["floor"], linewidth = 2.2, linestyle = ":", zorder = 1)
        ax.axhline(floor["kld"], color = colors["floor"], linewidth = 1.0, linestyle = ":", alpha = 0.65, zorder = 1)
        floor_texts = [
            ax.annotate(
                f"noise floor, median: {floor['kld_median']:.4f}",
                (x0, floor["kld_median"]),
                textcoords = "offset points", xytext = (6, 5),
                ha = "left", fontsize = 11, color = colors["floor"], zorder = 5,
            ),
            ax.annotate(
                f"noise floor, mean: {floor['kld']:.4f}",
                (x0, floor["kld"]),
                textcoords = "offset points", xytext = (6, 5),
                ha = "left", fontsize = 11, color = colors["floor"], alpha = 0.8, zorder = 5,
            ),
        ]
        ax.set_xlim(x0, x1)
        for y in (floor["kld_median"], floor["kld"]):
            line_records.append({"group": f"floor-{y}", "x": x0, "y": y})
            line_records.append({"group": f"floor-{y}", "x": x1, "y": y})
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        extra_obstacles = [t.get_window_extent(renderer).padded(2) for t in floor_texts]

    ax.set_yscale("log")
    ax.set_xlabel(
        r"quantized weight size $|W_q|$ / GiB (excl. embeddings, incl. output head)" if vram else
        r"bits per weight (excl. embeddings and output head)"
    )
    ax.set_ylabel(r"per-token KL divergence, $D_{\mathrm{KL}}(p_{\mathrm{FP}} \parallel p_{\mathrm{quant}})$")
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis = "both", which = "both", labelsize = 13, colors = colors["tick"])
    ax.legend(loc = "upper right", frameon = False, fontsize = 12)

    # Second scale in units of the floor median: the ratio is the scale-free reading (1x =
    # indistinguishable from run-to-run numerical variation), comparable across models with
    # very different floors
    ay = None
    if floor is not None:
        ay = ax.twinx()
        y0, y1 = ax.get_ylim()
        ay.set_yscale("log")
        ay.set_ylim(y0 / floor["kld_median"], y1 / floor["kld_median"])
        ay.set_ylabel(r"$\times$ noise floor")
        ay.yaxis.label.set_size(13)
        ay.tick_params(axis = "y", which = "both", labelsize = 12, colors = colors["tick"])
        ay.grid(False)
    if subtitle:
        ax.set_title(title, pad = 42)
        ax.text(
            0.5, 1.025, subtitle, transform = ax.transAxes, ha = "center", va = "bottom",
            fontsize = 13, color = colors["muted"],
        )
    else:
        ax.set_title(title, pad = 22)
    sns.despine(ax = ax, left = True, bottom = True)
    if ay is not None:
        sns.despine(ax = ay, left = True, bottom = True, right = True)

    # Leader-labeled point annotations, collision-relaxed against both lines of every group,
    # the floor lines and the floor labels
    line_df = pd.DataFrame(line_records).sort_values(["group", "x", "y"])
    _add_spread_point_labels(fig, ax, label_rows, line_df, dark, palette, extra_obstacles)

    if caption is not False:
        text = caption if isinstance(caption, str) else default_spread_caption(floor)
        fig.text(
            0.09, 0.094,
            "\n".join(textwrap.wrap(text, width = 158)),
            ha = "left", va = "top", fontsize = 10.5, linespacing = 1.45,
            color = colors["muted"],
        )

    fig.savefig(plot_file, dpi = 160)
    plt.close(fig)
