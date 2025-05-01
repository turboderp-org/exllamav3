from __future__ import annotations
from collections import deque
import math

"""
Quick and dirty visualizer for the paged cache, for debug purposes. Horribly slow and should probably be
rewritten to just draw on a bitmap. 
"""

job_colors = [
    "#00DDFF",
    "#9800FF",
    "#D8FF00",
    "#00FFA5",
    "#FF00E4",
    "#FF8800",
    "#057DFF",
    "#FF008C",
    "#00FFE1",
    "#FFFA00",
    "#B6FF00",
    "#D400FF",
    "#FF1900",
    "#FFCC00",
]

empty_color = "#505050"
empty_color_outline = "#707070"

class CacheVisualizer:

    def __init__(
            self,
            num_pages: int,
            window_size: int = (800, 600),
            gap: float = 0.75,
            margin: int = 25
    ):
        import tkinter as tk

        self.num_pages = num_pages
        self.window_size = window_size
        self.gap = gap
        self.margin = margin
        self.chains = []
        self.usage = []

        w, h = self.window_size
        self.root = tk.Tk()
        self.root.title("Cache Map")
        self.canvas = tk.Canvas(
            self.root,
            width = w,
            height = h,
            bg = "#242424",
            highlightthickness = 0,
            borderwidth = 0
        )
        self.canvas.pack(fill = "both", expand = True)

        self.page_rects = []
        for i in range(num_pages):
            rid = self.canvas.create_rectangle(0, 0, 10, 10, fill = empty_color, outline = empty_color_outline)
            self.page_rects.append(rid)

        self._page_grid_layout()
        self.canvas.after_idle(lambda: self.root.bind("<Configure>", self._on_resize))

        self.elements = []
        self.root.update()


    def _page_grid_layout(self):
        w, h = self.window_size
        w -= 2 * self.margin
        h -= 2 * self.margin
        ratio = w / h
        self.w_pages = min(int(math.ceil(math.sqrt(self.num_pages * ratio))), self.num_pages)
        self.h_pages = int(math.ceil(self.num_pages / self.w_pages))
        self.page_bboxes = []
        cell_size_a = w / (self.w_pages + (self.w_pages - 1) * self.gap)
        cell_size_b = h / (self.h_pages + (self.h_pages - 1) * self.gap)
        cell_size = min(cell_size_a, cell_size_b)
        cell_step = cell_size * (1 + self.gap)
        self.cell_step = cell_step
        self.cell_size = cell_size
        self.gap_size = self.gap * cell_size
        for y in range(self.h_pages):
            for x in range(self.w_pages):
                if len(self.page_bboxes) >= self.num_pages:
                    break
                x0 = self.margin + x * cell_step
                y0 = self.margin + y * cell_step
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                rid = self.page_rects[len(self.page_bboxes)]
                self.page_bboxes.append((x0, y0, x1, y1))
                self.canvas.coords(rid, x0, y0, x1, y1)


    def _on_resize(self, event):
        if event.width <= 1 or event.height <= 1:
            return
        w, h = self.window_size
        if (w, h) == (event.width, event.height):
            return
        self.window_size = (event.width, event.height)
        self._page_grid_layout()
        self._update_chains()


    def _update_chains(self):
        for e in self.elements:
            self.canvas.delete(e)
        self.elements.clear()

        cols = [list() for _ in range(self.num_pages)]
        for index, chain in self.chains:
            col = job_colors[index % len(job_colors)]
            for page in chain:
                cols[page] += [col]

        in_handles = []
        out_handles = []
        for page, col in enumerate(cols):
            if not col:
                in_handles.append([])
                out_handles.append([])
                continue
            bbox = self.page_bboxes[page]
            inh = deque()
            outh = deque()
            for i, c in enumerate(col):
                c_br = self.root.tk.call("tk::Darken", c,  135)
                c_dk = self.root.tk.call("tk::Darken", c,  60)
                a = i / len(col)
                b = (i + 1) / len(col)
                x0, y0, x1, y1 = bbox
                h = y1 - y0
                y0, y1 = y0 + a * h, y0 + b * h
                rid = self.canvas.create_rectangle(x0, y0, x1, y1, fill = c, outline = c_br)
                self.elements.append(rid)
                u = self.usage[page] or 0.0
                if u < 1.0:
                    mx = x0 + (x1 - x0) * u
                    rid = self.canvas.create_rectangle(mx, y0, x1, y1, fill = c_dk, outline = c_dk)
                    self.elements.append(rid)
                inh.append((x0, (y0 + y1) * 0.5))
                outh.append((x1, (y0 + y1) * 0.5))
            in_handles.append(inh)
            out_handles.append(outh)

        x = 0
        y = 0
        def start(_x, _y):
            nonlocal x, y
            x, y = _x, _y

        def line(_x, _y):
            nonlocal x, y, col
            aid = self.canvas.create_line(x, y, _x, _y, fill = col, width = 2.0)
            self.elements.append(aid)
            x, y = _x, _y

        def arrow(_x, _y):
            nonlocal x, y, col
            aid = self.canvas.create_line(x, y, _x, _y, arrow = 'last', tags = ("arrow",), fill = col, width = 2.0)
            self.elements.append(aid)
            x, y = _x, _y

        for l_index, (index, chain) in enumerate(self.chains):
            bcol = job_colors[index % len(job_colors)]
            bias = -self.gap_size / 6 + \
                   ((self.gap_size / 3) * l_index + (self.gap_size / 3) * (l_index + 1)) * 0.5 / len(self.chains)
            for page_a, page_b in zip(chain[:-1], chain[1:]):
                if self.usage[page_b]:
                    col = bcol
                else:
                    col = self.root.tk.call("tk::Darken", bcol, 60)

                x0, y0 = out_handles[page_a].popleft()
                x1, y1 = in_handles[page_b].popleft()
                ax0, ay0, ax1, ay1 =  self.page_bboxes[page_a]
                bx0, by0, bx1, by1 =  self.page_bboxes[page_b]
                dy = y1 - y0
                dx = x1 - x0
                cs = self.cell_size
                gs = self.gap_size
                if 0 < dx < cs and abs(dy) < cs:
                    start(x0, y0)
                    line(x0 + gs / 2 + bias, y0)
                    line(x0 + gs / 2 + bias, y1)
                    arrow(x1, y1)
                elif 0 < dx and abs(dy) < cs:
                    start(x0, y0)
                    line(x0 + gs / 2 + bias, y0)
                    line(x0 + gs / 2 + bias, ay1 + gs / 2 + bias)
                    line(x1 - gs / 2 + bias, ay1 + gs / 2 + bias)
                    line(x1 - gs / 2 + bias, y1)
                    arrow(x1, y1)
                elif dy > 0:
                    start(x0, y0)
                    line(x0 + gs / 2 + bias, y0)
                    line(x0 + gs / 2 + bias, by0 - gs / 2 + bias)
                    line(x1 - gs / 2 + bias, by0 - gs / 2 + bias)
                    line(x1 - gs / 2 + bias, y1)
                    arrow(x1, y1)
                else:
                    start(x0, y0)
                    line(x0 + gs / 2 + bias, y0)
                    line(x0 + gs / 2 + bias, by1 + gs / 2 + bias)
                    line(x1 - gs / 2 + bias, by1 + gs / 2 + bias)
                    line(x1 - gs / 2 + bias, y1)
                    arrow(x1, y1)


    def update(self, chains: list[tuple[int, list]], usage: list[float]):
        self.chains = chains
        self.usage = usage
        self._update_chains()
        self.root.update()
