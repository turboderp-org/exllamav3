
"""
Quick & dirty image gallery using TKInter and Pillow
"""

THUMBNAIL_SIZE = 256
THUMBNAIL_PADDING = 10
BACKGROUND = "#202020"
HOVER_BG = "#aaaaaa"
THUMB_TOTAL_WIDTH = THUMBNAIL_SIZE + THUMBNAIL_PADDING
SCREEN_PADDING = 50
DEFAULT_SIZE = "1094x720"
KWSTYLES = {"borderwidth": 0, "highlightthickness": 0, "bg": BACKGROUND}

def gallery(image_paths: list[str], title: str):
    import tkinter as tk
    from tkinter import Toplevel
    from PIL import Image, ImageTk, ImageOps

    def make_thumbnail(path):
        img = Image.open(path)
        thumb_img = ImageOps.pad(img, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), color = BACKGROUND)
        return ImageTk.PhotoImage(thumb_img)

    def show_full_image(path):
        top = Toplevel()
        top.bind("<Escape>", lambda e: top.destroy())
        top.bind("<Button-1>", lambda e: top.destroy())
        top.title(path)
        screen_w, screen_h = top.winfo_screenwidth(), top.winfo_screenheight()
        max_w = screen_w - SCREEN_PADDING * 2
        max_h = screen_h - SCREEN_PADDING * 2
        img = Image.open(path)
        img_w, img_h = img.size
        scale = min(max_w / img_w, max_h / img_h, 1.0)
        if scale < 1.0:
            img = img.resize((int(img_w * scale), int(img_h * scale)), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(top, image = tk_img, **KWSTYLES)
        lbl.image = tk_img
        lbl.pack()

    def draw_grid(columns):
        for widget in scrollable_frame.winfo_children():
            widget.destroy()
        for i, (thumb, path) in enumerate(thumbnails):
            label = tk.Label(scrollable_frame, image = thumb, bg = BACKGROUND, cursor = "hand2")
            label.image = thumb
            label.bind("<Button-1>", lambda e, p = path: show_full_image(p))
            label.grid(row = i // columns, column = i % columns, padx = 5, pady = 5)
            label.bind("<Enter>", lambda e, lbl=label: lbl.configure(bg = HOVER_BG))
            label.bind("<Leave>", lambda e, lbl=label: lbl.configure(bg = BACKGROUND))

    def on_resize(event):
        columns = max(1, (event.width - THUMBNAIL_PADDING) // THUMB_TOTAL_WIDTH)
        draw_grid(columns)

    def on_mousewheel(event):
        if event.num == 4 or event.delta > 0:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            canvas.yview_scroll(1, "units")

    # Layout
    root = tk.Tk()
    root.bind("<Escape>", lambda e: root.destroy())
    root.title(title)
    root.geometry(DEFAULT_SIZE)
    root.configure(**KWSTYLES)
    scrollbar = tk.Scrollbar(root,orient = "vertical", troughcolor = "#222", activebackground = "#666", bd = 0, **KWSTYLES)
    canvas = tk.Canvas(root, **KWSTYLES, yscrollcommand = scrollbar.set)
    scrollable_frame = tk.Frame(canvas, **KWSTYLES)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion = canvas.bbox("all")))
    canvas.create_window((0, 0), window = scrollable_frame, anchor = "nw")
    canvas.pack(side = "left", fill = "both", expand = True)
    scrollbar.pack(side = "right", fill = "y")
    canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows/macOS
    canvas.bind_all("<Button-4>", on_mousewheel)  # Linux up
    canvas.bind_all("<Button-5>", on_mousewheel)  # Linux down
    canvas.bind("<Configure>", on_resize)

    # Preload thumbnails
    thumbnails = [(make_thumbnail(path), path) for path in image_paths]

    # Initial grid
    root.update_idletasks()
    initial_columns = max(1, (root.winfo_width() - THUMBNAIL_PADDING) // THUMB_TOTAL_WIDTH)
    draw_grid(initial_columns)

    # Go
    root.mainloop()