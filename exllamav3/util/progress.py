import sys
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

class ProgressBar:

    def __init__(self, text: str, count: int, transient: bool = True):
        self.text = text
        self.count = count
        self.transient = transient
        if self.text:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width = None),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient = transient,
            )
            self.task_id = self.progress.add_task(text, total = count)

    def __enter__(self):
        if self.text:
            self.progress.start()
            sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.text:
            if not self.transient:
                self.progress.update(self.task_id, completed = self.count)
            self.progress.stop()

    def update(self, value: int):
        if self.text:
            self.progress.update(self.task_id, completed = value)
            sys.stdout.flush()

    def new_task(self, text: str, count: int):
        self.text = text
        self.count = count
        if self.text:
            self.progress.update(self.task_id, description = self.text, total = count, progress = 0)


