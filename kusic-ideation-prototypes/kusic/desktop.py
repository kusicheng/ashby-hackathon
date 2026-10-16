"""Kevin-C's offline desktop workspace; no web service or credentials."""

import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from kusic import presentation, workbench


class Workbench:
    def __init__(self, root):
        self.root = root
        self.report = None
        self.evaluated_payload = None
        self.busy = False
        self.completed = queue.Queue()
        root.title("Kevin-C | Pathology Research Workbench")
        root.geometry("1180x820")
        root.minsize(900, 640)
        root.configure(bg="#eef2f6")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", foreground="#132b40")
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("TButton", padding=(10, 7), font=("Segoe UI", 10))
        style.configure("TNotebook.Tab", padding=(16, 9))
        outer = ttk.Frame(root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Pathology Research Workbench", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Kevin-C  /  Ashby Prize prototypes  /  local research and education",
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(5, 14))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.idea = tk.StringVar(value="K04")
        self.selector = ttk.Combobox(
            controls,
            state="readonly",
            width=38,
            values=[f"{key} | {item['title']}" for key, item in workbench.IDEAS.items()],
        )
        self.selector.current(3)
        self.selector.pack(side="left", padx=(0, 10))
        self.selector.bind("<<ComboboxSelected>>", self._select)
        self.buttons = []
        for text, command in (
            ("Load sample", self.load_sample),
            ("Open JSON", self.open_input),
            ("Run input", self.run_input),
            ("Compare all ideas", self.compare),
        ):
            button = ttk.Button(controls, text=text, command=command)
            button.pack(side="left", padx=3)
            self.buttons.append(button)
        self.guidance = ttk.Label(outer, wraplength=1090, font=("Segoe UI", 10))
        self.guidance.pack(fill="x", pady=(14, 10))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.summary = self._text_tab("Result")
        self.input = self._text_tab("Input JSON (editable)")
        self.evidence = self._text_tab("Evidence JSON")
        self.summary.configure(state="disabled")
        self.evidence.configure(state="disabled")
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(10, 0))
        self.export_button = ttk.Button(actions, text="Export result JSON", command=self.export)
        self.export_button.pack(side="left", padx=(0, 5))
        self.csv_button = ttk.Button(actions, text="Export comparison CSV", command=self.export_csv)
        self.csv_button.pack(side="left", padx=5)
        self.next_button = ttk.Button(actions, text="Next hint", command=self.next_hint)
        self.next_button.pack(side="left", padx=5)
        self.source_button = ttk.Button(
            actions, text="Inspect prior source", command=self.inspect_source
        )
        self.source_button.pack(side="left", padx=5)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status, wraplength=1090).pack(anchor="w", pady=(10, 0))
        self.load_sample()

    def _text_tab(self, title):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text=title)
        text = ScrolledText(
            frame,
            wrap="word",
            font=("Consolas", 10),
            padx=14,
            pady=14,
            background="#ffffff",
            foreground="#132b40",
            borderwidth=0,
        )
        text.pack(fill="both", expand=True)
        return text

    def _set_text(self, widget, text, editable=False):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="normal" if editable else "disabled")

    def _invalidate(self):
        self.report = None
        self.evaluated_payload = None
        self._set_text(self.summary, "Run the input to inspect its results and evidence.")
        self._set_text(self.evidence, "No result available.")
        self._actions()

    def _actions(self):
        self.export_button.configure(
            state="normal" if self.report and not self.busy else "disabled"
        )
        comparison = bool(self.report and "ideas" in self.report)
        self.csv_button.configure(state="normal" if comparison and not self.busy else "disabled")
        single = self.report and "result" in self.report
        next_hint = (
            single and self.report["idea_id"] == "K03" and self.report["result"].get("next_step")
        )
        prior = single and self.report["idea_id"] == "K02" and self.report["result"].get("matches")
        self.next_button.configure(state="normal" if next_hint and not self.busy else "disabled")
        self.source_button.configure(state="normal" if prior and not self.busy else "disabled")

    def _select(self, _event=None):
        self.idea.set(self.selector.get().split(" | ")[0])
        self.load_sample()

    def load_sample(self):
        payload = workbench.module_for(self.idea.get()).demo_payload()
        self._set_text(self.input, json.dumps(payload, indent=2), editable=True)
        self.guidance.configure(text=presentation.GUIDANCE[self.idea.get()])
        self._invalidate()
        self.status.set(
            "Synthetic sample loaded. Choose Run input, or open an exported JSON cohort."
        )

    def open_input(self):
        path = filedialog.askopenfilename(
            title="Open local prototype JSON input",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = workbench.read_request(path)
            self._set_text(self.input, json.dumps(payload, indent=2), editable=True)
            self._invalidate()
            self.status.set("Local input loaded. Check the selected workflow, then Run input.")
            self.tabs.select(1)
        except (ValueError, OSError) as error:
            messagebox.showerror("Could not open input", str(error))

    def _start(self, operation, payload=None):
        self._invalidate()
        self.busy = True
        self.evaluated_payload = payload
        for button in self.buttons:
            button.configure(state="disabled")
        self.selector.configure(state="disabled")
        self.input.configure(state="disabled")
        self.status.set("Running a bounded local evaluation...")

        def execute():
            try:
                self.completed.put((operation(), None))
            except Exception as error:
                self.completed.put((None, f"{type(error).__name__}: {error}"))

        threading.Thread(target=execute, daemon=True).start()
        self.root.after(50, self._poll)

    def _poll(self):
        try:
            report, error = self.completed.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll)
            return
        self.busy = False
        for button in self.buttons:
            button.configure(state="normal")
        self.selector.configure(state="readonly")
        self.input.configure(state="normal")
        if error:
            self.status.set("Evaluation stopped. Correct the input and run again.")
            self._set_text(self.summary, error)
            self._actions()
            self.tabs.select(0)
            return
        self.report = report
        self._set_text(self.summary, presentation.summary(report))
        self._set_text(self.evidence, json.dumps(report, indent=2, allow_nan=False))
        self._actions()
        self.tabs.select(0)
        self.status.set(
            "Complete. Result and evidence are available; export only if you need a saved copy."
        )

    def run_input(self):
        try:
            payload = workbench.parse_request(self.input.get("1.0", "end"))
        except ValueError as error:
            self._invalidate()
            self.status.set(str(error))
            return
        idea_id = self.idea.get()
        self._start(lambda: workbench.run_request(idea_id, payload), payload)

    def compare(self):
        self._start(workbench.portfolio)

    def next_hint(self):
        payload = dict(self.evaluated_payload)
        payload["step"] = self.report["result"]["next_step"]
        self._set_text(self.input, json.dumps(payload, indent=2), editable=True)
        self.run_input()

    def inspect_source(self):
        from tkinter.simpledialog import askstring

        references = [item["case_ref"] for item in self.report["result"]["matches"]]
        reference = askstring(
            "Inspect local source",
            "Paste a case reference from the result:",
            initialvalue=references[0],
            parent=self.root,
        )
        if not reference:
            return
        try:
            # Use the current edited input so revoked eligibility is rechecked.
            payload = workbench.parse_request(self.input.get("1.0", "end"))
            detail = workbench.module_for("K02").resolve_match(payload, reference)
            messagebox.showinfo("Local source reference (not opened)", json.dumps(detail, indent=2))
        except ValueError as error:
            messagebox.showerror("Source unavailable", str(error))

    def export(self):
        path = filedialog.asksaveasfilename(
            title="Export the completed result",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            try:
                workbench.write_json(path, self.report)
                self.status.set("Completed result exported. Input edits require a new run.")
            except OSError as error:
                messagebox.showerror("Export failed", str(error))

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Export comparison scores", defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if path:
            try:
                workbench.write_comparison_csv(path, self.report)
                self.status.set("Comparable scores exported to CSV.")
            except OSError as error:
                messagebox.showerror("Export failed", str(error))


def launch():
    root = tk.Tk()
    Workbench(root)
    root.mainloop()
