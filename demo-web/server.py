#!/usr/bin/env python3
"""Browser presentation layer for the configured circuit pipeline."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import threading
import time
from urllib.parse import quote, unquote, urlsplit, parse_qs
import uuid
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError
import sympy as sp

MAX_BODY = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}\Z")
EXTENSIONS = {".png": "image", ".json": "json", ".md": "text", ".log": "text", ".cir": "text", ".spice": "text"}
STAGES = [
    ("input", "Original image"), ("detect", "YOLO detections"),
    ("segment", "SAM foreground mask"), ("graph", "Initial pins and nets"),
    ("vision", "Diagram reading"), ("assemble", "Finalized circuit graph"),
    ("prepare", "Prepare exact linear state"), ("target", "Question interpretation"),
    ("calculate", "Exact calculation"), ("spice", "SPICE simulation"),
    ("explain", "Worked solution"), ("answer", "Final answer"),
]
DATA_FILES = {
    "run": "run.json", "detections": "detections.json", "candidate": "candidate.json",
    "readings": "readings.json", "circuit": "circuit.json", "linear_state": "linear-state.json",
    "reasoning_plan": "reasoning-plan.json", "calculation": "calculation.json", "answer": "answer.json",
    "spice_result": "spice-result.json", "reader_mode": "reader-mode.json",
    "explanation": "explanation.json",
}


def configured_model(settings):
    """Mirror the provider's documented model override without loading it."""
    variable = {"codex_cli": "KBOT_CODEX_MODEL", "openai": "OPENAI_MODEL"}.get(settings.get("provider"))
    return (os.environ.get(variable, "").strip() if variable else "") or settings["model"]


def arithmetic_expression(source):
    """Read arithmetic data without evaluating Python or SymPy source code."""
    if not isinstance(source, str) or len(source) > 8192:
        raise ValueError("Unsupported expression.")
    tree = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("Expression is too large.")

    def read(node):
        if isinstance(node, ast.Expression):
            return read(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or len(str(node.value)) > 80:
                raise ValueError("Unsupported number.")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", node.id):
            return sp.I if node.id == "I" else sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = read(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = read(node.left), read(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right != 0:
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Integer and abs(right) <= 16:
                return sp.Pow(left, right, evaluate=False)
        raise ValueError("Unsupported arithmetic.")

    expression = read(tree)
    if expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Non-finite arithmetic.")
    return expression


def display_math(calculation, answer):
    """Format existing outputs only; never generate or solve an equation."""
    output = {"equations": []}
    for analysis in (calculation or {}).get("analyses", []):
        equations = []
        for equation in (analysis.get("system") or {}).get("equations", []):
            try:
                left, right = equation.split("=")
                equation = sp.Eq(arithmetic_expression(left.strip()), arithmetic_expression(right.strip()), evaluate=False)
                equations.append(sp.latex(equation, imaginary_unit="j"))
            except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
                continue
        row = {"target_id": analysis.get("target_id"), "equations": equations}
        try:
            row["exact_result"] = sp.latex(arithmetic_expression(analysis.get("exact_result")), imaginary_unit="j")
        except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
            pass
        if equations or "exact_result" in row:
            output["equations"].append(row)
    quantities = []
    for quantity in (answer or {}).get("quantities", []):
        try:
            real, imaginary = quantity.get("real"), quantity.get("imaginary")
            if type(real) not in (int, float) or type(imaginary) not in (int, float):
                continue
            value = arithmetic_expression(str(real)) + sp.I * arithmetic_expression(str(imaginary))
            name = arithmetic_expression(quantity.get("name") or quantity.get("id"))
            unit = quantity.get("unit") or ""
            if not isinstance(unit, str) or len(unit) > 40 or not re.fullmatch(r"[A-Za-z0-9 /Ωµμ°.-]*", unit):
                continue
            if quantity.get('kind')=='phasor':
                magnitude=math.hypot(real,imaginary)
                text=sp.latex(name)+r' &\approx '+format(magnitude,'.6g')
                if magnitude:
                    text+=r'\angle '+format(math.degrees(math.atan2(imaginary,real)),'.6g')+r'^{\circ}'
            else:
                text = sp.latex(name) + " &= " + sp.latex(value, imaginary_unit="j")
            if unit:
                text += r"\,\mathrm{" + unit + "}"
            quantities.append(text)
        except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
            continue
    if quantities:
        output["answer"] = r"\begin{aligned}" + r" \\ ".join(quantities) + r"\end{aligned}"
    return output


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def tail(path, limit=30000):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class RequestError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Application:
    def __init__(self, bundle_root=None, runs_dir=None, start_preflight=True):
        self.bundle = Path(bundle_root or Path(__file__).resolve().parent.parent).resolve()
        self.web = Path(__file__).resolve().parent
        self.runs_dir = Path(runs_dir or self.bundle / "UI-RUNS").resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.samples = {row["id"]: row for row in json.loads((self.bundle / "runtime/examples/demo/cases.json").read_text())}
        self.config = json.loads((self.bundle / "runtime/config/pipeline.json").read_text())
        sam_lock = json.loads((self.bundle / "runtime" / self.config["sam"]["lock"]).read_text())
        sam_checkpoint = sam_lock["artifacts"]["finetuned_checkpoint"]
        self.lock = threading.RLock()
        self.overlay_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self._cache = {}
        self.processes = {}
        self.active_id = None
        self.health = {"ready": False, "checking": False, "message": "Readiness check has not run.",
                       "models": {"detector": Path(self.config["detector"]["weights"]).name,
                                  "vision": configured_model(self.config["vision"]),
                                  "reasoning": configured_model(self.config["reasoning"]),
                                  "sam": sam_lock.get("provider_id") or Path(sam_checkpoint["path"]).name},
                       "providers": {"vision": self.config["vision"].get("provider", "local"),
                                     "reasoning": self.config["reasoning"].get("provider", "local"),
                                     "detector": "local", "sam": "local"}}
        self._detection_overlay = runpy.run_path(str(self.bundle / "runtime/demo"))["detection_overlay"]
        if start_preflight:
            self.preflight()

    def preflight(self):
        with self.lock:
            if self.health["checking"]:
                return dict(self.health)
            self.health.update(ready=False, checking=True, message="Checking configured models, credentials and pipeline files…")

        def check():
            try:
                result = subprocess.run([str(self.bundle / "check-ready")], cwd=self.bundle,
                                        capture_output=True, text=True, timeout=180)
                message = (result.stdout + result.stderr).strip()
                ready = result.returncode == 0
            except Exception as error:
                ready, message = False, f"Readiness check failed: {error}"
            with self.lock:
                self.health.update(ready=ready, checking=False,
                                   message=message or ("Ready." if ready else "Readiness check failed."))
        threading.Thread(target=check, daemon=True, name="demo-preflight").start()
        return dict(self.health)

    def sample_path(self, sample_id):
        if sample_id not in self.samples:
            raise RequestError(404, "Unknown prepared example.")
        return (self.bundle / "runtime" / self.samples[sample_id]["image"]).resolve()

    def _job_dir(self, run_id):
        if not SAFE_ID.fullmatch(run_id):
            raise RequestError(404, "Unknown run.")
        directory = self.runs_dir / run_id
        if not directory.is_dir() or directory.is_symlink() or not (directory / "job.json").is_file():
            raise RequestError(404, "Unknown run.")
        return directory

    def _metadata(self, run_id):
        return json.loads((self._job_dir(run_id) / "job.json").read_text())

    def _read_json(self, path):
        try:
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            with self.cache_lock:
                previous = self._cache.get(str(path))
                if previous and previous[0] == stamp:
                    return previous[1]
            value = json.loads(path.read_text())
            with self.cache_lock:
                self._cache[str(path)] = (stamp, value)
            return value
        except (OSError, ValueError):
            # A writer may be in the middle of saving a JSON artifact. Only a
            # previously complete version is displayable, never guessed data.
            with self.cache_lock:
                previous = self._cache.get(str(path))
                return previous[1] if previous else None

    def _active(self):
        if self.active_id:
            process = self.processes.get(self.active_id)
            if process and process.poll() is None:
                return self.active_id
            metadata = self._metadata(self.active_id)
            if metadata.get("status") in ("queued", "running"):
                return self.active_id
            self.active_id = None
        return None

    def start(self, fields, upload=None):
        sample_id = fields.get("sample_id", "").strip()
        question = fields.get("question", "").strip()
        if sample_id and upload is not None:
            raise RequestError(400, "Choose a prepared example or an uploaded image, not both.")
        if sample_id:
            source = self.sample_path(sample_id)
            content = source.read_bytes()
            question = question or self.samples[sample_id]["question"]
            label = self.samples[sample_id]["title"]
        elif upload is not None:
            _, content = upload
            label = "Uploaded circuit"
        else:
            raise RequestError(400, "Choose a circuit image.")
        if not question:
            raise RequestError(400, "Enter the question to answer about this circuit.")
        if len(question) > 16000:
            raise RequestError(400, "Question is too long (maximum 16,000 characters).")
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise RequestError(400, "Image exceeds 20 megapixels. Crop or resize it before uploading.")
                image_format = image.format
                if image_format not in ("PNG", "JPEG", "WEBP"):
                    raise RequestError(400, "Please upload a PNG, JPEG or WebP image.")
                image.verify()
        except RequestError:
            raise
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            raise RequestError(400, "The uploaded file is not a readable PNG, JPEG or WebP image.")
        with self.lock:
            if not self.health["ready"]:
                raise RequestError(503, self.health["message"])
            if self._active():
                raise RequestError(409, "A circuit is already running. Inspect it while it completes.")
            now = dt.datetime.now(ZoneInfo("Asia/Hong_Kong"))
            run_id = now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            directory = self.runs_dir / run_id
            directory.mkdir(exist_ok=False)
            suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[image_format]
            input_file = directory / ("original-upload" + suffix)
            input_file.write_bytes(content)
            metadata = {"id": run_id, "mode": "live", "status": "queued", "label": label,
                        "sample_id": sample_id or None, "question": question,
                        "started_at": now.isoformat(), "started_epoch": time.time(),
                        "input_file": input_file.name, "error": None}
            atomic_json(directory / "job.json", metadata)
            self.active_id = run_id
            threading.Thread(target=self._run, args=(run_id,), daemon=True, name="circuit-" + run_id).start()
            return {"id": run_id}

    def _run(self, run_id):
        directory = self._job_dir(run_id)
        metadata = self._metadata(run_id)
        command = [str(self.bundle / "run-demo"), "--image", str(directory / metadata["input_file"]),
                   "--question", metadata["question"], "--output", str(directory / "artifacts")]
        try:
            with (directory / "console.log").open("w") as log:
                process = subprocess.Popen(command, cwd=self.bundle, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                with self.lock:
                    self.processes[run_id] = process
                    metadata.update(status="running", pid=process.pid)
                    atomic_json(directory / "job.json", metadata)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                code = process.wait()
            record = self._read_json(directory / "artifacts/custom/run.json") or {}
            answer = self._read_json(directory / "artifacts/custom/answer.json") or {}
            metadata.update(status="completed" if record.get("status") == "completed" else "failed",
                            exit_code=code, answer_status=record.get("answer_status") or answer.get("status"),
                            error=record.get("error") or (None if record.get("status") == "completed" else f"Pipeline exited with code {code}. See console log."))
        except Exception as error:
            metadata.update(status="failed", error=f"Could not run the configured pipeline: {error}")
        finally:
            metadata["ended_epoch"] = time.time()
            with self.lock:
                atomic_json(directory / "job.json", metadata)
                if self.active_id == run_id:
                    self.active_id = None

    def _derive_overlay(self, directory):
        if (directory / "detections-overlay.png").exists():
            return
        if not (directory / "source.png").exists() or not self._read_json(directory / "detections.json"):
            return
        with self.overlay_lock:
            if (directory / "detections-overlay.png").exists():
                return
            try:
                self._detection_overlay(directory)
            except (OSError, ValueError, KeyError):
                # No replacement visualization is substituted. A complete
                # source/detection pair will be retried on the next poll.
                pass

    def _snapshot(self, run_id, metadata, directory, prefix, saved=False):
        if not saved:
            self._derive_overlay(directory)
        data = {key: self._read_json(directory / filename) for key, filename in DATA_FILES.items()}
        data["display_math"] = display_math(data["calculation"], data["answer"])
        data["reader_observations"] = {}
        for file in directory.glob("vision-components/*/vision-raw.json"):
            raw = self._read_json(file)
            if not isinstance(raw, dict):
                continue
            content = (raw.get("message") or {}).get("content")
            try:
                observation = json.loads(content) if isinstance(content, str) else None
            except ValueError:
                continue
            if isinstance(observation, dict):
                data["reader_observations"][file.parent.name] = observation
        record, answer = data["run"] or {}, data["answer"] or {}
        status = metadata["status"]
        if record.get("status") == "completed":
            status = "completed"
        elif record.get("status") == "failed":
            status = "failed"
        answer_status = record.get("answer_status") or answer.get("status") or metadata.get("answer_status")
        artifacts = []
        if directory.is_dir():
            for item in sorted(directory.rglob("*")):
                if not item.is_file() or item.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    relative = item.relative_to(directory).as_posix()
                    item.resolve().relative_to(directory.resolve())
                    stat = item.stat()
                    if stat.st_size:
                        artifacts.append({"path": relative, "url": prefix + quote(relative, safe="/"),
                                          "kind": EXTENSIONS[item.suffix.lower()],
                                          "version": f"{stat.st_mtime_ns}-{stat.st_size}"})
                except (ValueError, OSError):
                    continue
        available = {row["path"] for row in artifacts}
        readiness = ["source.png" in available, data["detections"] is not None,
                     "sam/mask.png" in available,
                     data["candidate"] is not None, data["readings"] is not None,
                     data["circuit"] is not None, data["linear_state"] is not None,
                     data["reasoning_plan"] is not None, data["calculation"] is not None,
                     data["spice_result"] is not None, data["explanation"] is not None,
                     data["answer"] is not None]
        timings = {stage["stage"]: stage["seconds"] for stage in record.get("stages", [])}
        blocked = bool((data["circuit"] or {}).get("electrical_issues"))
        terminal = status in ("completed", "failed")
        first_pending = next((i for i, ready in enumerate(readiness) if not ready), None)
        if blocked:
            first_pending = 11 if not readiness[11] else None
        failed_stage = {"detect": 1, "segment": 2, "graph": 3, "vision": 4,
                        "assemble": 5, "reasoning": 6, "spice": 9, "explain": 10, "prepare": 0}.get(record.get("failed_stage"))
        stages = []
        for i, ((key, label), ready) in enumerate(zip(STAGES, readiness)):
            state = "ready" if ready else "pending"
            if not ready:
                if blocked and 6 <= i <= 9:
                    state = "skipped"
                elif terminal:
                    state = "failed" if (failed_stage == i or (failed_stage is None and i == first_pending and status == "failed")) else "skipped"
                elif i == first_pending:
                    state = "running"
            if key == "spice" and data["spice_result"]:
                spice_status = data["spice_result"].get("status")
                if spice_status in ("failed", "error"):
                    state = "failed"
                elif spice_status in ("skipped", "unsupported", "not_supported", "not_run"):
                    state = "skipped"
            if key == "explain" and data["explanation"] and data["explanation"].get("status")!='available':
                state = "failed" if data["explanation"].get("error") else "skipped"
            stage = {"id": key, "label": label, "status": state}
            if key in timings:
                stage["seconds"] = timings[key]
            elif key == "spice" and data["spice_result"] and data["spice_result"].get("seconds") is not None:
                stage["seconds"] = data["spice_result"]["seconds"]
            stages.append(stage)
        detection_count = len((data["detections"] or {}).get("components", []))
        reader_count = sum(1 for file in directory.glob("vision-components/*/vision-raw.json") if self._read_json(file) is not None)
        reader_mode = (data["reader_mode"] or {}).get("mode")
        if reader_mode and "single" in reader_mode:
            reader_count = len((data["readings"] or {}).get("components", []))
        elif not reader_mode and reader_count:
            reader_mode = "per_component_then_global"
        elapsed = record.get("seconds") if terminal and "seconds" in record else max(0, metadata.get("ended_epoch", time.time()) - metadata.get("started_epoch", time.time()))
        console_path = (self.bundle / "LIVE-RUNS/PASS1" / (run_id + ".log")) if saved else self._job_dir(run_id) / "console.log"
        return {"id": run_id, "mode": "saved" if saved else "live", "status": status,
                "answer_status": answer_status, "label": metadata["label"], "question": metadata["question"],
                "started_at": metadata.get("started_at"), "elapsed_seconds": round(elapsed or 0, 2),
                "error": record.get("error") or metadata.get("error"), "stages": stages,
                "artifacts": artifacts, "data": data, "console": tail(console_path),
                "reader_progress": {"mode": reader_mode, "completed": reader_count, "total": detection_count},
                "source_url": prefix + "source.png" if "source.png" in available else None,
                "input_url": prefix + "source.png" if saved else "/inputs/" + run_id,
                "recorded_total_seconds": record.get("seconds"),
                "models": self.health["models"], "providers": self.health["providers"]}

    def snapshot(self, run_id):
        metadata = self._metadata(run_id)
        return self._snapshot(run_id, metadata, self._job_dir(run_id) / "artifacts/custom",
                              "/artifacts/" + run_id + "/")

    def backup(self, sample_id):
        if sample_id not in self.samples:
            raise RequestError(404, "Unknown saved example.")
        sample = self.samples[sample_id]
        directory = self.bundle / "LIVE-RUNS/PASS1" / sample_id
        if not (directory / "run.json").is_file():
            raise RequestError(404, "Saved rehearsal is unavailable.")
        metadata = {"status": "completed", "label": sample["title"], "question": sample["question"]}
        return self._snapshot(sample_id, metadata, directory, "/backup-artifacts/" + sample_id + "/", saved=True)

    def runs(self):
        rows = []
        for directory in sorted(self.runs_dir.iterdir(), reverse=True):
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                m = json.loads((directory / "job.json").read_text())
                record = self._read_json(directory / "artifacts/custom/run.json") or {}
                rows.append({k: m.get(k) for k in ("id", "mode", "status", "label", "started_at")})
                if record.get("status") in ("completed", "failed"):
                    rows[-1]["status"] = record["status"]
            except (OSError, ValueError):
                continue
        return {"runs": rows[:30], "backups": [{"id": k, "label": v["title"], "mode": "saved"} for k, v in self.samples.items()]}

    def artifact(self, root, relative):
        relative = unquote(relative)
        if not relative or "\\" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RequestError(404, "Artifact not found.")
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                raise ValueError()
        except (ValueError, OSError):
            raise RequestError(404, "Artifact not found.")
        return path


def make_server(app, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def _origin(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            if host not in allowed:
                raise RequestError(403, "This demo accepts localhost requests only.")
            origin = self.headers.get("Origin")
            if origin and (urlsplit(origin).scheme != "http" or urlsplit(origin).netloc not in allowed):
                raise RequestError(403, "Cross-origin requests are not accepted.")

        def send_data(self, data, content_type, filename=None):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if filename:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(filename, safe=""))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, value):
            return self.send_data(json_bytes(value), "application/json; charset=utf-8")

        def send_file(self, path):
            content_type = {".md": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8",
                            ".json": "application/json; charset=utf-8"}.get(path.suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            download = parse_qs(urlsplit(self.path).query).get("download") == ["1"]
            return self.send_data(path.read_bytes(), content_type, path.name if download else None)

        def error(self, error):
            self.send_response(error.status)
            data = json_bytes({"error": error.message})
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                self._origin()
                route = urlsplit(self.path).path
                if route in ("/", "/index.html", "/app.js", "/style.css"):
                    file = app.web / ("index.html" if route in ("/", "/index.html") else route[1:])
                    if not file.is_file():
                        raise RequestError(404, "The demo page is not available.")
                    return self.send_file(file)
                if route.startswith("/vendor/katex/"):
                    relative = unquote(route[len("/vendor/katex/"):])
                    if not relative or "\\" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                        raise RequestError(404, "Vendor asset not found.")
                    root = app.web / "vendor/katex"
                    file = root / relative
                    try:
                        file.resolve().relative_to(root.resolve())
                        if relative not in ("katex.min.js", "katex.min.css") and not (relative.startswith("fonts/") and file.suffix in (".woff2", ".woff", ".ttf")):
                            raise ValueError()
                        if not file.is_file():
                            raise ValueError()
                    except (ValueError, OSError):
                        raise RequestError(404, "Vendor asset not found.")
                    return self.send_file(file)
                if route == "/api/health":
                    with app.lock:
                        return self.send_json({**app.health, "active_run_id": app._active()})
                if route == "/api/samples":
                    return self.send_json([{**{k: v[k] for k in ("id", "title", "question")},
                                            "exposure": v["exposure"], "image_url": "/samples/" + v["id"] + "/source"}
                                           for v in app.samples.values()])
                if route == "/api/runs":
                    return self.send_json(app.runs())
                if route.startswith("/api/runs/"):
                    return self.send_json(app.snapshot(route[len("/api/runs/"):]))
                if route.startswith("/api/backups/"):
                    return self.send_json(app.backup(route[len("/api/backups/"):]))
                if route.startswith("/samples/") and route.endswith("/source"):
                    return self.send_file(app.sample_path(route[len("/samples/"):-len("/source")]))
                if route.startswith("/inputs/"):
                    run_id = route[len("/inputs/"):]
                    metadata = app._metadata(run_id)
                    return self.send_file(app._job_dir(run_id) / metadata["input_file"])
                if route.startswith("/artifacts/"):
                    run_id, separator, relative = route[len("/artifacts/"):].partition("/")
                    if not separator:
                        raise RequestError(404, "Artifact not found.")
                    return self.send_file(app.artifact(app._job_dir(run_id) / "artifacts/custom", relative))
                if route.startswith("/backup-artifacts/"):
                    sample_id, separator, relative = route[len("/backup-artifacts/"):].partition("/")
                    if not separator or sample_id not in app.samples:
                        raise RequestError(404, "Artifact not found.")
                    return self.send_file(app.artifact(app.bundle / "LIVE-RUNS/PASS1" / sample_id, relative))
                raise RequestError(404, "Not found.")
            except RequestError as error:
                self.error(error)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.error(RequestError(500, f"Could not read demo state: {error}"))

        def do_POST(self):
            try:
                self._origin()
                route = urlsplit(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_BODY:
                    raise RequestError(413, "Upload is too large (maximum request size 20 MiB).")
                if route == "/api/preflight":
                    if length:
                        self.rfile.read(length)
                    return self.send_json(app.preflight())
                if route != "/api/runs":
                    raise RequestError(404, "Not found.")
                content_type = self.headers.get("Content-Type", "")
                if not content_type.startswith("multipart/form-data;"):
                    raise RequestError(400, "Upload must use multipart form data.")
                if not length:
                    raise RequestError(400, "Upload body is empty.")
                body = self.rfile.read(length)
                message = BytesParser(policy=policy.default).parsebytes(("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
                if not message.is_multipart():
                    raise RequestError(400, "Invalid multipart upload.")
                fields, upload = {}, None
                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    payload = part.get_payload(decode=True) or b""
                    if name == "image" and part.get_filename() and payload:
                        if upload is not None:
                            raise RequestError(400, "Upload exactly one circuit image.")
                        upload = (part.get_filename(), payload)
                    elif name in ("sample_id", "question"):
                        fields[name] = payload.decode("utf-8", errors="strict")
                return self.send_json(app.start(fields, upload))
            except RequestError as error:
                self.error(error)
            except (ValueError, UnicodeError):
                self.error(RequestError(400, "Malformed upload."))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.error(RequestError(500, f"Could not start the run: {error}"))

        def log_message(self, format, *args):
            if "/api/" not in self.path:
                super().log_message(format, *args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    app = Application()
    server = make_server(app, args.port)
    print(f"K-Bot live demo: http://127.0.0.1:{server.server_port}", flush=True)
    print("New images invoke the configured pipeline. Saved rehearsals are explicitly labelled.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped. Saved run files are preserved.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
