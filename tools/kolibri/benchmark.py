#!/usr/bin/env python3
"""Custom Linux/CUDA Kolibri workload matrix; not the standard Qwen protocol.

Prepare immutable fixtures without loading the engine, then use the same directory
for baseline and candidate runs. Each case starts a fresh process and sends three
identical greedy requests: one cold expert-cache trial and two warm trials. KV
state resets on every GEN; OS file cache is not dropped. Natural EOS is retained.
Source excerpts default to HEAD, resolved once to a full commit. Use
--source-revision COMMIT when preparing fixtures for a particular source version.

  python tools/kolibri/benchmark.py prepare --model MODEL --template TEMPLATE \
    --model-sha256 VERIFIED_SHA256 --fixtures build/kolibri-fixtures
  python tools/kolibri/benchmark.py run --model MODEL --engine EXECUTABLE \
    --fixtures build/kolibri-fixtures --out build/kolibri-baseline

Resume skips only complete cases with matching binary, fixtures and settings.
Incomplete attempts remain on disk and are repeated in a new process directory.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "serve"))
from corpus import prose
from frontend import ChatTemplate
from strata_tokenizer import Tokenizer
from strata_bench import cpu_name, gpu_info_local, storage_info, total_ram_gib

CODE_FILES = ["strata_control/atomic.py", "strata_control/private_files.py",
              "strata_control/paths.py", "strata_control/profiles.py",
              "strata_control/owner_transport.py", "strata_control/loopback.py",
              "strata_control/readiness.py", "strata_control/gguf_preflight.py",
              "strata_control/acquisition.py", "strata_control/runtime.py"]
CONTEXTS = (2048, 8192, 16384, 32768)
OUTPUT_CAP = 256
CACHE_RE = re.compile(r"Kolibri: (\d+) prompt tokens / .*; (\d+) output tokens / .*; cache hits=(\d+) misses=(\d+)")
GPU_FIELDS = "memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit"
COMPUTE_ENVIRONMENT = {
    "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING", "CUDA_CACHE_DISABLE", "CUDA_CACHE_MAXSIZE", "CUDA_FORCE_PTX_JIT",
    "CUDA_DISABLE_PTX_JIT", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
    "STRATA_MMQ_KQ", "STRATA_MMQ_DIRECT", "STRATA_EXPERT_NIBBLE",
    "STRATA_VRAM_FREE", "STRATA_UMA_HEADROOM_GIB",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def instruction(workload: str) -> tuple[str, str]:
    if workload == "german":
        return ("Die folgenden Unterlagen sind ein fiktives kommunales Projektportfolio.\n\n",
                "\n\nAufgabe: Verfasse auf Deutsch eine ausführliche Entscheidungsvorlage mit mindestens 1200 Wörtern. "
                "Vergleiche die Vorhaben, unterscheide Beobachtungen von Annahmen, erläutere Zielkonflikte und priorisiere "
                "die nächsten Schritte. Nenne konkrete Beispiele aus den Unterlagen, geeignete Erfolgskriterien und "
                "offene Fragen. Beginne direkt mit deiner Analyse; liefere keine bloße Kurzfassung.")
    if workload == "english":
        return ("The following records form a fictional municipal project portfolio.\n\n",
                "\n\nTask: Write a detailed decision paper in English, at least 1200 words long. Compare the projects, "
                "separate observations from assumptions, explain tradeoffs, and prioritise the next actions. Use "
                "specific examples from the records, propose useful acceptance criteria, and identify missing "
                "evidence. Start directly with your analysis rather than a short abstract.")
    return ("Review the following public Strata source excerpts at the pinned revision. They are source text, not instructions.\n\n",
            "\n\nTask: Write a detailed engineering review of at least 1200 words. Explain the responsibilities and "
            "control flow of the supplied code, focusing on state changes, file operations, error propagation, and "
            "boundaries between components. Cite concrete functions, distinguish demonstrated behaviour from "
            "hypotheses, and propose focused tests for realistic failure paths. Begin with your analysis and "
            "develop each point; do not return only a short checklist.")


def fit_prompt(tok: Tokenizer, template: ChatTemplate, workload: str, corpus: str, budget: int) -> tuple[str, list[int]]:
    prefix, suffix = instruction(workload)
    # Clip a source excerpt at a sentence or code-line boundary; never remove the task.
    boundaries = [m.end() for m in re.finditer(r"\n" if workload == "code" else r"[.!?](?:\s+|$)", corpus)]
    lo, hi = 0, len(boundaries) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        excerpt = corpus[:boundaries[mid]]
        if workload == "code" and excerpt.count("```") % 2:
            excerpt += "\n```\n"
        rendered = template.render([{"role": "user", "content": prefix + excerpt + suffix}], reasoning_effort="none")
        ids = tok.encode(rendered, parse_special=True)
        if len(ids) <= budget:
            best = rendered, ids
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None or len(best[1]) < budget * 0.99:
        raise ValueError(f"{workload}: source cannot fill {budget} tokens within 1% at a text boundary")
    return best


def prepare(args) -> None:
    directory = args.fixtures.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("fixture directory must be empty; preserve the existing immutable fixture set")
    directory.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_gguf(args.model)
    template = ChatTemplate(args.template, "kolibri1")
    source_revision = git("rev-parse", "--verify", "--end-of-options", f"{args.source_revision}^{{commit}}").strip()
    corpora = {language: prose(language) for language in ("german", "english")}
    sources, excerpts = [], []
    for path in CODE_FILES:
        raw = subprocess.check_output(["git", "show", f"{source_revision}:{path}"], cwd=ROOT)
        sources.append({"path": path, "revision": source_revision, "sha256": hashlib.sha256(raw).hexdigest()})
        excerpts.append(f"\nSource: {path} at {source_revision}\n```python\n{raw.decode('utf-8')}\n```\n")
    corpora["code"] = "\n".join(excerpts)
    model_stat = args.model.stat()
    manifest = {"schema": 1, "kind": "custom-kolibri-workload-matrix", "created": utc(),
                "source_revision": source_revision, "code_sources": sources,
                "model": {"filename": args.model.name, "sha256": args.model_sha256,
                          "sha256_provenance": "supplied previously verified digest",
                          "size": model_stat.st_size, "mtime_ns": model_stat.st_mtime_ns},
                "corpus_description": "Original synthetic German/English municipal dossiers: 20 projects, 12 recurring report forms, distinct record contents; pinned public repository code. Prose has repeated document structure, not repeated filler blocks. Source prefixes end at sentence or code-line boundaries.",
                "tokenizer": {"source": "GGUF", "architecture": tok.arch, "pre": tok.pre,
                              "vocabulary_size": len(tok.tokens), "special_ids": tok.special_ids},
                "template_sha256": sha256(args.template), "reasoning_effort": "none",
                "generator_sha256": sha256(Path(__file__)), "corpus_generator_sha256": sha256(Path(__file__).with_name("corpus.py")),
                "tokenizer_implementation_sha256": sha256(ROOT / "tools/strata_tokenizer.py"),
                "cases": [], "files": {}}
    (directory / "chat_template.jinja").write_bytes(args.template.read_bytes())
    for workload, corpus in corpora.items():
        (directory / f"corpus-{workload}.txt").write_text(corpus, encoding="utf-8")
        for context in CONTEXTS:
            name = f"{workload}-{context}"
            rendered, ids = fit_prompt(tok, template, workload, corpus, context - OUTPUT_CAP)
            (directory / f"{name}.txt").write_text(rendered, encoding="utf-8")
            json_write(directory / f"{name}.tokens.json", ids)
            manifest["cases"].append({"name": name, "workload": workload, "context_budget": context,
                                      "prompt_budget": context - OUTPUT_CAP, "prompt_tokens": len(ids),
                                      "prompt_file": f"{name}.txt", "tokens_file": f"{name}.tokens.json"})
            print(f"prepared {name}: {len(ids)} prompt tokens + {OUTPUT_CAP} output cap", flush=True)
    for path in sorted(directory.iterdir()):
        manifest["files"][path.name] = sha256(path)
    json_write(directory / "manifest.json", manifest)


def check_fixtures(directory: Path, model: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"fixture changed: {name}")
    current = model.stat()
    if (current.st_size, current.st_mtime_ns) != (manifest["model"]["size"], manifest["model"]["mtime_ns"]):
        raise ValueError("model size/mtime changed since fixtures were frozen; verify model and prepare new fixtures")
    return manifest


def read_queue(lines: queue.Queue, deadline: float, description: str):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(description)
    try:
        item = lines.get(timeout=remaining)
    except queue.Empty:
        raise TimeoutError(description) from None
    if item is None:
        raise RuntimeError(f"engine pipe closed: {description}")
    return item


def memory_sample(pid: int, start: float) -> dict:
    sample = {"elapsed_s": time.monotonic() - start}
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        for key in ("VmRSS", "VmHWM", "VmSwap"):
            match = re.search(rf"^{key}:\s+(\d+)", status, re.M)
            sample[key + "_kib"] = int(match[1]) if match else None
        gpu = subprocess.check_output(["nvidia-smi", "--id=0", f"--query-gpu={GPU_FIELDS}",
                                       "--format=csv,noheader,nounits"], text=True, timeout=5)
        for name, value in zip(GPU_FIELDS.split(","), gpu.strip().split(",")):
            try:
                sample["gpu_" + name] = float(value.strip())
            except ValueError:
                sample["gpu_" + name] = value.strip()
        process_rows = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_memory",
                                                "--format=csv,noheader,nounits"], text=True, timeout=5)
        sample["gpu_process_memory_mib"] = None
        for row in process_rows.splitlines():
            fields = row.split(",")
            if len(fields) == 2 and fields[0].strip() == str(pid):
                sample["gpu_process_memory_mib"] = float(fields[1].strip())
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        sample["error"] = repr(error)
    return sample


def peak(samples: list[dict], key: str):
    values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
    return max(values) if values else None


def run_case(args, case: dict, tok: Tokenizer, attempt: Path, identity: dict) -> dict:
    attempt.mkdir(parents=True)
    ids = json.loads((args.fixtures / case["tokens_file"]).read_text())
    command = [str(args.engine.resolve()), "--serve", "--native", str(args.model.resolve()),
               "--max-context", str(args.context), "--chunk", str(args.chunk), "--expert-vram", str(args.expert_vram), *args.engine_arg]
    result = {"case": case, "identity": identity, "command": command, "created": utc(),
              "status": "running", "trials": [], "memory_sampling_interval_s": 1,
              "memory_note": "RSS includes resident file-backed model pages; total GPU memory includes non-engine allocations. Process GPU memory is sampled separately. VmHWM is process lifetime high-water, not trial-local.",
              "os_file_cache": "not dropped; may be warm from previous runs", "model_loading_in_throughput": False}
    json_write(attempt / "result.json", result)
    started = time.monotonic()
    samples, readers = [], []
    lines, counters = queue.Queue(), queue.Queue()
    stopped = threading.Event()
    proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    result["pid"] = proc.pid

    def pipe_reader(pipe, path, output_queue, is_stderr=False):
        with path.open("w", encoding="utf-8") as log:
            for line in pipe:
                log.write(line)
                log.flush()
                if is_stderr:
                    match = CACHE_RE.search(line)
                    if match:
                        output_queue.put((tuple(map(int, match.groups())), line.rstrip()))
                else:
                    output_queue.put(line.rstrip())
        output_queue.put(None)

    def monitor():
        with (attempt / "memory.jsonl").open("w", encoding="utf-8") as stream:
            while not stopped.is_set():
                sample = memory_sample(proc.pid, started)
                samples.append(sample)
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                stopped.wait(1)

    for pipe, path, target, is_stderr in ((proc.stdout, attempt / "stdout.log", lines, False),
                                         (proc.stderr, attempt / "stderr.log", counters, True)):
        thread = threading.Thread(target=pipe_reader, args=(pipe, path, target, is_stderr), daemon=True)
        thread.start()
        readers.append(thread)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    active = None
    try:
        while True:
            line = read_queue(lines, started + args.timeout, "waiting for READY")
            if line.startswith("ERR"):
                raise RuntimeError(line)
            if line.startswith("READY "):
                if line != f"READY {args.context} stop":
                    raise ValueError(f"unexpected engine readiness: {line}")
                break
        result["startup_seconds"] = time.monotonic() - started
        previous_hits = previous_misses = 0
        for trial in range(1, 4):
            begin = time.monotonic()
            sample_begin = len(samples)
            active = {"trial": trial, "expert_cache_state": "cold" if trial == 1 else "warm",
                      "kv_prefix_reused_tokens": 0, "requested_output_tokens": OUTPUT_CAP,
                      "tokens": [], "prefill_progress_lines": [], "status": "running"}
            result["trials"].append(active)
            proc.stdin.write(f"GEN {OUTPUT_CAP} temperature=0 cvec=1 " + ",".join(map(str, ids)) + "\n")
            proc.stdin.flush()
            while True:
                line = read_queue(lines, begin + args.timeout, f"waiting for DONE ({case['name']}, trial {trial})")
                if line.startswith("ERR"):
                    raise RuntimeError(line)
                if line.startswith("PP "):
                    active["prefill_progress_lines"].append(line)
                elif line.startswith("T "):
                    if not active["tokens"]:
                        active["time_to_first_token_s"] = time.monotonic() - begin
                    token = int(line.split()[1])
                    if not 0 <= token < len(tok.tokens):
                        raise ValueError(f"token outside vocabulary: {token}")
                    active["tokens"].append(token)
                elif line.startswith("DONE "):
                    active["done"] = line
                    active["wall_seconds"] = time.monotonic() - begin
                    fields = line.split()
                    if len(fields) != 16 or int(fields[1]) != len(active["tokens"]) or int(fields[2]) != len(ids):
                        raise ValueError(f"malformed DONE: {line}")
                    if int(fields[14]) != len(ids) or fields[5] not in ("stop", "length"):
                        raise ValueError(f"incomplete request: {line}")
                    prompt_ms, decode_ms = float(fields[3]), float(fields[4])
                    if not all(math.isfinite(v) and v > 0 for v in (prompt_ms, decode_ms)):
                        raise ValueError(f"invalid timings: {line}")
                    cache, cache_line = read_queue(counters, time.monotonic() + 10, "waiting for cache counters")
                    read, generated, hits, misses = cache
                    if (read, generated) != (len(ids), len(active["tokens"])) or hits < previous_hits or misses < previous_misses:
                        raise ValueError(f"inconsistent cache counters: {cache_line}")
                    if not 0 < generated <= OUTPUT_CAP or (fields[5] == "length" and generated != OUTPUT_CAP):
                        raise ValueError(f"invalid generated length: {line}")
                    trial_samples = samples[sample_begin:]
                    active.update(status="complete", generated_tokens=generated, finish=fields[5],
                                  prompt_tokens_read=read, prompt_ms=prompt_ms, decode_ms=decode_ms,
                                  prefill_tokens_per_s=1000 * read / prompt_ms,
                                  decode_tokens_per_s=1000 * generated / decode_ms,
                                  text=tok.decode(active["tokens"]),
                                  cache_hits=hits - previous_hits, cache_misses=misses - previous_misses,
                                  cumulative_cache_hits=hits, cumulative_cache_misses=misses,
                                  cache_log=cache_line, peak_rss_kib=peak(trial_samples, "VmRSS_kib"),
                                  peak_swap_kib=peak(trial_samples, "VmSwap_kib"),
                                  peak_gpu_total_mib=peak(trial_samples, "gpu_memory.used"),
                                  peak_gpu_process_mib=peak(trial_samples, "gpu_process_memory_mib"),
                                  memory_sample_count=len(trial_samples))
                    active["output_sha256"] = hashlib.sha256(json.dumps(active["tokens"]).encode()).hexdigest()
                    previous_hits, previous_misses = hits, misses
                    json_write(attempt / "result.json", result)
                    print(json.dumps({"case": case["name"], **{k: active[k] for k in
                                      ("trial", "expert_cache_state", "generated_tokens", "finish", "prefill_tokens_per_s", "decode_tokens_per_s", "cache_hits", "cache_misses")}}), flush=True)
                    break
        proc.stdin.close()
        if proc.wait(timeout=30) != 0:
            raise RuntimeError(f"engine exited {proc.returncode}")
        result["status"] = "complete"
        result["deterministic_across_trials"] = len({t["output_sha256"] for t in result["trials"]}) == 1
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        if active is not None and active["status"] != "complete":
            active["status"] = "failed"
            active["text"] = tok.decode(active["tokens"])
            active["error"] = repr(error)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        stopped.set()
        watcher.join(timeout=12)
        for reader in readers:
            reader.join(timeout=5)
        result.update(exit_code=proc.returncode, elapsed_seconds=time.monotonic() - started,
                      peak_process_lifetime_rss_kib=peak(samples, "VmHWM_kib"),
                      peak_gpu_total_mib=peak(samples, "gpu_memory.used"))
        json_write(attempt / "result.json", result)
    return result


def summarise(out: Path, identity: dict) -> dict:
    rows = []
    attempts = []
    for path in sorted(out.glob("*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        if result["identity"] != identity:
            raise ValueError(f"mixed run identities in {path}")
        attempts.append({"path": str(path.relative_to(out)), "case": result["case"]["name"], "status": result["status"], "error": result.get("error")})
        if result["status"] != "complete":
            continue
        for state in ("cold", "warm"):
            trials = [t for t in result["trials"] if t["expert_cache_state"] == state]
            row = {"case": result["case"]["name"], "prompt_tokens": result["case"]["prompt_tokens"],
                   "expert_cache_state": state, "trials": len(trials),
                   "generated_tokens": [t["generated_tokens"] for t in trials]}
            for key in ("prefill_tokens_per_s", "decode_tokens_per_s", "time_to_first_token_s", "wall_seconds"):
                values = [t[key] for t in trials]
                row[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
            rows.append(row)
    summary = {"identity": identity, "attempts": attempts, "rows": rows,
               "note": "One cold and two warm expert-cache trials per case; cold and warm are summarised separately. Natural EOS, no output padding; timings exclude loading. Custom synthetic-prose and pinned-code matrix, not a standard benchmark or quality evaluation."}
    json_write(out / "summary.json", summary)
    return summary


def run(args) -> None:
    manifest = check_fixtures(args.fixtures, args.model)
    selected = [c for c in manifest["cases"] if not args.cases or c["name"] in args.cases]
    if args.cases and set(args.cases) != {c["name"] for c in selected}:
        raise ValueError("unknown case name")
    for case in selected:
        if case["prompt_tokens"] + OUTPUT_CAP > args.context:
            raise ValueError(f"{case['name']}: prompt + output cap exceeds configured context")
    engine = args.engine.resolve()
    identity = {"engine_sha256": sha256(engine), "fixture_manifest_sha256": sha256(args.fixtures / "manifest.json"),
                "model_sha256": manifest["model"]["sha256"], "context": args.context, "chunk": args.chunk,
                "expert_cache_gib": args.expert_vram, "trials_per_case": 3, "output_cap": OUTPUT_CAP,
                "temperature": 0, "eos_suppressed": False, "engine_args": args.engine_arg,
                "harness_sha256": sha256(Path(__file__)),
                "engine_adjacent_libraries": {p.name: sha256(p) for p in engine.parent.glob("*.so*") if p.is_file()}}
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if not args.resume or metadata["identity"] != identity:
            raise ValueError("existing output requires --resume and identical binary, fixtures, harness and settings")
    else:
        gpu_processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                                 "--format=csv,noheader,nounits"], text=True)
        if gpu_processes.strip():
            raise RuntimeError(f"GPU compute process already running; wait for an exclusive slot: {gpu_processes.strip()}")
        metadata = {"identity": identity, "created": utc(), "command": sys.argv,
                    "engine_path": str(engine), "model": manifest["model"],
                    "git_head": git("rev-parse", "HEAD").strip(), "git_status": git("status", "--short"),
                    "tracked_patch": git("diff", "--", ":(exclude)AGENTS.md"),
                    "cpu": cpu_name(), "ram_gib": total_ram_gib(), "gpu": gpu_info_local(),
                    "storage": storage_info(), "os": platform.platform(), "python": sys.version,
                    "initial_compute_processes": gpu_processes, "background": args.background,
                    "os_file_cache": "not dropped", "environment": {k: v for k, v in os.environ.items() if k in COMPUTE_ENVIRONMENT},
                    "engine_adjacent_libraries": identity["engine_adjacent_libraries"]}
        json_write(metadata_path, metadata)
    tok = Tokenizer.from_gguf(args.model)
    for case in selected:
        previous = sorted((args.out / case["name"]).glob("attempt-*/result.json"))
        if any(json.loads(p.read_text())["status"] == "complete" for p in previous):
            print(f"resume: {case['name']} already complete", flush=True)
            continue
        # Partial and failed attempts remain in their own directories.
        attempt = args.out / case["name"] / f"attempt-{len(previous) + 1:02d}"
        result = run_case(args, case, tok, attempt, identity)
        summarise(args.out, identity)
        if result["status"] != "complete":
            raise RuntimeError(f"case {case['name']} failed; evidence retained in {attempt}")
    summarise(args.out, identity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare", help="freeze fixtures, no GPU work")
    prepare_parser.add_argument("--model", type=Path, required=True)
    prepare_parser.add_argument("--template", type=Path, required=True)
    prepare_parser.add_argument("--model-sha256", required=True)
    prepare_parser.add_argument("--fixtures", type=Path, required=True)
    prepare_parser.add_argument("--source-revision", default="HEAD", help="revision for public code excerpts; resolved to a full commit (default: HEAD)")
    run_parser = sub.add_parser("run", help="run the matrix in exclusive GPU time")
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--engine", type=Path, required=True)
    run_parser.add_argument("--fixtures", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--context", type=int, default=32768,
                            help="allocated context capacity, 1..262144 (default: 32768); recorded separately from fixture length")
    run_parser.add_argument("--chunk", type=int, choices=(256, 512, 1024, 2048, 4096), default=256,
                            help="prompt chunk size; compare each setting under its own run identity")
    run_parser.add_argument("--expert-vram", type=float, default=24,
                            help="expert-cache budget in GiB (default: 24); recorded in the run identity")
    run_parser.add_argument("--engine-arg", action="append", default=[], help="repeat for each extra engine argument; use --engine-arg=--flag")
    run_parser.add_argument("--cases", nargs="+", help="optional case names, e.g. german-2048 german-32768")
    run_parser.add_argument("--timeout", type=float, default=1800, help="startup/per-request timeout, seconds")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--background", default="No competing GPU compute process at startup; desktop GPU allocations may remain.")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not re.fullmatch(r"[0-9a-f]{64}", args.model_sha256):
            parser.error("model SHA256 must be a 64-character lowercase hexadecimal digest")
        prepare(args)
    else:
        if not 1 <= args.context <= 262144:
            parser.error("context capacity must be in 1..262144")
        if not math.isfinite(args.expert_vram) or args.expert_vram <= 0:
            parser.error("expert-cache budget must be positive and finite")
        if any(arg.split("=", 1)[0] in {"--native", "--model", "--max-context", "--chunk", "--expert-vram", "--expert-slots"}
               for arg in args.engine_arg):
            parser.error("extra arguments must not override fixed model/context/chunk/cache settings")
        run(args)


if __name__ == "__main__":
    main()
