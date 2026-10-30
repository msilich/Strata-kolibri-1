#!/usr/bin/env python3
"""Run frozen Kolibri fixtures through an unmodified standalone llama-server.

Uses exact prompt token IDs, natural EOS and no prompt/KV reuse. Each case gets
a fresh server and one cold plus two warm expert-cache requests by default.
Single GPU only: use CUDA device 0 without visibility remapping; telemetry
samples physical GPU 0. Model splitting across GPUs is disabled explicitly.
The Python imports below supply measurement helpers only; inference is llama.cpp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from benchmark import (ROOT, check_fixtures, cpu_name, gpu_info_local, json_write,
                       memory_sample, peak, sha256, storage_info,
                       total_ram_gib, utc)

REFERENCE = "9df0b73c707900b7277a46b5ae52ff7e7e567d13"
DEFAULT_CASES = ["german-2048", "german-8192", "german-16384", "german-32768",
                 "english-8192", "code-8192"]
COMPUTE_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUDA_DEVICE_MAX_CONNECTIONS",
               "CUDA_CACHE_DISABLE", "CUDA_CACHE_MAXSIZE", "CUDA_LAUNCH_BLOCKING",
               "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_CUDA_DISABLE_GRAPHS",
               "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS",
               "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY")


def server_command(args) -> list[str]:
    command = [str(args.server.resolve()), "--model", str(args.model.resolve()),
               "--host", "127.0.0.1", "--port", str(args.port), "--ctx-size", "32768",
               "--parallel", "1", "--batch-size", str(args.batch), "--ubatch-size", str(args.ubatch),
               "--threads", str(args.threads), "--threads-batch", str(args.threads),
               "--n-gpu-layers", "99", "--split-mode", "none", "--main-gpu", "0",
               "--flash-attn", "on", "--cache-type-k", "f16",
               "--cache-type-v", "f16", "--fit", "off", "--cache-ram", "0",
               "--ctx-checkpoints", "0", "--no-context-shift", "--no-warmup",
               "--log-verbosity", "4"]
    if args.cpu_moe_layers is None:
        command += ["--cpu-moe", "--moe-cache-mib", str(args.cache_mib)]
    else:
        command += ["--n-cpu-moe", str(args.cpu_moe_layers)]
    return command


def completion(url: str, payload: dict, timeout: float, path: Path) -> dict:
    request = urllib.request.Request(url + "/completion", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    started = time.monotonic()
    text, tokens, final, first = [], [], None, None
    with urllib.request.urlopen(request, timeout=timeout) as response, path.open("wb") as raw:
        for line in response:
            raw.write(line)
            raw.flush()
            if not line.startswith(b"data: "):
                continue
            if line[6:].strip() == b"[DONE]":
                break
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"])
            if data.get("tokens"):
                if first is None:
                    first = time.monotonic() - started
                tokens.extend(data["tokens"])
            text.append(data.get("content", ""))
            if data.get("stop"):
                final = data
    if final is None or first is None:
        raise RuntimeError("completion did not return generated tokens and a final result")
    return {"text": "".join(text), "tokens": tokens, "response": final,
            "time_to_first_token_s": first, "wall_seconds": time.monotonic() - started}


def run_case(args, case: dict, attempt: Path, identity: dict) -> dict:
    attempt.mkdir(parents=True)
    ids = json.loads((args.fixtures / case["tokens_file"]).read_text())
    command = server_command(args)
    result = {"case": case, "identity": identity, "command": command, "created": utc(),
              "status": "running", "trials": [], "memory_sampling_interval_s": 1,
              "os_file_cache": "not dropped; may be warm", "model_loading_in_throughput": False,
              "memory_note": "RSS includes mapped model pages. Total GPU memory includes the desktop. VmHWM is process-lifetime peak."}
    json_write(attempt / "result.json", result)
    started = time.monotonic()
    samples = []
    stopped = threading.Event()
    out = (attempt / "stdout.log").open("wb")
    err = (attempt / "stderr.log").open("wb")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
    result["pid"] = proc.pid

    def monitor():
        with (attempt / "memory.jsonl").open("w") as log:
            while not stopped.is_set():
                sample = memory_sample(proc.pid, started)
                samples.append(sample)
                log.write(json.dumps(sample) + "\n")
                log.flush()
                stopped.wait(1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    url = f"http://127.0.0.1:{args.port}"
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited before readiness: {proc.returncode}")
            if time.monotonic() - started > args.timeout:
                raise TimeoutError("server startup")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    health = json.load(response)
                if health.get("status") == "ok":
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.25)
        result["startup_seconds"] = time.monotonic() - started
        for trial in range(1, args.trials + 1):
            payload = {"prompt": ids, "n_predict": args.output, "temperature": 0,
                       "top_k": 128, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                       "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": 0,
                       "cache_prompt": False, "ignore_eos": False, "stream": True,
                       "return_tokens": True, "id_slot": 0}
            json_write(attempt / f"request-{trial}.json", payload)
            sample_begin = len(samples)
            active = {"trial": trial, "expert_cache_state": "cold" if trial == 1 else "warm",
                      "requested_output_tokens": args.output, "status": "running"}
            result["trials"].append(active)
            active.update(completion(url, payload, args.timeout, attempt / f"response-{trial}.sse"))
            response, timings = active["response"], active["response"]["timings"]
            generated = int(timings["predicted_n"])
            prompt_ms, decode_ms = float(timings["prompt_ms"]), float(timings["predicted_ms"])
            if timings["cache_n"] != 0 or timings["prompt_n"] != len(ids) or response["tokens_evaluated"] != len(ids):
                raise RuntimeError(f"prompt/KV reuse or incomplete prompt: {timings}")
            if response.get("truncated") or not 1 <= generated <= args.output or response["stop_type"] not in ("eos", "limit"):
                raise RuntimeError(f"incomplete generation: {response}")
            if response["stop_type"] == "limit" and generated != args.output:
                raise RuntimeError("generation limit differs from requested output cap")
            if not math.isfinite(prompt_ms) or prompt_ms <= 0 or not math.isfinite(decode_ms) or decode_ms < 0 or (generated > 1 and decode_ms == 0):
                raise RuntimeError(f"invalid server timing: {timings}")
            trial_samples = samples[sample_begin:]
            active.update(status="complete", generated_tokens=generated,
                          finish="stop" if response["stop_type"] == "eos" else "length",
                          prompt_tokens_read=len(ids), kv_prefix_reused_tokens=timings["cache_n"],
                          prompt_ms=prompt_ms, decode_ms=decode_ms,
                          prefill_tokens_per_s=1000 * len(ids) / prompt_ms,
                          decode_tokens_per_s=float(timings["predicted_per_second"]),
                          decode_forward_tokens_per_s=1000 * (generated - 1) / decode_ms if decode_ms else 0,
                          generated_tokens_per_decode_second=1000 * generated / decode_ms if decode_ms else 0,
                          stream_token_ids_complete=len(active["tokens"]) == generated,
                          peak_rss_kib=peak(trial_samples, "VmRSS_kib"),
                          peak_swap_kib=peak(trial_samples, "VmSwap_kib"),
                          peak_gpu_total_mib=peak(trial_samples, "gpu_memory.used"),
                          peak_gpu_process_mib=peak(trial_samples, "gpu_process_memory_mib"),
                          memory_sample_count=len(trial_samples))
            # Native SSE may suppress intermediate token IDs while assembling
            # incomplete UTF-8 bytes. Text and server generation counts remain complete.
            active["output_sha256"] = hashlib.sha256(active["text"].encode()).hexdigest()
            json_write(attempt / "result.json", result)
            print(json.dumps({"case": case["name"], **{key: active[key] for key in
                  ("trial", "generated_tokens", "finish", "prefill_tokens_per_s", "decode_tokens_per_s")}}), flush=True)
        result["status"] = "complete"
        result["deterministic_across_trials"] = len({t["output_sha256"] for t in result["trials"]}) == 1
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        if result["trials"] and result["trials"][-1]["status"] == "running":
            result["trials"][-1].update(status="failed", error=repr(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        stopped.set()
        watcher.join(timeout=12)
        out.close()
        err.close()
        result.update(exit_code=proc.returncode, elapsed_seconds=time.monotonic() - started,
                      peak_process_lifetime_rss_kib=peak(samples, "VmHWM_kib"),
                      peak_gpu_total_mib=peak(samples, "gpu_memory.used"))
        json_write(attempt / "result.json", result)
    return result


def summarise(out: Path, identity: dict) -> None:
    rows, attempts = [], []
    for path in sorted(out.glob("*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        if result["identity"] != identity:
            raise RuntimeError(f"mixed benchmark identities in {path}")
        attempts.append({"path": str(path.relative_to(out)), "case": result["case"]["name"],
                         "status": result["status"], "error": result.get("error")})
        if result["status"] != "complete":
            continue
        for state in ("cold", "warm"):
            trials = [t for t in result["trials"] if t["expert_cache_state"] == state]
            if not trials:
                continue
            row = {"case": result["case"]["name"], "prompt_tokens": result["case"]["prompt_tokens"],
                   "expert_cache_state": state, "trials": len(trials),
                   "generated_tokens": [t["generated_tokens"] for t in trials]}
            for key in ("prefill_tokens_per_s", "decode_tokens_per_s", "generated_tokens_per_decode_second",
                        "time_to_first_token_s", "wall_seconds"):
                values = [t[key] for t in trials]
                row[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
            rows.append(row)
    json_write(out / "summary.json", {"identity": identity, "attempts": attempts, "rows": rows,
               "note": "Standalone pinned llama.cpp, real greedy generation with natural EOS; no prompt/KV reuse. Native decode rate counts generated-1 forwards. Cold and warm expert-cache trials are separate; OS file cache is not dropped."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "source", "model", "fixtures", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--ubatch", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=24576)
    parser.add_argument("--cpu-moe-layers", type=int, help="static expert placement calibration; disables native cache")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=int, default=256)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or not 1 < args.output <= 256 or not 1 <= args.ubatch <= args.batch:
        parser.error("require trials >= 1, output in 2..256, and 1 <= ubatch <= batch")
    manifest = check_fixtures(args.fixtures, args.model)
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    source_status = subprocess.check_output(["git", "-C", str(args.source), "status", "--short"], text=True)
    if revision != REFERENCE or source_status:
        raise RuntimeError("reference source must be clean at pinned revision")
    identity = {"reference_revision": revision, "server_sha256": sha256(args.server),
                "fixture_manifest_sha256": sha256(args.fixtures / "manifest.json"),
                "model_sha256": manifest["model"]["sha256"], "command": server_command(args),
                "trials_per_case": args.trials, "output_cap": args.output,
                "harness_sha256": sha256(Path(__file__)), "measurement_helpers_sha256": sha256(Path(__file__).with_name("benchmark.py")),
                "adjacent_libraries": {p.name: sha256(p) for p in args.server.parent.glob("*.so.*") if p.is_file()}}
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if not args.resume or metadata["identity"] != identity:
            raise RuntimeError("existing output requires --resume and identical settings/binaries/fixtures")
    else:
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                             "--format=csv,noheader,nounits"], text=True)
        if processes.strip():
            raise RuntimeError(f"GPU busy; wait for an exclusive slot: {processes}")
        metadata = {"identity": identity, "created": utc(), "command": sys.argv,
                    "model": manifest["model"], "cpu": cpu_name(), "ram_gib": total_ram_gib(),
                    "gpu_placement": "Single CUDA device 0, --split-mode none --main-gpu 0; telemetry samples physical GPU 0. CUDA visibility must not remap device 0.",
                    "gpu": gpu_info_local(), "storage": storage_info(), "os": platform.platform(),
                    "environment": {key: os.environ[key] for key in COMPUTE_ENV if key in os.environ},
                    "timing_note": "Native llama decode rate counts generated-1 decode forwards. generated_tokens_per_decode_second also records generated/decode_s for comparison with Strata's current counter. Loading is excluded. TTFT is the first delivered SSE token chunk over local HTTP; assembling incomplete UTF-8 may delay delivery after the first internal token.",
                    "prefix_cache": "cache_prompt=false; assert timings.cache_n=0 and prompt_n=full fixture length",
                    "cold_note": "Fresh process per case; --no-warmup avoids dummy inference populating native expert cache. OS file cache is not dropped."}
        metadata["cmake_cache"] = (args.server.parent.parent / "CMakeCache.txt").read_text()
        json_write(metadata_path, metadata)
    selected = [c for c in manifest["cases"] if c["name"] in args.cases]
    if set(args.cases) != {c["name"] for c in selected}:
        raise ValueError("unknown fixture case")
    for case in selected:
        previous = sorted((args.out / case["name"]).glob("attempt-*/result.json"))
        if any(json.loads(p.read_text())["status"] == "complete" for p in previous):
            continue
        attempt = args.out / case["name"] / f"attempt-{len(previous) + 1:02d}"
        result = run_case(args, case, attempt, identity)
        summarise(args.out, identity)
        if result["status"] != "complete":
            raise RuntimeError(f"case failed; evidence retained at {attempt}")


if __name__ == "__main__":
    main()
