"""Compare Kolibri's layers and logits with a pinned llama.cpp on identical token IDs.

Build tools/kolibri/ref_dump.cpp against llama.cpp revision
9df0b73c707900b7277a46b5ae52ff7e7e567d13. It reuses the generic dumper in
tools/qwen35/q36_llama_dump.cpp and enables the reference's GPU expert cache:

  g++ -O2 -std=c++17 tools/kolibri/ref_dump.cpp -I<LLAMA>/include \
    -I<LLAMA>/ggml/include -L<LLAMA_BUILD>/bin -Wl,-rpath,<LLAMA_BUILD>/bin \
    -lllama -lggml -lggml-base -lggml-cpu -o build/kolibri-reference-gpu-dump

Export the GGUF tokenizer with tools/strata_tokenizer.py and pass its chat template.
The reference generates a continuation, which Strata reads with teacher forcing.
Both executables run sequentially so their model allocations do not overlap.

Example:
  python tools/kolibri/parity.py --strata build/strata-kolibri \
    --reference build/kolibri-reference-gpu-dump --model MODEL.gguf \
    --template PACK/tokenizer/chat_template.jinja --out build/kolibri-parity \
    --ngl 99 --reference-moe-cache-mib 24576 --reference-ubatch 1 --chunk 1

--trace-routing also compares both routers with stable top-6(logits + GGUF bias).
The pinned cache accepts microbatches up to 32. The generic layer dumper retains
only the last microbatch of a prompt; zero-filled uncaptured rows are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serve"))
sys.path.insert(0, str(ROOT / "tools"))
from frontend import ChatTemplate
from gguf_reader import GGUFFile

N_LAYER, N_EMBD = 50, 2560
REFERENCE_REVISION = "9df0b73c707900b7277a46b5ae52ff7e7e567d13"
PROMPTS = [
    "Erkläre in einem kurzen Satz, warum der Himmel blau ist.",
    "Explain in one short sentence why the sky is blue.",
]


def run(command: list[str], env: dict[str, str], log: Path) -> str:
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800)
    log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}; see {log}")
    return result.stdout


def token_ids(output: str, name: str) -> list[int]:
    for line in output.splitlines():
        if line.startswith(name + ":"):
            return [int(value) for value in line.split(":", 1)[1].split()]
    raise RuntimeError(f"missing {name}: in executable output")


def binary_info(path: str) -> dict:
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(Path(path).resolve()), "sha256": digest}


def logits_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError("logit shapes differ or logits contain non-finite values")
    difference = actual - expected
    ref_top1 = expected.argmax(axis=1)
    strata_top1 = actual.argmax(axis=1)
    ref_top2 = np.partition(expected, -2, axis=1)[:, -2:]
    matches = strata_top1 == ref_top1
    divergences = np.flatnonzero(~matches)
    logp = expected - expected.max(axis=1, keepdims=True)
    logq = actual - actual.max(axis=1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
    logq -= np.log(np.exp(logq).sum(axis=1, keepdims=True))
    return {
        "steps": len(actual),
        "vocabulary": actual.shape[1],
        "max_absolute_error": float(np.abs(difference).max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)),
        "mean_kl_reference_to_strata": float(np.mean(np.sum(np.exp(logp) * (logp - logq), axis=1))),
        "top1_matches": int(matches.sum()),
        "top1_agreement": float(matches.mean()),
        "reference_top1": ref_top1.tolist(),
        "strata_top1": strata_top1.tolist(),
        "reference_top1_margin": (ref_top2[:, 1] - ref_top2[:, 0]).tolist(),
        "first_top1_difference": int(divergences[0]) if len(divergences) else None,
    }


def layer_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError("layer shapes differ or layers contain non-finite values")
    norms = np.linalg.norm(expected, axis=2)
    # llama.cpp skips prompt output rows it does not need in the final layer.
    present = norms > 0
    if not present.any(axis=0).all():
        raise RuntimeError("reference did not capture every layer")
    difference = actual - expected
    relative = np.linalg.norm(difference, axis=2) / np.maximum(norms, 1e-30)
    per_layer = []
    for layer in range(N_LAYER):
        mask = present[:, layer]
        delta = difference[mask, layer]
        per_layer.append({
            "layer": layer,
            "positions": int(mask.sum()),
            "max_relative_l2": float(relative[mask, layer].max()),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_absolute_error": float(np.abs(delta).max()),
        })
    worst = max(per_layer, key=lambda item: item["max_relative_l2"])
    return {
        "positions": len(actual),
        "max_relative_l2": worst["max_relative_l2"],
        "worst_layer": worst["layer"],
        "median_relative_l2": float(np.median(relative[present])),
        "per_layer": per_layer,
    }


def routing_metrics(output: Path, tag: str, model_path: str) -> dict:
    actual = np.fromfile(output / f"{tag}.strata.router", dtype="<f4").reshape(-1, N_LAYER, 384)
    actual_ids = np.fromfile(output / f"{tag}.strata.ids", dtype="<i4").reshape(-1, N_LAYER, 6)
    valid = np.zeros(actual.shape[:2], dtype=bool)
    directory = output / f"{tag}.reference.trace"

    def load_trace(kind: str, width: int, dtype: str = "<f4") -> np.ndarray:
        result = np.zeros((*actual.shape[:2], width), dtype=dtype)
        for layer in range(N_LAYER):
            suffix = "i32" if dtype == "<i4" else "f32"
            values = np.fromfile(directory / f"{kind}-{layer}.{suffix}", dtype=dtype).reshape(-1, width)
            if not 0 < len(values) <= len(actual):
                raise RuntimeError("reference routing trace has an unexpected position count")
            result[-len(values):, layer] = values
            valid[-len(values):, layer] = True
        return result

    reference = load_trace("ffn_moe_logits", 384)
    biased = load_trace("ffn_moe_logits_biased", 384)
    reference_ids = load_trace("ffn_moe_topk", 6, "<i4")
    weights = load_trace("ffn_moe_weights", 6)
    model = GGUFFile(Path(model_path))
    tensors = {tensor.name: tensor for tensor in model.tensors}
    bias = np.empty((N_LAYER, 384), dtype=np.float32)
    with model.path.open("rb") as stream:
        for layer in range(N_LAYER):
            tensor = tensors[f"blk.{layer}.exp_probs_b.bias"]
            if tensor.type_id != 0 or tensor.shape != [384]:
                raise RuntimeError("unexpected GGUF router bias type or shape")
            stream.seek(model.data_start + tensor.offset)
            bias[layer] = np.frombuffer(stream.read(384 * 4), dtype="<f4")
    actual_biased = actual + bias
    expected_ids = np.argsort(-actual_biased, axis=-1, kind="stable")[..., :6]
    expected_reference_ids = np.argsort(-biased, axis=-1, kind="stable")[..., :6]
    different_sets = valid & np.any(np.sort(actual_ids, axis=-1) != np.sort(reference_ids, axis=-1), axis=-1)
    differences = np.argwhere(different_sets)
    first = None
    if len(differences):
        position, layer = map(int, differences[0])
        scores = np.sort(biased[position, layer])
        first = {"position": position, "layer": layer,
                 "strata_ids": actual_ids[position, layer].tolist(),
                 "reference_ids": reference_ids[position, layer].tolist(),
                 "max_logit_difference": float(np.abs(actual[position, layer] - reference[position, layer]).max()),
                 "reference_sixth_seventh_margin": float(scores[-6] - scores[-7])}
    expected_weights = 1.0 / (1.0 + np.exp(-np.take_along_axis(reference, reference_ids, axis=-1).astype(np.float64)))
    return {
        "captured_position_layers": int(valid.sum()),
        "strata_rank_errors_against_own_scores": int(np.count_nonzero(actual_ids != expected_ids)),
        "reference_rank_errors_against_own_scores": int(np.count_nonzero(reference_ids[valid] != expected_reference_ids[valid])),
        "reference_bias_max_error": float(np.abs(biased[valid] - (reference + bias)[valid]).max()),
        "reference_sigmoid_weight_max_error": float(np.abs(weights[valid] - expected_weights[valid]).max()),
        "different_selected_sets": int(different_sets.sum()),
        "first_different_set": first,
        "per_layer_different_sets": different_sets.sum(axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--ngl", type=int, default=28, help="reference GPU layers; remaining layers run on CPU")
    parser.add_argument("--reference-moe-cache-mib", type=int,
                        help="GPU expert cache for tools/kolibri/ref_dump.cpp; use --ngl 99 for all GPU layers")
    parser.add_argument("--reference-ubatch", type=int,
                        help="reference microbatch 1..32 for tools/kolibri/ref_dump.cpp")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--expert-vram", type=float, default=24.0)
    parser.add_argument("--prompt", action="append", help="user prompt; repeat for multiple cases")
    parser.add_argument("--reuse-reference", action="store_true", help="reuse existing reference dumps and logs")
    parser.add_argument("--trace-routing", action="store_true", help="capture and validate both routers with ref_dump.cpp")
    parser.add_argument("--max-logit-rmse", type=float)
    parser.add_argument("--max-layer-relative-error", type=float)
    parser.add_argument("--min-top1-agreement", type=float)
    args = parser.parse_args()
    if not 1 <= args.n or not 1 <= args.chunk <= 256 or not args.n < args.context <= 32768:
        parser.error("require n >= 1, chunk 1..256, and n < context <= 32768")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    template = ChatTemplate(args.template, family="kolibri1")
    env = dict(os.environ, STRATA_DUMP_CTX=str(args.context))
    if args.reference_moe_cache_mib is not None:
        if args.reference_moe_cache_mib < 1:
            parser.error("reference MoE cache must be positive")
        env["STRATA_DUMP_MOE_CACHE_MIB"] = str(args.reference_moe_cache_mib)
    if args.reference_ubatch is not None:
        if not 1 <= args.reference_ubatch <= 32:
            parser.error("reference microbatch must be 1..32")
        env["STRATA_DUMP_UBATCH"] = str(args.reference_ubatch)
    summary = {
        "required_reference_revision": REFERENCE_REVISION,
        "executables": {"strata": binary_info(args.strata), "reference": binary_info(args.reference)},
        "model": str(Path(args.model).resolve()),
        "reference_gpu_layers": args.ngl,
        "reference_moe_cache_mib": args.reference_moe_cache_mib,
        "reference_ubatch": args.reference_ubatch,
        "context": args.context,
        "chunk": args.chunk,
        "teacher_forced": True,
        "cases": [],
    }
    failed = False
    for index, prompt in enumerate(args.prompt or PROMPTS):
        tag = f"p{index:02d}"
        rendered = template.render([{"role": "user", "content": prompt}], enable_thinking=False)
        reference_log = output / f"{tag}.reference.log"
        run_env = dict(env)
        if args.trace_routing:
            run_env["STRATA_DUMP_TENSORS"] = str(output / f"{tag}.reference.trace")
        inputs_path = output / f"{tag}.inputs.json"
        inputs = {"rendered_prompt": rendered, "model": summary["model"], "steps": args.n,
                  "context": args.context, "reference_gpu_layers": args.ngl,
                  "reference_binary_sha256": summary["executables"]["reference"]["sha256"]}
        if args.reference_moe_cache_mib is not None:
            inputs["reference_moe_cache_mib"] = args.reference_moe_cache_mib
        if args.reference_ubatch is not None:
            inputs["reference_ubatch"] = args.reference_ubatch
        if args.reuse_reference:
            if json.loads(inputs_path.read_text(encoding="utf-8")) != inputs:
                raise RuntimeError("cached reference was produced with different inputs")
            reference_output = reference_log.read_text(encoding="utf-8").split("\n--- stderr ---\n", 1)[0]
        else:
            reference_output = run([
                args.reference, "--model", args.model, "--prompt", rendered, "--n", str(args.n),
                "--ngl", str(args.ngl), "--dump-layers", str(output / f"{tag}.reference.layers"),
                "--dump-full-logits", str(output / f"{tag}.reference.logits"),
            ], run_env, reference_log)
            inputs_path.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
        ids = token_ids(reference_output, "prompt")
        reference_gen = token_ids(reference_output, "gen")
        if len(reference_gen) != args.n or len(ids) + args.n > args.context:
            raise RuntimeError("reference step count or prompt length does not match the requested run")
        command = [
            args.strata, "--model", args.model, "--tokens", ",".join(map(str, ids)), "--n", str(args.n),
            "--max-context", str(args.context), "--chunk", str(args.chunk),
            "--expert-vram", str(args.expert_vram), "--dump-layers", str(output / f"{tag}.strata.layers"),
            "--dump-full-logits", str(output / f"{tag}.strata.logits"),
        ]
        if len(reference_gen) > 1:
            command.extend(["--force", ",".join(map(str, reference_gen[:-1]))])
        if args.trace_routing:
            command.extend(["--dump-router", str(output / f"{tag}.strata.router"),
                            "--dump-route-ids", str(output / f"{tag}.strata.ids")])
        run(command, env, output / f"{tag}.strata.log")
        logits = [np.fromfile(output / f"{tag}.{engine}.logits", dtype="<f4").astype(np.float64).reshape(args.n, -1)
                  for engine in ("strata", "reference")]
        layers = [np.fromfile(output / f"{tag}.{engine}.layers", dtype="<f4").astype(np.float64).reshape(-1, N_LAYER, N_EMBD)
                  for engine in ("strata", "reference")]
        record = {"prompt": prompt, "prompt_ids": ids, "forced_ids": reference_gen[:-1],
                  "logits": logits_metrics(*logits), "layers": layer_metrics(*layers)}
        if args.trace_routing:
            record["routing"] = routing_metrics(output, tag, args.model)
            routing = record["routing"]
            failed |= bool(routing["strata_rank_errors_against_own_scores"] or routing["reference_rank_errors_against_own_scores"])
        summary["cases"].append(record)
        lg, ly = record["logits"], record["layers"]
        failed |= args.max_logit_rmse is not None and lg["rmse"] > args.max_logit_rmse
        failed |= args.max_layer_relative_error is not None and ly["max_relative_l2"] > args.max_layer_relative_error
        failed |= args.min_top1_agreement is not None and lg["top1_agreement"] < args.min_top1_agreement
        print(f"{tag}: {len(ids)} prompt tokens; top-1 agrees {lg['top1_matches']}/{lg['steps']}; "
              f"logit RMSE {lg['rmse']:.6g}, max abs {lg['max_absolute_error']:.6g}; "
              f"layer max relative L2 {ly['max_relative_l2']:.6g} at layer {ly['worst_layer']}", flush=True)
        for layer in ly["per_layer"]:
            print(f"  layer {layer['layer']:02d}: max relative L2 {layer['max_relative_l2']:.6g}; "
                  f"RMSE {layer['rmse']:.6g}; {layer['positions']} positions", flush=True)
        (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
