# Install and build from a source checkout

Use this guide to work on Strata's code or maintain a source-based installation. It is for developers.
To just use Strata (version 0.2.5), install the [desktop app or the Terminal](INSTALL.md). Installed apps do not use `START-HERE.bat`.

The repository includes `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat`, `setup.sh` and `update.sh`.
Run them from a cloned or extracted **source folder**. They may install dependencies, prepare model files and
acquire or build an engine. Check [model requirements](MODELS.md) before downloading.

## Quick start

| I use | Run in the source folder |
| --- | --- |
| Windows | `START-HERE.bat` ([steps](#from-source-windows)) |
| Linux | `./setup.sh` ([steps](#from-source-linux)) |
| Mac (Apple Silicon) | `./setup.sh` ([steps](#from-source-mac)) |
| A source build to test a change | `./setup.sh --build --no-start` ([steps](#build-from-git-on-linux-to-try-a-pull-request-or-change-the-code)) |
| An update | `UPDATE.bat` or `./update.sh` ([steps](#updating)) |

## From source: Windows

1. Clone the [repository](https://github.com/Niko1221/Strata) or download and extract its source archive.
2. Open the source folder and run `START-HERE.bat`.
3. Choose a supported model, storage folder and settings. Keep the window open while setup works.

For an explicit original ISTA-DASLab Flash-Next pack, from Command Prompt in that folder:

```bat
START-HERE.bat --family qwen --model Q2_0
```

Choose a pack that fits your machine; Q2_0 is an example, not a universal minimum-memory option.
In PowerShell, prefix the wrapper with `.\`, for example `.\START-HERE.bat`.
`SETUP.bat` opens setup again to add a model or change settings. The wrapper can reuse an existing install;
first-start and later load times depend on the pack, disk, RAM and backend. A model can take several minutes to load.
Run again over an installed model, setup pre-selects the size that is installed and marks it "(installed)", and `--yes` takes it: a repeat run
does not move the model to the recommended size.

When no ready-made engine fits the PC, setup compiles one. That needs the Visual Studio 2022 Build Tools with the C++ workload;
Visual Studio 2019 is too old (the sources are C++20). With a CUDA toolkit, CUDA 13.0 to 13.2 accept Visual Studio 2022 only, and
Visual Studio 2026 works from CUDA 13.3. The CPU image encoder alone (no CUDA toolkit on the PC) builds with MSVC and takes any
Visual Studio that has the C++ tools. If setup cannot find an install it could use, name the file before running it, for example
`set STRATA_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`. Setup uses that file
only if it exists; a wrong path is answered as no build tools at all.

## From source: Linux

Clone or extract the source, enter its folder, then run:

```bash
./setup.sh
```

On Debian and Ubuntu, the wrapper can offer to install missing system packages through `sudo apt`.
Other distributions need the equivalent dependencies installed first. Python dependencies use the checkout's
`.venv`. NVIDIA source builds need the CUDA toolkit; AMD builds need HIP/ROCm. See [Linux details](DETAILS.md#linux).

Nix/NixOS users can enter the supplied development environment:

```bash
nix develop -c ./setup.sh
```

### Build from git on Linux (to try a pull request or change the code)

```bash
git clone https://github.com/Niko1221/Strata.git
cd Strata
./setup.sh --build --no-start
```

`--build` requests a source engine rather than a ready-made one. `--no-start` prevents serving;
this is still setup and can download and prepare a model. Inspect the printed paths and generated config before
claiming to have tested a code change. A released engine cannot validate changes to your local C++ files.

For a direct NVIDIA build, with the required compiler and CUDA toolkit installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target strata -j8
```

`native` builds for the GPUs in this PC. Name the card instead for another one: `75` (RTX 20), `86` (RTX 30), `89` (RTX 40),
`120` (RTX 50), or a list such as `"86;120"`. Without the option the build is for RTX 50 only and the configure step says so.

Run against the resulting binary and the matching model config. Use the build instructions for
[HIP](AMD_HIP.md), [SYCL](INTEL_ARC.md) or [Metal](MAC.md) when changing those backends.
Follow [contribution validation requirements](../CONTRIBUTING.md) before requesting review.

For the Terminal monitor from a checkout:

```bash
.venv/bin/pip install -r requirements-monitor.txt
STRATA_PYTHON="$PWD/.venv/bin/python3" bin/strata monitor --url http://127.0.0.1:8080
```

## From source: Mac

Use an Apple Silicon Mac with macOS 15 or newer. Install Apple's Command Line Tools once:

```bash
xcode-select --install
```

Then run `./setup.sh` in the source folder. macOS uses its own setup implementation and builds the Metal engine;
its options and model preparation differ from Windows/Linux. Read [MAC.md](MAC.md#install) and
`./setup.sh --help`. The released app and Terminal do not require a compiler.

## AMD cards

Use `START-HERE.bat` on Windows or `./setup.sh` on Linux. Source setup detects supported Radeon cards;
`--backend hip` selects HIP explicitly on a mixed-vendor PC. Build dependencies, architecture support and
measurements are in [AMD_HIP.md](AMD_HIP.md). Ryzen AI Max uses the [Strix Halo guide](STRIX_HALO.md).

Do not assume every AMD card, quantization or feature has been validated. Follow the exact backend guide.
Intel Arc is a separate Linux SYCL path with its own [model preparation instructions](INTEL_ARC.md).

## Two or three cards

Source setup uses `--gpus` to select multiple cards of the same supported backend:

```bash
./setup.sh --gpus 0,1
```

Windows equivalent: `START-HERE.bat --gpus 0,1`. The installed Terminal instead uses
`strata launch --gpu 0,1`. Main/Helper roles, memory limits and measurements are in [MULTI_GPU.md](MULTI_GPU.md).
Several cards are not always faster than one; measure the workload you care about.

## Docker (Linux)

Use [DOCKER.md](DOCKER.md) for released images, GPU access and the container build instructions.
A container is a separate installation route. Intel model preparation and host driver requirements still apply.

## Older CPUs (experimental)

Ready-made x86 engines require AVX2. Source setup can build experimental AVX or SSE4.2 paths on older CPUs;
this does not make every model pack compatible. Q2_0's CPU requirements differ from the i-quants.
See the [CPU and build notes](DETAILS.md) and [older GPU guide](OLDER_GPUS.md) for the relevant hardware.
Record the actual CPU, compiler, model and validation results when reporting an experimental build.

## Experimental Kolibri-1 (Linux, NVIDIA CUDA)

The optional source engine loads Kolibri-1 Q4_K_M directly from GGUF. It computes
on one GPU and transfers experts from RAM when they are not in the GPU cache.
It does not requantize the model. `--max-context` accepts up to 262,144 tokens
including the answer; the default remains 32,768, with one request at a time.
The shared HTTP server keeps its usual
eight-token reserve within that limit. CPU expert computation, other GPU
backends, vision and speculative decoding are not implemented in this path.
Source setup supports this experimental path on Linux with one NVIDIA GPU of
compute capability 8.0 or newer and a CUDA toolkit. Measurements so far are
from an RTX 5090 only. This is a source feature, not a model in the released app
or `strata launch`.

Run `./setup.sh --setup` and choose **Kolibri-1** in the model-family menu, or
select it explicitly:

```sh
./setup.sh --setup --family kolibri --model Q4_K_M --context 262144 --yes --no-start
```

Setup offers 32k, 64k, 128k and 256k context capacities; the default is 32k.
These limits include the answer. To use the pinned GGUF you already have, add
`--gguf-dir /path/to/Kolibri-1`, naming the folder containing
`Kolibri-1-Q4_K_M.gguf`.

Setup downloads the pinned file when needed, checks its size and SHA-256, and
exports its tokenizer and chat template. It builds the optional engine in
`build-kolibri/` and installs it with its matching library in `engine-kolibri/`.
It writes `strata-kolibri-q4_k_m.json` and `run-kolibri-q4_k_m.sh` in the checkout.
The command above prepares everything without starting the server. On the next
run, use `./setup.sh` to choose the installed model, or run
`./run-kolibri-q4_k_m.sh` directly.

For a direct development build, configure CUDA as described above and enable
the optional target:

```sh
cmake -S . -B build -DSTRATA_BUILD_KOLIBRI=ON
cmake --build build --target strata-kolibri kolibri-math-test kolibri-model-test -j 4
./build/kolibri-math-test
python tools/download_kolibri.py --connections 32
```

The download script pins the Bartowski GGUF revision, verifies its SHA-256 and
resumes interrupted downloads. It downloads about 49.76 GB without a bandwidth
cap. The default destination is `../Strata-data/models/Kolibri-1/`. Export the
tokenizer from that same file:

```sh
python tools/strata_tokenizer.py \
  --gguf ../Strata-data/models/Kolibri-1/Kolibri-1-Q4_K_M.gguf \
  --out ../Strata-data/packs/kolibri-q4_k_m
./build/kolibri-model-test ../Strata-data/models/Kolibri-1/Kolibri-1-Q4_K_M.gguf
```

Create a run config outside the checkout, replacing each absolute path below.
Use the locally compiled executable: released engines do not include this path.

```json
{
  "exe": "/path/to/Strata/build/strata-kolibri",
  "args": ["--native", "/path/to/Strata-data/models/Kolibri-1/Kolibri-1-Q4_K_M.gguf", "--max-context", "32768"],
  "arch": "kolibri1",
  "tokenizer": "/path/to/Strata-data/packs/kolibri-q4_k_m/tokenizer",
  "model_name": "kolibri-1-q4_k_m",
  "cwd": "/path/to/Strata",
  "log": "/path/to/Strata-data/kolibri.log",
  "parallel": 1,
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
strata serve --config /path/to/Strata-data/kolibri.json --plan
strata serve --config /path/to/Strata-data/kolibri.json
```

The existing chat API handles Kolibri's chat template, reasoning and JSON tool
calls. Its sampling defaults are temperature 1.0, top-p 0.97 and top-k 128;
explicit request settings take precedence. Use `reasoning_effort: "none"` to
disable reasoning. An explicit reasoning effort takes precedence over
`enable_thinking`, as in Kolibri's own template.

GPU cache space is determined after the fixed weights and workspaces are
allocated. `--expert-vram N` limits it to N GiB; `--expert-slots 0` disables it
for comparison. By default the full configured context is allocated at startup.
Increasing it leaves less GPU memory for experts, even with a short prompt. Each request
reads its complete history again. Long-context
latency and cache effectiveness depend on the workload; a successful build
alone is not a model accuracy or performance measurement.

`--kv-grow` optionally uses Strata's existing CUDA virtual-memory support for
global keys and values. It initially maps up to 32,768 tokens, then grows in
32k steps as needed, up to the configured context. Local attention is unchanged.
This lets shorter requests keep more experts in GPU memory. When growth needs
space, the engine empties and rebuilds a smaller expert cache, preserving the
existing keys and values and the 768 MiB memory reserve. Growing and rewarming
the expert cache take time. The mapped capacity and smaller cache persist across
requests until the engine exits. Allocation failures are reported explicitly;
the engine never silently reduces the configured context. The flag is off by
default and requires CUDA virtual-memory support.

For a numerical comparison, `tools/kolibri/parity.py --help` documents a
separate llama.cpp reference build pinned to
`9df0b73c707900b7277a46b5ae52ff7e7e567d13`. It compares layers, logits and
optional routing traces on identical token IDs. Keep this reference separate
from Strata's existing llama.cpp dependency.

`--split-k-attention` is an optional optimization for global attention while
generating an answer. It divides the visible keys among GPU blocks and combines
their results in FP32. It applies from 1,024 keys onward. Local attention and
prompt processing use the existing path. Add the flag to the run config's
`args` to enable it; it is off by default.
The changed reduction order can cross activation-quantization rounding
boundaries and change generated tokens, including with greedy sampling.
The optimized path is not bitwise identical to the original path.

`--split-k-local-attention` separately enables FP32 split attention during
generation once the local window contains all 513 keys. Shorter local windows
and prompt processing keep their existing path. It is off by default and can
change generated tokens through its different reduction order.

`--grouped-experts` is a separate optional decode optimization. It processes
the six selected experts together and reuses their quantized input. It supports
Q4_K, Q6_K and Q8_0 expert weights, including mixed gate/up formats. Prompt
processing uses the existing path. The grouped path needs additional GPU
buffers; these are allocated before the automatic expert-cache budget is set.

`--grouped-prefill` separately batches prompt experts, including mixed
Q4_K, Q6_K and Q8_0 gate/up formats. Each group contains at most 16 experts and
one chunk's routed rows, including experts selected by only one token.
Cache admission and expert accumulation order are unchanged. Small batches use
MMQ instead of MMVQ; this changes rounding and can change generated tokens.
The flag is off by default and adds
about 128 MiB of workspace for this Q4_K_M model, or about 104 MiB when grouped
decode already provides part of the staging space. These buffers are allocated
before the automatic expert-cache budget is set.

`--fused-decode` combines projection launches and residual-normalization steps
during single-token generation. It is designed to preserve the existing FP32
rounding and produce bitwise-equal outputs with the same attention settings.
It works independently of `--grouped-experts`; enabling both also reuses the
shared expert's quantized input for the six selected experts. Prompt processing
keeps its existing path. The flag is off by default.
The fused path also uses a single warp to select the same six experts. With
grouped experts enabled, it downloads their IDs while the shared expert runs.
It swaps local KV buffers after copying their retained history, avoiding a
second copy. This needs 8 MiB of extra GPU scratch space at chunk size 4,096.

The `--fused-decode` and `--prefill-attention` paths compute each position's
RoPE sine/cosine table once per forward pass and reuse it across local layers.
This preserves the existing angle calculation and FP32 rounding. The table
adds at most 2 MiB of GPU memory with a 4,096-token chunk.

`--gpu-greedy` chooses the next token on the GPU for serving requests with
temperature zero, no effective repetition/frequency/presence penalties, no
logit bias and no requested log probabilities. It preserves the lowest-token-ID
tie rule and rejects non-finite logits. Other requests use the existing CPU
sampler, including seeded sampling; diagnostic full-logit output is unchanged.
The flag is off by default and is independent of the attention and expert flags.

`--decode-cache-lru` requires `--grouped-experts`. During generation it replaces
the least recently used resident expert when a cache miss needs a slot. All six
experts selected for the current layer are protected from replacement. Prompt
processing retains its existing cache-admission behavior.

`--pin-experts` optionally keeps all expert weights in CUDA-registered RAM on
Linux. The Q4_K_M model used here needs about 43.62 GiB for these weights.
Startup requires that amount plus 8 GiB of available RAM. It copies the weights
in small batches and releases their original file-cache pages without changing
the GGUF. Registration uses normal pages for these allocations; system memory
settings are unchanged. This adds preparation time and keeps that RAM occupied
until the engine exits. Experimental startup checks stop preparation if available
RAM falls below 8 GiB, system swap use grows by more than 256 MiB, or the engine
has more than 64 MiB in swap. These checks do not run during generation.

`--overlap-prefill-uploads` requires `--pin-experts --grouped-prefill`. It overlaps
expert transfers with prompt computation on a separate CUDA stream. The existing
expert order and arithmetic stay the same. `--gpu-expert-cache` requires
`--pin-experts --grouped-experts --decode-cache-lru --fused-decode`. During
generation the GPU selects cache slots and copies missing experts from registered
RAM. The CPU reconciles the cache after each token. Layer and routing diagnostic
dumps retain the CPU-managed path. All three flags are off by default.

`--overlap-decode-cache` requires `--gpu-expert-cache --overlap-prefill-uploads`.
During generation it prepares the selected expert weights while the shared
expert computes, using the existing upload stream. The expert calculations and
cache updates stay the same. Prompt processing is unchanged. It is off by default.

The optional `--prefill-attention` kernel uses tensor cores and requires CUDA
compute capability 8.0 or newer. It keeps FP16 keys/values and represents queries
and probabilities with a pair of half-precision values. Its rounding differs
from the original FP32 attention path. `--chunk` accepts up to 4,096 prompt
tokens; the default remains 256. Larger chunks also change expert batching and
can change generated tokens.

`--two-query-prefill` requires `--prefill-attention` and is off by default. It
shares each global-attention key/value tile between two prompt tokens while
keeping each token's arithmetic and causal masking. Local attention, generation
and prompt chunks containing only one token keep their existing paths.

`--tensor-decode-attention` enables tensor-core attention during generation,
independently of `--prefill-attention`. It requires compute capability 8.0 or
newer and applies to global attention and full 513-key local windows. Where it
applies, it takes precedence over both split-attention flags. Shorter local
windows keep their existing path. Like the tensor-core prompt path, it uses
paired half-precision queries and probabilities, changes rounding, and can
change generated tokens. It is off by default.

The custom workload runner in `tools/kolibri/benchmark.py` compares German and
English synthetic project dossiers and pinned source-code excerpts. It prepares
four input sizes per workload, reserving 256 output tokens within 2k, 8k, 16k
and 32k budgets. These are developer measurements, not the standard Qwen
benchmark protocol. Use `prepare --help` to freeze the prompts, then run both
executables against the same fixture directory:

```sh
python tools/kolibri/benchmark.py run --model MODEL.gguf \
  --engine /path/to/baseline/strata-kolibri --fixtures FIXTURES --out BASELINE_RESULTS
python tools/kolibri/benchmark.py run --model MODEL.gguf \
  --engine /path/to/candidate/strata-kolibri --engine-arg=--split-k-attention \
  --fixtures FIXTURES --out CANDIDATE_RESULTS
```

Keep the matching `libstrata_kq.so` beside each executable. Run one engine at a
time. The runner defaults to 32,768 tokens of context; `--context` records a
different capacity separately from the prompt length. The expert cache defaults to
24 GiB; `--expert-vram` records a different budget as a separate experiment.
Prompt chunks default to 256; use `--chunk 512`, `1024`, `2048` or `4096` for
separate experiments. Additional VRAM is needed for the rest of the
model and its working buffers. Each case starts a fresh engine and makes three
requests. The first has
an empty expert cache; the next two retain it. The OS file cache is not cleared,
and every request processes the full prompt. Reports retain early EOS, actual
token counts, separate prompt and generation times, and sampled RAM/VRAM.

`--profile` writes diagnostic stage timings to stderr. These include CUDA
stream intervals and CPU submission time, which overlap and must not be added
together. Profiling adds overhead; leave it off for throughput comparisons.

`tools/kolibri/llama_benchmark.py` runs the same frozen token IDs through an
unmodified llama.cpp server at the reference revision above. It disables prompt
reuse and records native timings, generated text and memory samples. Use
`--help` for placement and batch settings. The script records both llama.cpp's
native decode rate and the rate using Strata's token-count convention; the
first output token accounts for the difference.

## Updating

Stop the running source engine before replacing it. Run `UPDATE.bat` on Windows or `./update.sh` on Linux.
In a Git checkout the updater can pull source changes, update dependencies and refresh installed configs without
starting the model. A source ZIP cannot pull Git history: extract a newer source archive instead.
Preserve your configs and data location when moving between checkouts.

If you changed engine code, rebuild it and verify the executable path again. The presence of updated source files
is not evidence that a running engine contains those changes. Desktop and Terminal component updates are covered
separately in [INSTALL.md](INSTALL.md#updating).

## Where things are stored

Source setup writes model run configs (`strata-<model>.json`), launch scripts (`run-<model>.bat` / `.sh`) and
logs in the checkout. Models, packs and draft data normally go in `Strata-data` beside it, or the location selected
with `--data-dir`. Dependencies and build products stay in the source installation.

These are source paths. The desktop app and installed Terminal use the [release data locations](INSTALL.md#where-things-are-stored).
Back up a config before editing it: setup can rewrite the fields it owns. Do not publish API keys or private paths in logs.

## Setup's questions

Setup detects the machine and offers compatible choices. The list varies by platform, available memory and model
catalog; it is not a fixed two-model menu. Read the proposed download, memory use and settings before accepting.
For documentation recommendations, prefer ISTA-DASLab when a compatible pack fits, then Unsloth, Gyro and others.
Executable defaults may differ, so select the family explicitly when that choice matters.

## Tuning for your PC

Use source setup's `--calibrate` only where that backend supports it. Terminal tuning instead uses
`strata launch --tune quick` or `--tune thorough`; those are separate interfaces. Tuning can take a substantial
amount of time and should use the same hardware and model configuration as the intended workload.

## Options without questions

Run the relevant wrapper with `--help` for the complete options on that operating system. Common source examples:

```bat
START-HERE.bat --family qwen --model IQ3_XXS --context 32768 --vision no
START-HERE.bat --setup
```

```bash
./setup.sh --family qwen --model IQ3_XXS --context 32768 --vision no
./setup.sh --no-start
```

`--yes` accepts setup's recommended answers and can start large downloads. Use it only after choosing the pack
and storage location. Keep source flags (`--gpus`, `--vision yes|no`, numeric `--context`) separate from
[Terminal flags](TERMINAL_LAUNCH.md) and raw engine options.
