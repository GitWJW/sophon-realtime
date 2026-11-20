# Community-maintained Flash deployment on Ascend 910C

> This experimental recipe is maintained by community contributors.
> The core maintainers do not have Ascend hardware to test it.
> Validation applies to the checkpoint, source revisions and runtime recorded below.
> For issues, include the model revision, runtime versions, deployment configuration
> and relevant logs.

This recipe documents HTTP streaming inference experiments with the
32B Flash model on one physical 910C card exposing two NPU devices. It does not
establish support for every Flash variant or full-duplex multi-session serving.
The configuration was exercised with the 1.4.0 plugin at source revisions
`5c73a54fccfe9b8341fba46d7146448f18c64f1e` (earlier reconstructed BF16) and
`f37547cd9047b50bd7c1f5fa19fbe303cffedec3` (official BF16).
The results are recorded separately below.

## Tested environment

- Linux aarch64, Python 3.12; image `quay.io/ascend/vllm-omni:v0.30.0-a3`.
- Two visible NPU devices, approximately 61 GiB usable memory each.
- PyTorch `2.10.0+cpu`, torch_npu `2.10.0.post4`.
- vLLM commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- vLLM-Omni commit `a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`.
- vLLM-Ascend commit `16074608c71ef5adce4e2a4e475848b8fc037297`.
- duplexjev-vllm `0.3.0`.

The collected freeze records Sophon revision
`85502291d4c68f09aa5e10168400d34064c5a821`; local experimental changes were also
present. This is a historical environment record, not a clean-checkout
reproducibility claim. Pin image digests and record host driver/CANN versions when
reproducing; an image tag alone does not define the complete runtime.

## Official BF16 validation (2026-11-20)

The official BF16 checkpoint from ModelScope loaded successfully after the
maintainers restored the missing `thinker/audio/` assets. No audio assets were
borrowed from the older checkpoint. Validation used source `f37547c`, Flash plugin
1.4.0, duplexjev-vllm 0.3.0, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post4,
vLLM 0.30.0 and vLLM-Omni 0.30.0, with the HTTP TP2 configuration in this directory.

Five consecutive smoke requests returned text and audio. The tester reported
consistent text/audio, no noise and no abnormal pauses in those five samples.
This is a limited subjective check, not formal quality evaluation. The first
smoke request took 10.46 s to first audio; subsequent requests took 0.79-0.89 s.
These smoke timings are excluded from the benchmark results below.

| In-flight cap | Successful requests | First audio receipt P50 / P90 (s) | Stall P50 / P90 (s) | Audio seconds / wall second |
| --- | --- | --- | --- | --- |
| 1 | 40/40 | 0.806 / 0.854 | 0.949 / 1.235 | 1.050 |
| 4 | 40/40 | 1.456 / 1.673 | 3.524 / 5.272 | 2.440 |
| 8 | 40/40 | 1.727 / 2.338 | 4.331 / 7.052 | 3.796 |
| 4, extended run | 200/200 | 1.482 / 1.636 | 2.996 / 4.789 | 2.361 |

Each run used five excluded warmups, `hello_zh.wav` (6.28 seconds), the repository
Chinese system prompt, voice `chelsie`, `max_tokens=300` and zero initial playback
buffer. All 320 measured requests succeeded. The extended run lasted 436.97 s
(7.28 minutes). The supplied server log had no ERROR, Traceback, out-of-memory or
EngineDeadError matches; warnings were present. All thirteen benchmark tests passed.

First-audio receipt is timestamped before WAV decoding. Stall is simulated
playback underrun with zero initial buffer, not measured browser playback.
Concurrency 4 and 8 produced substantial underrun: successful completion does
not establish smooth real-time serving at those loads. The older first-audio
metric below includes decoding, and the runs are not a controlled comparison.
The official two-H100 observation is not a controlled hardware comparison either.

The benchmark used commit `f80c13773276c8ffaa46257596c29b129180c295` with a clean
worktree. Supplemental audio-file hashes match the prefixes supplied by the
maintainers. Full local hashes, benchmark script hash and measurements are in the
[validation record](ascend_910c_validation.json). These provenance records were
collected after the run; unchanged files since measurement are not independently
established. An immutable model revision and full-checkpoint checksums were not captured.

## Checkpoint prerequisite

Use the official unquantized
[Sophon-Realtime-32B-Flash-BF16](https://www.modelscope.cn/models/AdventistsAI/Sophon-Realtime-32B-Flash-BF16)
checkpoint. It contains the original BF16 thinker; reconstruction is not required.
Download the full model into a separate directory:

```bash
modelscope download --model AdventistsAI/Sophon-Realtime-32B-Flash-BF16 \
  --local_dir /path/to/Sophon-Realtime-32B-Flash-BF16-Official
```

Required layout includes `config.json`, `thinker/`, `talker/` and `voices/`.
In particular, `thinker/audio/` must contain `config.json`,
`preprocessor_config.json` and `model.safetensors`. An early BF16 upload omitted
that directory; if using that download, fetch the restored official assets:

```bash
modelscope download --model AdventistsAI/Sophon-Realtime-32B-Flash-BF16 \
  --include 'thinker/audio/*' \
  --local_dir /path/to/Sophon-Realtime-32B-Flash-BF16-Official
```

Earlier experiments reconstructed approximate BF16 weights from FP8 on CPU.
Those results below are historical, not results for the official BF16 checkpoint.
The tested runtime rejected the dynamic FP8 activation path. Setting
`dtype: bfloat16` does not convert an FP8 checkpoint; use the official BF16 release.

## Container checks

Run inside an existing, correctly provisioned Ascend container. Follow the
[vLLM-Omni NPU installation guide](https://docs.vllm.ai/projects/vllm-omni/en/stable/getting_started/installation/npu/)
for device and driver mounts. Flash stages communicate through shared memory;
ensure the container has sufficient shared memory.

```bash
npu-smi info
python - <<'PY'
import torch
import torch_npu
assert torch.npu.is_available()
assert torch.npu.device_count() >= 2
x = torch.randn(256, 256, device="npu", dtype=torch.bfloat16)
y = x @ x
torch.npu.synchronize()
print(y.shape)
PY
```

Logical devices `0,1` must refer to the intended devices. Do not infer the number
of visible devices solely from physical card count.

## Install the plugin without replacing the NPU stack

From `flash/`, using the container's existing Python:

```bash
python -m pip install --no-deps duplexjev-vllm==0.3.0
python -m pip install --no-deps --no-build-isolation -e ./plugin
python -m pip check
python - <<'PY'
import torch_npu
from sophon_realtime import register
register()
print("registration OK")
PY
```

Inspect missing dependencies individually. Do not run the CUDA-oriented
`install.sh` or replace torch/vLLM packages to resolve an unrelated dependency.
The non-isolated build requires compatible setuptools already in the container.
Import success does not validate NPU forward execution or the new Talker scheduler
patches; those require actual requests.

## Start HTTP inference

The configuration uses thinker TP2 on devices `0,1`; Talker and Code2Wav share
device `1`. Sequence limits are `8/8/4`; memory fractions are `0.70/0.12/0.08`.
These are experimental budgets, not an OOM guarantee.

```bash
cd /path/to/sophon-realtime/flash
export SOPHON_MODEL=/path/to/Sophon-Realtime-32B-Flash-BF16-Official
export SOPHON_WARMUP=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

set -o pipefail
mkdir -p logs
vllm serve "$SOPHON_MODEL" \
  --omni --host 0.0.0.0 --port 8000 --trust-remote-code \
  --init-timeout 3000 --stage-init-timeout 3000 \
  --deploy-config ../community/ascend/sophon_realtime_flash_910c_bf16_tp2.yaml \
  2>&1 | tee logs/ascend-http.log
```

From another terminal, also in `flash/`:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
python client_example.py --url http://127.0.0.1:8000 \
  --wav examples/hello_zh.wav --lang zh --voice chelsie --out reply.wav
```

Warm up before recording latency. Listen to the output, check nonempty audio, and
test long replies as well as short ones. A healthy HTTP endpoint alone is not a
successful model execution check.

Check the output structure separately (this does not replace listening):

```bash
python - <<'PY'
import numpy as np
import soundfile as sf
x, sr = sf.read("reply.wav")
assert sr == 24000
assert len(x) > 2400
assert np.isfinite(x).all()
assert np.abs(x).max() > 1e-5
print("Audio structure check passed; listen to verify quality.")
PY
```

## Earlier reconstructed-BF16 validation (2026-11-17)

The environment freeze identifies the Flash plugin at
`5c73a54fccfe9b8341fba46d7146448f18c64f1e` (1.4.0). Engine revisions match the
ones recorded above. Additional environment files were collected on
2026-11-17 at 04:10:09 UTC:

- Reported container image: `quay.io/ascend/vllm-omni:v0.30.0-a3`.
- Driver package: `26.1.1`; Ascend HAL: `7.35.23`.
- Driver internal version: `V100R001C10SPC009B220`.
- CANN OPP and compiler: `9.1.0`, both with package timestamp
  `20260730_231653901`.
- Toolkit path resolves to `/usr/local/Ascend/cann-9.1.0`.

These files describe the environment at collection time. They do not independently
prove that no component changed since the benchmark. The image tag was confirmed
by the tester; its immutable repository digest remains unrecorded.

Five consecutive audio-input requests returned 24 kHz WAV output, with durations
of 3.44–8.32 seconds. A separate text-input long-reply probe returned 631 text
characters and 122.08 seconds of 24 kHz audio. The server logged three Talker
segment restarts for that request. This exercises the new segmentation path;
listening-based quality, continuity and semantic completeness are not established
by these logs alone. The tester completed manual listening and reported no obvious noise, repetition,
truncation, or disruptive segment transitions. This is a limited subjective check
of the supplied samples, not a formal speech-quality evaluation.

| In-flight requests | Requests | First audio P50 / P90 (s) | Stall P50 / P90 (s) | Measured run time (s) |
| --- | --- | --- | --- | --- |
| 1 | 40/40 | 0.619 / 0.659 | 0.574 / 0.814 | 157.65 |
| 4 | 40/40 | 0.905 / 0.993 | 0.621 / 1.027 | 45.23 |
| 8 | 40/40 | 1.039 / 1.389 | 0.916 / 1.477 | 27.85 |
| 4, extended run | 200/200 | 0.916 / 1.014 | 0.744 / 1.093 | 300.69 |

These 320 measured HTTP requests succeeded according to the client (audio present
and stream terminated normally). This does not certify audio quality or answer
correctness. The extended run lasted approximately five minutes, not ten minutes.
The supplied server log contained no `ERROR`, `Traceback`, `out of memory`, or
`EngineDeadError` matches; it did contain warnings, so this is not a warning-free
runtime claim. See the [compact measurement record](ascend_910c_validation.json).

The external benchmark used `hello_zh.wav`, the repository Chinese system prompt,
voice `chelsie`, `max_tokens=300`, temperature 0.7 and top_p 0.9. Each run used
five unmeasured warmups. Concurrency was a client-side in-flight request cap; it
was not a duplex session count. Benchmark tooling is outside this PR, so the
record is evidence of the reported run rather than a complete reproducibility
bundle. Metric definitions are given below with the historical results.

## Historical HTTP results

One test per point, 40 successful requests per point, repeated `hello_zh.wav`,
warm runtime, BF16 reconstructed thinker, TP2 and eager execution. Latency starts
at the HTTP request and ends after the first WAV chunk is received and decoded.
It does not include real-time input capture, VAD, or duplex turn-taking.

| In-flight requests | First audio P50 / P90 (s) | Stall P50 / P90 (s) |
| --- | --- | --- |
| 1 | 0.603 / 0.635 | 0.487 / 0.829 |
| 4 | 0.870 / 1.052 | 0.534 / 0.901 |
| 8 | 1.007 / 1.275 | 0.807 / 1.377 |

At 8 in-flight requests, changing sequence limits from `4/4/2` to `8/8/4`
reduced first-audio P90 from 3.647 s to 1.275 s. This is configuration-specific,
not a hardware speedup claim. Stall is simulated playback underrun with immediate
playback at the first chunk and no initial buffer. It is not established to match
the upstream published stall metric. Response lengths differed between runs;
requests/second alone is not a controlled throughput comparison.

## Limitations and next validation

- Revalidate after engine/plugin changes. Multi-input runs, repeated load trials,
  longer soak tests and formal speech-quality evaluation remain pending.
- Keep `enforce_eager: true` initially. Graph execution and INT8 conversion are
  separate validation tasks.
- Preserve the existing GPU configuration. This configuration is for the tested
  32B layout, not the 4B or 4-bit model variants.
- Full-duplex cache isolation, session cleanup and Live benchmarking are separate
  work. Do not claim Live concurrency from HTTP measurements.
- If loading fails or OOM occurs, stop and inspect the first error; do not raise
  all stage memory fractions or discard weights to bypass validation.
