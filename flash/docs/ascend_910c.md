# Experimental Flash deployment on Ascend 910C

This recipe documents HTTP streaming inference experiments with the
32B Flash model on one physical 910C card exposing two NPU devices. It does not
establish support for every Flash variant or full-duplex multi-session serving.
The configuration was exercised with the 1.4.0 plugin at PR revision
`5c73a54fccfe9b8341fba46d7146448f18c64f1e`. Historical and current-revision
measurements are reported separately below.

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

## Checkpoint prerequisite

The tested source checkpoint was Sophon-Realtime-32B-Flash with compressed-tensors
FP8 E4M3 per-output-channel weights and dynamic per-token FP8 activations. In the
tested runtime, `aclnnDynamicQuantV2` rejected FP8 output and listed integer output
types. Merely copying an FP8 tensor to NPU did not establish executable FP8 support.

This is not an end-to-end checkpoint conversion guide: the startup steps require
a separately prepared and validated, unquantized BF16 checkpoint with the original layout:

```text
model/
  config.json
  thinker/
  talker/
  voices/
```

The experiment reconstructed BF16 weights on CPU using `weight.float() *
weight_scale.float()` for each symmetric channel-quantized matrix, then cast to
BF16. This reconstructs approximate quantized weights, not the original BF16
model. The resulting thinker contained 1,359 BF16 tensors totaling 62.51 GiB.
Conversion tooling and numerical/quality validation are outside this documentation
PR. Do not remove quantization metadata from an unchanged FP8 checkpoint or cast
FP8 values without their scales. Setting `dtype: bfloat16` is not a conversion.

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
export SOPHON_MODEL=/path/to/Sophon-Realtime-32B-Flash-BF16
export SOPHON_WARMUP=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

set -o pipefail
mkdir -p logs
vllm serve "$SOPHON_MODEL" \
  --omni --host 0.0.0.0 --port 8000 --trust-remote-code \
  --init-timeout 3000 --stage-init-timeout 3000 \
  --deploy-config config/sophon_realtime_flash_910c_bf16_tp2.yaml \
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

## Current-revision validation (2026-11-17)

The environment freeze identifies the Flash plugin at
`5c73a54fccfe9b8341fba46d7146448f18c64f1e` (1.4.0). Engine revisions match the
ones recorded above. Archived version-file paths point to `cann-9.1.0`, but their
contents, the host driver version and image digest were not captured; the exact
CANN patch/build remains unverified.

Five consecutive audio-input requests returned 24 kHz WAV output, with durations
of 3.44–8.32 seconds. A separate text-input long-reply probe returned 631 text
characters and 122.08 seconds of 24 kHz audio. The server logged three Talker
segment restarts for that request. This exercises the new segmentation path;
listening-based quality, continuity and semantic completeness are not established
by these logs alone.

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
  longer soak tests and listening-based quality checks remain to be completed.
- Keep `enforce_eager: true` initially. Graph execution and INT8 conversion are
  separate validation tasks.
- Preserve the existing GPU configuration. This configuration is for the tested
  32B layout, not the 4B or 4-bit model variants.
- Full-duplex cache isolation, session cleanup and Live benchmarking are separate
  work. Do not claim Live concurrency from HTTP measurements.
- If loading fails or OOM occurs, stop and inspect the first error; do not raise
  all stage memory fractions or discard weights to bypass validation.
