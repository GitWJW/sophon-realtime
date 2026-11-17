# Restricted FP8-to-BF16 reconstruction

This CPU tool reconstructs approximate BF16 thinker weights for the Flash model
layout. It does not recover the original pre-quantization BF16 model or guarantee
speech/text quality. It is not a general FP8, INT8, block-FP8, MXFP8 or 4-bit
converter. It accepts compressed-tensors `float-quantized` static symmetric
per-output-channel FP8 E4M3 matrices with `[out_features, 1]` scales. Untouched
thinker tensors must be BF16. Unexpected formats are rejected.

Dependencies: an existing PyTorch installation with CPU FP8 conversion support
and `safetensors`. Do not reinstall the Ascend PyTorch stack for this script.

Run from `flash/`:

```bash
python convert_fp8_to_bf16.py --src /models/Flash-FP8 --dst /models/Flash-BF16 --check-only
python convert_fp8_to_bf16.py --src /models/Flash-FP8 --dst /models/Flash-BF16
```

Source and destination must be separate non-nested paths, and the destination
must not exist. Input must include root `config.json`, `thinker/` and `talker/`.
Do not run concurrently with another writer to either directory. Source weights
are never edited. All other model files, including talker and voices, are copied;
this requires more disk than a symlink-based layout but keeps output independent.

`--check-only` validates config, tensor inventory, index consistency when present,
scale shapes and values, and reports expected thinker tensor bytes. It does not
scan all weight values, allocate full reconstructed tensors or perform model
quality validation. Conversion checks free disk against tensor bytes, copied
files and 1 GiB overhead; the estimate is conservative guidance, not a reservation.
RAM must accommodate one reconstructed output shard plus its inputs and temporary
row blocks. Conversion runs on CPU even in an NPU container.

Each supported weight is reconstructed as
`BF16(float32(weight_fp8) * float32(weight_scale))` with row blocking. Scales are
removed only after corresponding weights are reconstructed, and the thinker
config and weight index are regenerated. Non-finite scales or outputs fail.
The output includes `BF16_CONVERSION.json` with source config hash, tensor counts
and expected weight size. This is provenance, not a full source-weight checksum.

Files are built in a sibling `*.incomplete-*` directory. The final destination is
published only after conversion completes. On failure the incomplete directory is
retained for diagnosis; do not serve it. The tool does not resume or automatically
delete incomplete output. Check the absolute path carefully before any manual
cleanup. Use the final directory only after independently checking its tensors,
index and inference outputs.

From the repository root, run the small synthetic tests without model weights:

```bash
python -m unittest discover -s flash/tests -p test_convert_fp8_to_bf16.py -v
```

Tests cover cross-shard scales, exact reconstruction against the quantized
reference, index generation, unchanged source, check-only/no-overwrite behavior,
unsupported schemes, invalid scales, nested paths and non-finite outputs.
The four test methods passed on the validation server. The local authoring
Python environment does not have torch, so no local execution pass is claimed.

## Full-checkpoint validation

The supplied validation bundle records a completed conversion of
Sophon-Realtime-32B-Flash:

- Source thinker config SHA256:
  `b48dc9247bf75aa54ba2af1109ae9e2e5414db54ce0331dba8169ccd6f4405ba`.
- 448 FP8 matrices reconstructed.
- 1,359 BF16 output tensors, totaling 67,119,747,296 tensor bytes (62.51 GiB).
- Output tensor inventory matches the regenerated index.
- All 1,359 output tensors exactly match the corresponding tensors in the
  previously deployed BF16 reconstruction, using `torch.equal`.
- Five consecutive HTTP audio-input requests returned text and audio.
- The tester confirmed that the inference and manual-listening checks passed.

The five client-reported first-audio times were 10.66, 0.91, 0.63, 0.62 and
0.63 seconds. These are a functional smoke test, including an initial slow
request, not a controlled performance benchmark. No `ERROR`, `Traceback`,
`out of memory` or `EngineDeadError` matches were found in the supplied server
log; this does not mean that the runtime had no warnings.

See the [compact validation record](convert_fp8_to_bf16_validation.json).
Exact agreement with the older BF16 reconstruction validates consistency between
the two conversions, not agreement with unavailable pre-quantization weights.
The bundle does not pin an immutable source model revision or a complete
container/runtime environment; the config hash alone is not a weight checksum.
Formal quality equivalence and arbitrary checkpoint-format support are not claimed.

## Additional input guards

Explicit nonempty transform metadata is rejected, including `transform_config`
and input/output/weight transform declarations in the thinker configuration.
The utility does not attempt to undo Hadamard or other weight/activation transforms.
This guard is conservative; it is not a claim to detect every future format.

During conversion, all thinker output tensors (including preserved BF16 tensors)
are checked for NaN/Inf in blocks of at most 1,048,576 elements. A failure retains
an incomplete directory and does not publish a completed output. This check does
not inspect copied Talker/voice assets. `--check-only` still validates metadata
and scales without scanning every source weight value, so a successful preflight
is not a complete finite-weight validation.

All seven updated synthetic test methods passed on the server. The supported
full checkpoint also completed conversion with these guards enabled: 448 FP8
matrices, 1,359 BF16 tensors and 67,119,747,296 tensor bytes. Independent checks
confirmed that all 1,359 thinker output tensors were finite and that the index
matched the output. The source config hash matched the earlier conversion.

The earlier tensor-equality comparison and five-request inference/listening checks
predate these guards; they were not rerun on the new guarded output in the supplied
bundle. The reconstruction formula is unchanged. Expected conversion-failure
messages in the synthetic test log are negative-test results, not failures of the
full-checkpoint run. Local functional execution is still not claimed because the
local authoring environment lacks torch.
