# HTTP streaming audio benchmark

Run from `flash/` with `aiohttp`, `numpy`, and `soundfile` installed. This measures
`/v1/chat/completions`, not Live, VAD, duplex turn-taking or barge-in.

```bash
python benchmark_http.py --wav examples/hello_zh.wav --concurrency 1 --requests 30 --warmup 5 --out bench/single
python benchmark_http.py --wav examples/hello_zh.wav --concurrency 4 --requests 80 --out bench/c4
python benchmark_http.py --wav examples/hello_zh.wav --requests 40 --arrival-window 10 --out bench/arrival40
python benchmark_http.py --manifest inputs.json --concurrency 4 --requests 80 --out bench/multi
```

Output directories must not exist. Concurrency and arrival-window options are
mutually exclusive. The latter schedules evenly spaced arrivals without an
in-flight cap. Peak in-flight count includes server queueing.

Manifest paths are relative to the manifest file:

```json
[
  {"id": "short-zh", "wav": "audio/short-zh.wav", "language": "zh"},
  {"id": "short-en", "wav": "audio/short-en.wav", "language": "en"}
]
```

IDs must be unique. Each language needs a system prompt in `--system-prompt`,
which defaults to the repository prompt file. Selection defaults to round-robin;
`--selection random --seed 42` is reproducible. WAV validation, hashing and base64
encoding happen before timing. Do not publish private recordings or credentials.

`requests.jsonl` flushes each completed request. `summary.json` uses successful
measured requests only. Warmups are excluded and abort the run on failure.
Success requires valid audio and a normal `[DONE]` terminator. HTTP errors,
SSE errors, missing audio, invalid WAV/base64, timeout and truncation are failures.
Partial latency is retained for diagnostics, but excluded from success percentiles.

Metrics (seconds unless stated):

- `first_audio_received_s`: application-level dispatch of the first valid audio
  SSE event, timestamped before WAV decoding; not packet arrival time.
- `first_audio_decoded_s`: first audio decode completion. The old script's
  `first_audio_s` represented this kind of measurement; names deliberately differ.
- `first_audible_s`: arrival of the first chunk with peak above
  `--audible-threshold` (default .02 normalized amplitude); chunk-level proxy,
  not exact sample onset.
- `stall_s`: simulated playback underrun from chunk arrivals and durations,
  starting at first arrival plus `--initial-buffer` (default zero).
- `e2e_s`: request start to normal stream completion processing.
- `audio_duration_s`: sum of chunk durations, assuming sequential nonoverlapping
  WAV payloads as in the repository client.
- `requests_per_s`: successful requests / measured wall time.
- `audio_seconds_per_s`: successful output audio seconds / measured wall time.
- `send_lag_s`: dispatch lateness against the open-loop schedule; null otherwise.

Silence inside actual audio counts as content. The simulated player does not
model browser buffering or sound hardware. Client decoding, JSON, network and
event-loop load affect results. A bigger initial buffer trades later playback for
fewer stalls; it does not accelerate inference. The stall algorithm is not claimed
to match the upstream published benchmark. All-failed runs have null percentiles.

Record server revisions/configuration, hardware, precision, answer durations and
cache state. Reused inputs may benefit from caching. Compare audio throughput,
not just request throughput, when answer lengths differ. This tool implements
the repository's base64 WAV SSE shape, not arbitrary provider audio protocols.
Result schema version is 1; old collectors must explicitly adapt.

Local tests (run from repository root, no NPU/model required):

```bash
python -m unittest discover -s flash/tests -p test_benchmark_http.py -v
```

Tests cover fragmented/multiline SSE, completion, HTTP/SSE errors, malformed
audio, missing audio, truncation, timeout, manifests, stall arithmetic and
success-only summary aggregation. Load-generation integration tests additionally check bounded in-flight counts,
round-robin assignment, staggered schedule metadata, and all-failed aggregation.

## Recorded server smoke tests

An Ascend-backed Flash HTTP endpoint completed all 45 measured requests:

| Workload | Success | Peak in flight | First audio receipt P50 / P90 (s) | Stall P50 / P90 (s) |
| --- | --- | --- | --- | --- |
| Single input, concurrency 1 | 5/5 | 1 | 0.623 / 0.651 | 0.744 / 0.998 |
| Three inputs, concurrency 4 | 20/20 | 4 | 0.996 / 1.137 | 0.710 / 1.071 |
| Three inputs, 20 arrivals over 10 s | 20/20 | 12 | 1.041 / 1.192 | 1.033 / 1.946 |

Both multi-input runs assigned inputs 7/7/6 times. Maximum observed scheduled
arrival lag was about 1.8 ms. The compact [validation record](benchmark_http_validation.json)
was checked against per-request records for success counts and audio throughput.
These validate the client against a real endpoint, not a controlled hardware
comparison or a serving-capacity guarantee. The supplied client results do not
capture the complete server revision/configuration; reproduce with recorded server
metadata before comparing hardware. Raw recordings and full server logs are not
included in this PR.
