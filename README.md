# IndexTTS-2 — Replicate Cog wrapper

This repository packages the IndexTeam IndexTTS-2 text-to-speech system for use with Replicate Cog. It exposes a single `Predictor` that performs zero-shot speaker cloning with optional emotion control.

- Upstream model: [IndexTeam/IndexTTS-2](https://github.com/IndexTeam/IndexTTS-2)
- Serving runtime: [Replicate Cog](https://github.com/replicate/cog)

## Quickstart

### 1) Prerequisites
- NVIDIA GPU recommended (FP16 enabled automatically)
- CUDA-enabled PyTorch (already pinned in `requirements.txt`)
- System packages: `ffmpeg`, `libsndfile1` (installed by Cog via `cog.yaml`)
- Python 3.10 (managed by Cog)

### 2) Create a local env (optional for local dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Download checkpoints
Place the official IndexTTS-2 release weights under `checkpoints/` alongside a `checkpoints/config.yaml` expected by the upstream code. Do not commit large binaries.

- See the upstream project for assets and structure: [IndexTeam/IndexTTS-2](https://github.com/IndexTeam/IndexTTS-2)
- Example layout (illustrative):
```
checkpoints/
  config.yaml
  ... model files and subfolders ...
```

### 4) Run the predictor with Cog
Install Cog and run an inference:
```bash
# Once per machine
pip install cog

# Predict (returns a WAV file path)
cog predict \
  -i text="Hello from IndexTTS-2." \
  -i speaker_audio=@/path/to/speaker.wav
```

Inputs accept absolute paths or `@/path` file uploads. The output is a path to a synthesized `output.wav`.

## Inputs
The `Predictor.predict` interface exposes the following inputs. Types and defaults mirror the implementation in `predict.py`.

| Name | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize. |
| `speaker_audio` | file (WAV) | required | Reference audio for the target speaker. Recommended 16–48 kHz WAV. |
| `emotion_audio` | file (WAV) | `null` | Optional emotion reference; falls back to `speaker_audio` when omitted. |
| `emotion_scale` | float | `1.0` | Blend ratio when both speaker and emotion prompts are used. Range [0, 1]. |
| `emotion_vector` | string | `null` | Comma-separated or JSON list of 8 emotion weights to bypass the classifier. See “Emotion control” below. |
| `emotion_text` | string | `null` | When provided, auto-detect emotions from text using Qwen. |
| `randomize_emotion` | bool | `false` | Pick emotion embeddings randomly instead of nearest-neighbour when vectors are provided. |
| `interval_silence_ms` | int | `200` | Silence inserted between long segments in milliseconds. Range [0, 2000]. |
| `max_text_tokens_per_segment` | int | `120` | Maximum BPE tokens per autoregressive segment. Range [32, 300]. |
| `top_p` | float | `0.8` | Top-p (nucleus) sampling for GPT stage. Range [0, 1]. |
| `top_k` | int | `30` | Top-k sampling for GPT stage. Range [1, 200]. |
| `temperature` | float | `0.8` | Sampling temperature for GPT stage. Range [0, 2]. |
| `length_penalty` | float | `0.0` | Beam search length penalty. Range [0, 5]. |
| `num_beams` | int | `3` | Beam width for GPT stage. Range [1, 8]. |
| `repetition_penalty` | float | `10.0` | Penalty for repeated tokens. Range [1, 30]. |
| `max_mel_tokens` | int | `1500` | Maximum mel tokens to generate per segment. Range [256, 4096]. |

### Example invocations

- Basic cloning:
```bash
cog predict \
  -i text="This is IndexTTS-2 speaking." \
  -i speaker_audio=@/data/voice_samples/alice.wav
```

- With a separate emotion prompt and blend:
```bash
cog predict \
  -i text="I am very excited to be here!" \
  -i speaker_audio=@/data/voice_samples/alice.wav \
  -i emotion_audio=@/data/voice_samples/excited.wav \
  -i emotion_scale=0.7
```

- With a manual 8-dim emotion vector (comma-separated or JSON):
```bash
# Order (English): [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
cog predict \
  -i text="The results were surprising." \
  -i speaker_audio=@/data/voice_samples/alice.wav \
  -i emotion_vector="0.2,0.0,0.1,0.0,0.0,0.0,0.8,0.1"
```

- With `emotion_text` using the built-in classifier:
```bash
cog predict \
  -i text="今天的天气真让人兴奋！" \
  -i speaker_audio=@/data/voice_samples/mandarin.wav \
  -i emotion_text="兴奋 开心 激动"
```

## Emotion control notes
This wrapper supports three ways to control emotion:
- Provide `emotion_audio` and optional `emotion_scale` to blend with the speaker identity
- Provide `emotion_vector` to bypass classification; 8 weights are expected in the order:
  - Chinese keys: [高兴, 愤怒, 悲伤, 恐惧, 反感, 低落, 惊讶, 自然]
  - English mapping: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
- Provide `emotion_text` to auto-detect emotion via Qwen; if unavailable or failing, the wrapper falls back to a neutral vector (`{"calm": 1.0}`)

On CPU-only hosts, a lightweight compatibility path is enabled for Qwen; if any error occurs, neutral emotion is used to avoid failures.

## Development
- Run locally with your Python after installing `requirements.txt`
- The Cog runtime uses:
  - Python 3.10
  - GPU: enabled (`build.gpu: true`)
  - System packages: `ffmpeg`, `libsndfile1`

Useful commands:
```bash
# Lint or tests (add tests under tests/)
python -m pytest -q

# End-to-end via Cog
cog predict -i text="Hello" -i speaker_audio=@/path/to/speaker.wav
```

## Deployment to Replicate
If you intend to publish:
```bash
# Authenticate (once)
cog login

# Push (replace <user>/<repo>)
cog push r8.im/<user>/indextts-2
```

## Troubleshooting
- Ensure checkpoints are present under `checkpoints/` with a valid `config.yaml`.
- Use 16–48 kHz WAV inputs to avoid resampling artifacts; `ffmpeg` is available for conversions.
- CPU runs are possible but slow; GPU is strongly recommended.
- If emotion detection fails or required models are missing, the predictor logs a message and uses a neutral emotion vector.

## License and credits
- This wrapper mirrors and serves the upstream IndexTTS-2 project. Check the original repository for model licenses and usage terms.
- Please do not commit or redistribute large model binaries in this repo.
