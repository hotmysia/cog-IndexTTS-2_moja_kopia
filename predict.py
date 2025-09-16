import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import torch
from cog import BasePredictor, Input, Path as CogPath

from indextts import infer_v2


def _maybe_patch_qwen_for_cpu() -> None:
    """Ensure the Qwen emotion model can run on CPU-only hosts."""
    if torch.cuda.is_available():
        return

    from modelscope import AutoModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer

    class CPUQwenEmotion:
        def __init__(self, model_dir: str):
            self.model_dir = model_dir
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
            )
            self.prompt = "文本情感分类"
            self.cn_key_to_en = {
                "高兴": "happy",
                "愤怒": "angry",
                "悲伤": "sad",
                "恐惧": "afraid",
                "反感": "disgusted",
                "低落": "melancholic",
                "惊讶": "surprised",
                "自然": "calm",
            }
            self.desired_vector_order = [
                "高兴",
                "愤怒",
                "悲伤",
                "恐惧",
                "反感",
                "低落",
                "惊讶",
                "自然",
            ]
            self.melancholic_words = {
                "低落",
                "melancholy",
                "melancholic",
                "depression",
                "depressed",
                "gloomy",
            }
            self.max_score = 1.2
            self.min_score = 0.0

        def clamp_score(self, value: float) -> float:
            return max(self.min_score, min(self.max_score, value))

        def convert(self, content):
            emotion_dict = {
                self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
                for cn_key in self.desired_vector_order
            }
            if all(val <= 0.0 for val in emotion_dict.values()):
                emotion_dict["calm"] = 1.0
            return emotion_dict

        def inference(self, text_input: str):
            import re

            messages = [
                {"role": "system", "content": f"{self.prompt}"},
                {"role": "user", "content": f"{text_input}"},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
            try:
                parsed = json.loads(content)
            except json.decoder.JSONDecodeError:
                parsed = {
                    match.group(1): float(match.group(2))
                    for match in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
                }
            lower = text_input.lower()
            if any(word in lower for word in self.melancholic_words):
                parsed["悲伤"], parsed["低落"] = parsed.get("低落", 0.0), parsed.get("悲伤", 0.0)
            return self.convert(parsed)

    infer_v2.QwenEmotion = CPUQwenEmotion


def _wrap_qwen_with_fallback() -> None:
    """Wrap QwenEmotion so missing architectures fall back gracefully."""

    original_cls = getattr(infer_v2, "QwenEmotion", None)
    if original_cls is None:
        return

    # Avoid wrapping more than once.
    if getattr(original_cls, "_is_safe_wrapper", False):
        return

    class SafeQwenEmotion:
        _is_safe_wrapper = True

        def __init__(self, model_dir: str):
            self._inner = None
            self._inner_exc: Optional[Exception] = None
            try:
                self._inner = original_cls(model_dir)
            except Exception as exc:  # pragma: no cover - relies on external packages
                self._inner_exc = exc
                print(
                    ">> QwenEmotion unavailable ({}). Falling back to neutral emotion.".format(
                        exc
                    )
                )

        def inference(self, text_input: str):
            if self._inner is None:
                return {"calm": 1.0}
            try:
                return self._inner.inference(text_input)
            except Exception as exc:  # pragma: no cover
                print(
                    ">> QwenEmotion inference failed ({}). Returning neutral emotion.".format(
                        exc
                    )
                )
                return {"calm": 1.0}

    infer_v2.QwenEmotion = SafeQwenEmotion


def _parse_vector(raw: Optional[str]) -> Optional[Iterable[float]]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        if raw.startswith("["):
            values = json.loads(raw)
        else:
            values = [float(tok) for tok in raw.split(",") if tok.strip()]
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Could not parse emotion_vector; use JSON or comma separated numbers") from exc
    if not values:
        return None
    return [float(val) for val in values]


class Predictor(BasePredictor):
    def setup(self) -> None:
        _maybe_patch_qwen_for_cpu()
        _wrap_qwen_with_fallback()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_fp16 = device.startswith("cuda")
        self.tts = infer_v2.IndexTTS2(
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            use_fp16=use_fp16,
            device=device,
            use_cuda_kernel=use_fp16,
        )

    def predict(
        self,
        text: str = Input(description="Text to synthesize."),
        speaker_audio: CogPath = Input(description="Reference audio for the target speaker (16k-48kHz WAV)."),
        emotion_audio: Optional[CogPath] = Input(
            description="Optional emotion reference audio. Defaults to speaker audio when omitted.",
            default=None,
        ),
        emotion_scale: float = Input(
            description="Blend ratio for the emotion reference when both speaker and emotion prompts are used.",
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        emotion_vector: Optional[str] = Input(
            description="Optional comma separated or JSON list of 8 emotion weights to bypass the classifier.",
            default=None,
        ),
        emotion_text: Optional[str] = Input(
            description="Text prompt used to auto-detect emotions via Qwen when provided.",
            default=None,
        ),
        randomize_emotion: bool = Input(
            description="Pick emotion embeddings randomly instead of nearest-neighbour selection when vectors are provided.",
            default=False,
        ),
        interval_silence_ms: int = Input(
            description="Silence inserted between long segments in milliseconds.",
            default=200,
            ge=0,
            le=2000,
        ),
        max_text_tokens_per_segment: int = Input(
            description="Maximum BPE tokens per autoregressive segment.",
            default=120,
            ge=32,
            le=300,
        ),
        top_p: float = Input(description="Top-p nucleus sampling for GPT stage.", default=0.8, ge=0.0, le=1.0),
        top_k: int = Input(description="Top-k sampling for GPT stage.", default=30, ge=1, le=200),
        temperature: float = Input(description="Sampling temperature for GPT stage.", default=0.8, ge=0.0, le=2.0),
        length_penalty: float = Input(description="Beam search length penalty.", default=0.0, ge=0.0, le=5.0),
        num_beams: int = Input(description="Beam width for GPT stage.", default=3, ge=1, le=8),
        repetition_penalty: float = Input(description="Penalty for repeated tokens.", default=10.0, ge=1.0, le=30.0),
        max_mel_tokens: int = Input(description="Maximum mel tokens to generate per segment.", default=1500, ge=256, le=4096),
    ) -> CogPath:
        emotion_values = _parse_vector(emotion_vector)
        tmp_dir = Path(tempfile.mkdtemp())
        output_path = tmp_dir / "output.wav"
        emo_prompt = str(emotion_audio) if emotion_audio is not None else None
        result_path = self.tts.infer(
            spk_audio_prompt=str(speaker_audio),
            text=text,
            output_path=str(output_path),
            emo_audio_prompt=emo_prompt,
            emo_alpha=emotion_scale,
            emo_vector=emotion_values,
            use_emo_text=emotion_text is not None and emotion_text.strip() != "",
            emo_text=emotion_text,
            use_random=randomize_emotion,
            interval_silence=interval_silence_ms,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens,
        )
        return CogPath(result_path)
