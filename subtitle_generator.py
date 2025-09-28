# subtitle_generator.py
from __future__ import annotations
import sys

_BACKEND = None
try:
    from faster_whisper import WhisperModel
    _BACKEND = "faster"
except Exception:
    WhisperModel = None
    _BACKEND = None

class SubtitleGenerator:
    def __init__(self, model_size: str = "small", prefer_gpu: bool = True):
        self.model_size = model_size
        self.prefer_gpu = prefer_gpu

        if _BACKEND == "faster":
            self.backend = "faster"
            self.model, self.device, self.compute_type = self._load_faster()
        else:
            try:
                import whisper  # type: ignore
                import torch    # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Thiếu 'faster-whisper'. pip install -U faster-whisper\n"
                    "Hoặc dùng openai-whisper: pip install -U openai-whisper\n"
                    f"Chi tiết: {e}"
                )
            self.backend = "openai"
            self.whisper = whisper
            self.torch = torch
            self.model, self.device = self._load_openai()

    def _try_load_faster(self, device: str, compute_type: str):
        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def _load_faster(self):
        if self.prefer_gpu:
            try:
                m = self._try_load_faster("cuda", "float32")
                print(f"✓ Loaded faster-whisper '{self.model_size}' on GPU (float32).")
                return m, "cuda", "float32"
            except Exception as e:
                print(f"⚠️ GPU load failed ({e}), fallback CPU.")
        try:
            m = self._try_load_faster("cpu", "int8")
            print(f"✓ Loaded faster-whisper '{self.model_size}' on CPU (int8).")
            return m, "cpu", "int8"
        except Exception:
            m = self._try_load_faster("cpu", "float32")
            print(f"✓ Loaded faster-whisper '{self.model_size}' on CPU (float32).")
            return m, "cpu", "float32"

    def _load_openai(self):
        device = "cuda" if self.prefer_gpu else "cpu"
        try:
            if device == "cuda":
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    device = "cpu"
            m = self.whisper.load_model(self.model_size, device=device)
            print(f"✓ Loaded openai-whisper '{self.model_size}' on {device}.")
            return m, device
        except Exception as e:
            if device == "cuda":
                print(f"⚠️ CUDA load failed ({e}), fallback CPU...")
                m = self.whisper.load_model(self.model_size, device="cpu")
                return m, "cpu"
            raise

    def generate_segments(
        self,
        video_path: str,
        language: str | None = None,   # None = auto-detect
        beam_size: int = 5,
        vad: bool = True,
        word_timestamps: bool = False
    ):
        """
        Trả về tuple: (segments, detected_language)
        segments: list[{start, end, text}]
        detected_language: mã ISO (vd: 'zh', 'en') nếu có
        """
        if self.backend == "faster":
            try:
                print(f"  → Đang transcribe (language={'auto' if not language else language}, VAD={'on' if vad else 'off'})...")
                seg_iter, info = self.model.transcribe(
                    video_path,
                    task="transcribe",
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad,            # Silero VAD tích hợp :contentReference[oaicite:2]{index=2}
                    word_timestamps=word_timestamps,  # word-level timestamps :contentReference[oaicite:3]{index=3}
                    condition_on_previous_text=True
                )
                detected = getattr(info, "language", language)
                if detected:
                    prob = getattr(info, "language_probability", None)
                    if prob is not None:
                        print(f"  → Detected language: {detected} (probability: {prob:.2f})")

                segments = []
                for i, s in enumerate(seg_iter):
                    item = {"start": s.start, "end": s.end, "text": (s.text or "").strip()}
                    if hasattr(s, "words") and s.words:
                        item["words"] = [{"start": w.start, "end": w.end, "text": w.word} for w in s.words]
                    segments.append(item)
                    if (i + 1) % 10 == 0:
                        sys.stdout.write(f"\r  → Processed {i + 1} segments...")
                        sys.stdout.flush()
                if segments:
                    print(f"\r  → Processed {len(segments)} segments total.    ")
                else:
                    print("  → No speech detected in video.")
                return segments, detected
            except Exception as e:
                raise RuntimeError(f"Transcription failed (faster-whisper) for '{video_path}': {e}")
        else:
            try:
                res = self.model.transcribe(
                    video_path, task="transcribe", language=language,
                    beam_size=beam_size, fp16=(self.device == "cuda")
                )
                detected = res.get("language", language)
                segs = res.get("segments", []) or []
                segments = [{"start": s.get("start", 0.0), "end": s.get("end", 0.0), "text": (s.get("text") or '').strip()} for s in segs]
                print(f"  → Processed {len(segments)} segments.")
                return segments, detected
            except Exception as e:
                raise RuntimeError(f"Transcription failed (openai-whisper) for '{video_path}': {e}")
