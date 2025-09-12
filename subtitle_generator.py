# subtitle_generator.py
from __future__ import annotations
import sys

# Ưu tiên faster-whisper; fallback openai-whisper nếu thiếu
_BACKEND = None
try:
    from faster_whisper import WhisperModel
    _BACKEND = "faster"
except Exception:
    WhisperModel = None
    _BACKEND = None

class SubtitleGenerator:
    """
    Load Whisper và tạo segments cho 1 video.
    - Ưu tiên GPU (CUDA) nếu khả dụng; fallback CPU khi lỗi/thiếu VRAM.
    - Hỗ trợ auto-detect language hoặc chỉ định cụ thể
    - Trả về: list[dict] [{start: float, end: float, text: str}, ...]
    """
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
                    "Thiếu 'faster-whisper'. Cài nhanh: pip install -U faster-whisper\n"
                    "Hoặc dùng openai-whisper: pip install -U openai-whisper\n"
                    f"Chi tiết: {e}"
                )
            self.backend = "openai"
            self.whisper = whisper
            self.torch = torch
            self.model, self.device = self._load_openai()

    # ---------- faster-whisper (CTranslate2) ----------
    def _try_load_faster(self, device: str, compute_type: str):
        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def _load_faster(self):
        # GPU FP16 -> GPU FP32 -> CPU INT8 -> CPU FP32
        if self.prefer_gpu:
            try:
                m = self._try_load_faster("cuda", "float32")
                print(f"✓ Loaded faster-whisper '{self.model_size}' on GPU (float32).")
                return m, "cuda", "float32"
            except Exception as e:
                msg = str(e).lower()
                if "float16" in msg or "half" in msg:
                    try:
                        m = self._try_load_faster("cuda", "float32")
                        print(f"✓ Loaded faster-whisper '{self.model_size}' on GPU (float32).")
                        return m, "cuda", "float32"
                    except Exception:
                        pass
                if "out of memory" in msg or "cuda" in msg:
                    print("⚠️ CUDA OOM/error → fallback CPU.")

        try:
            m = self._try_load_faster("cpu", "int8")
            print(f"✓ Loaded faster-whisper '{self.model_size}' on CPU (int8).")
            return m, "cpu", "int8"
        except Exception:
            m = self._try_load_faster("cpu", "float32")
            print(f"✓ Loaded faster-whisper '{self.model_size}' on CPU (float32).")
            return m, "cpu", "float32"

    # ---------- openai-whisper (PyTorch) ----------
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

    # ---------- API ----------
    def generate_segments(
        self,
        video_path: str,
        language: str = None,  # None = auto-detect
        beam_size: int = 1,     # Giảm xuống 1 cho tốc độ nhanh
        vad: bool = True,       # Bật VAD mặc định
        word_timestamps: bool = False  # Tắt word-level timestamps cho nhanh
    ):
        """
        Trả về list[dict]: [{start: float, end: float, text: str}, ...]
        - language: None (auto), 'zh' (Chinese), 'en' (English), etc.
        - VAD: Bật để bỏ qua đoạn im lặng
        - Tối ưu cho tốc độ với beam_size=1
        """
        if self.backend == "faster":
            try:
                # Thông báo progress
                print(f"  → Đang transcribe (language={'auto' if not language else language}, VAD={'on' if vad else 'off'})...")
                
                # Tham số tối ưu cho tốc độ và độ chính xác
                seg_iter, info = self.model.transcribe(
                    video_path,
                    task="transcribe",
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad,             # sẽ là False từ main.py
                    word_timestamps=word_timestamps,
                    condition_on_previous_text=True
                )
                
                # Hiển thị ngôn ngữ được detect
                if not language and hasattr(info, 'language'):
                    print(f"  → Detected language: {info.language} (probability: {info.language_probability:.2f})")
                
                # Thu thập segments với progress
                segments = []
                for i, s in enumerate(seg_iter):
                    segments.append({
                        "start": s.start, 
                        "end": s.end, 
                        "text": (s.text or "").strip()
                    })
                    # Progress indicator
                    if (i + 1) % 10 == 0:
                        sys.stdout.write(f"\r  → Processed {i + 1} segments...")
                        sys.stdout.flush()
                
                if segments:
                    print(f"\r  → Processed {len(segments)} segments total.    ")
                else:
                    print("  → No speech detected in video.")
                    
                return segments
                
            except Exception as e:
                raise RuntimeError(f"Transcription failed (faster-whisper) for '{video_path}': {e}")
        else:
            # openai-whisper
            try:
                print(f"  → Đang transcribe với openai-whisper...")
                result = self.model.transcribe(
                    video_path,
                    task="transcribe",
                    language=language,
                    beam_size=beam_size,
                    fp16=(self.device == "cuda"),
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.4
                )
                
                # Hiển thị ngôn ngữ detected
                if not language and 'language' in result:
                    print(f"  → Detected language: {result['language']}")
                    
                segs = result.get("segments", []) or []
                segments = [
                    {
                        "start": s.get("start", 0.0), 
                        "end": s.get("end", 0.0), 
                        "text": (s.get("text") or "").strip()
                    }
                    for s in segs
                ]
                print(f"  → Processed {len(segments)} segments.")
                return segments
                
            except Exception as e:
                raise RuntimeError(f"Transcription failed (openai-whisper) for '{video_path}': {e}")