# subtitle_writer.py
import re
from math import ceil

try:
    import jieba  # dùng để token hóa tiếng Trung (ưu tiên)
except Exception:
    jieba = None

# Dấu câu CJK + bản Latin tương đương
CJK_PUNCTS = set(list("，。、？！；：…—,.?!;:"))

class SubtitleWriter:
    """
    Ghi .srt với:
      - Ngôn ngữ Latin (mặc định): 2 dòng, 42 CPL, CPS~15, 1–6s (giữ logic cũ)
      - CJK mode (zh): 1 dòng, 14 CPL, CPS mục tiêu ~9.5, 1–3s, NGẮT THEO TOKEN/DẤU CÂU.
    Không sử dụng danh sách 'break words' cố định.
    """

    def __init__(
        self,
        max_line_chars: int = 42,
        max_lines_per_caption: int = 2,
        target_cps: float = 15.0,
        min_duration: float = 1.0,
        max_duration: float = 6.0,
        merge_short_gap: float = 1.0,
        merge_max_chars: int = 200,
        cjk_mode: bool = False
    ):
        self.max_line_chars = max_line_chars
        self.max_lines_per_caption = max_lines_per_caption
        self.target_cps = target_cps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_short_gap = merge_short_gap
        self.merge_max_chars = merge_max_chars
        self.cjk_mode = cjk_mode

        self._punct_only = re.compile(r"^[\s\W_]+$")
        self._music_pattern = re.compile(r"^[\s\[\(]*(?:♪|♫|music|música|音楽|音乐)[\s\]\)]*$", re.I)
        self._noise_pattern = re.compile(r"^[\s\[\(]*(?:noise|static|silence|inaudible|unintelligible|不清楚)[\s\]\)]*$", re.I)

    # ---------- utils ----------
    def _format_timestamp(self, seconds: float) -> str:
        total_ms = int(round(max(0.0, seconds) * 1000))
        h = total_ms // 3600000
        rem = total_ms % 3600000
        m = rem // 60000
        rem = rem % 60000
        s = rem // 1000
        ms = rem % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _should_skip_segment(self, text: str) -> bool:
        if not text:
            return True
        if self._punct_only.match(text):
            return True
        if self._music_pattern.match(text):
            return True
        if self._noise_pattern.match(text):
            return True
        if len(text.strip()) < 1:
            return True
        return False

    def _clean_text(self, text: str) -> str:
        if self.cjk_mode:
            return text.strip()
        # Latin: giữ logic cũ
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])(\w)', r'\1 \2', text)
        return text.strip()

    # ---------- CJK tokenization ----------
    def _tokenize_cjk(self, text: str):
        """Trả về list token theo từ. Không cắt giữa một 'từ'."""
        if jieba:
            toks = [t.strip() for t in jieba.cut(text, cut_all=False) if t.strip()]
            return toks if toks else list(text.strip())
        return list(text.strip())

    # ---------- CJK chunking w/o fixed break words ----------
    def _chunk_tokens_cjk(self, tokens):
        """
        Gom token thành các 'chunk' 1 dòng, ưu tiên ngắt ở dấu câu,
        kẹp CPL theo self.max_line_chars * self.max_lines_per_caption (thường 14).
        """
        limit = self.max_line_chars * self.max_lines_per_caption  # =14 khi CJK
        chunks, cur = [], []
        cur_len = 0

        for i, tok in enumerate(tokens):
            tlen = len(tok)
            is_punct = (tok in CJK_PUNCTS)

            # Nếu thêm tok vượt limit -> chốt chunk trước đó (nếu có)
            if cur_len + tlen > limit and cur:
                chunks.append("".join(cur))
                cur, cur_len = [], 0

            # Thêm token hiện tại
            if tlen <= limit:
                # nếu tok bản thân quá dài (hiếm), cắt cứng
                if tlen > limit:
                    # never happen with CJK single tokens, but keep it robust
                    pos = 0
                    while pos < tlen:
                        chunks.append(tok[pos:pos+limit])
                        pos += limit
                    continue
                cur.append(tok)
                cur_len += tlen

            # Ngắt tự nhiên tại dấu câu / hoặc gần hết limit
            next_tok = tokens[i+1] if i+1 < len(tokens) else ""
            near_full = (cur_len >= limit * 0.8)
            if is_punct or (near_full and (next_tok in CJK_PUNCTS or not next_tok)):
                if cur:
                    chunks.append("".join(cur))
                    cur, cur_len = [], 0

        if cur:
            chunks.append("".join(cur))

        # Cắt cứng chunk quá dài (phòng hờ)
        refined = []
        for c in chunks:
            while len(c) > limit:
                refined.append(c[:limit])
                c = c[limit:]
            if c:
                refined.append(c)
        return refined

    # ---------- Latin splitter giữ nguyên ----------
    def _soft_split_chunks(self, text: str):
        limit = self.max_line_chars * self.max_lines_per_caption
        words = text.split()
        chunks, current = [], []

        def count_chars(ws):
            return (sum(len(w) for w in ws) + max(0, len(ws)-1)) if ws else 0

        for w in words:
            if count_chars(current + [w]) <= limit:
                current.append(w)
            else:
                if not current:
                    chunks.append(w[:limit])
                    rest = w[limit:]
                    while rest:
                        chunks.append(rest[:limit])
                        rest = rest[limit:]
                else:
                    chunks.append(" ".join(current))
                    current = [w]
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _layout_two_lines(self, chunk: str):
        words = chunk.split()
        line1, line2 = [], []
        half = ceil(len(chunk) / 2)
        cur_len = 0
        for w in words:
            if cur_len + (len(w) + (1 if line1 else 0)) <= min(self.max_line_chars, half):
                line1.append(w); cur_len = len(" ".join(line1))
            else:
                line2.append(w)
        if not line2 and len(" ".join(line1)) > self.max_line_chars:
            s = " ".join(line1)
            return s[:self.max_line_chars], s[self.max_line_chars:]
        return " ".join(line1), " ".join(line2)

    # ---------- duration allocation ----------
    def _allocate_durations(self, start, end, chunks):
        """
        Phân bổ thời lượng cho các chunk theo độ dài ký tự + ràng buộc
        CPS/min/max. Trả về list duration (giây) tổng đúng = end-start.
        """
        total_dur = max(0.0, end - start)
        if total_dur <= 0 or not chunks:
            return []

        lens = [max(1, len(c)) for c in chunks]
        total_chars = sum(lens)

        # duration "mong muốn" theo CPS
        desired = [max(self.min_duration, l / self.target_cps) for l in lens]
        # kẹp max
        desired = [min(d, self.max_duration) for d in desired]

        # scale để tổng = total_dur, nhưng ưu tiên các clip chưa chạm min/max
        sum_desired = sum(desired)
        if sum_desired == 0:
            return [total_dur / len(chunks)] * len(chunks)

        scale = total_dur / sum_desired
        scaled = [d * scale for d in desired]

        # đảm bảo không vượt trần và không quá ngắn sau scale
        result = []
        for i, dur in enumerate(scaled):
            dur = max(self.min_duration, min(self.max_duration, dur))
            result.append(dur)

        # chỉnh lần cuối cho tổng khớp chính xác
        diff = total_dur - sum(result)
        if abs(diff) > 1e-3:
            # cộng/trừ dàn đều
            step = diff / len(result)
            for i in range(len(result)):
                result[i] = max(self.min_duration, min(self.max_duration, result[i] + step))
            # sửa sai số nhỏ
            result[-1] += (total_dur - sum(result))

        return result

    def _split_and_time(self, seg):
        start, end, text = seg["start"], seg["end"], seg["text"]
        duration = max(0.0, end - start)
        if duration <= 0 or not text:
            return []

        if self.cjk_mode:
            # 1) token hóa theo từ (không cắt giữa từ)
            tokens = self._tokenize_cjk(text)
            # 2) gom thành các chunk ngắn theo CPL & dấu câu (không dùng break-words cố định)
            chunks = self._chunk_tokens_cjk(tokens)
            if not chunks:
                return []
            # 3) phân bổ thời lượng hợp thức theo CPS/min/max
            durs = self._allocate_durations(start, end, chunks)
            # 4) xuất sub 1 dòng
            subs, cur_t = [], start
            for c, d in zip(chunks, durs):
                subs.append({"start": cur_t, "end": min(end, cur_t + d), "text": c.strip()})
                cur_t += d
            if subs:
                subs[-1]["end"] = end
            return subs

        # ----- non-CJK giữ nguyên logic cũ -----
        chunks = self._soft_split_chunks(text)
        if not chunks:
            return []

        # thời lượng theo CPS mục tiêu + kẹp min/max, rồi scale về seg duration
        durs = [max(len(c)/self.target_cps, self.min_duration) for c in chunks]
        durs = [min(d, self.max_duration) for d in durs]
        sum_raw = sum(durs) or len(chunks)
        scale = duration / sum_raw
        subs, cur_t = [], start
        for c, d in zip(chunks, durs):
            dur = d * scale
            # 2 dòng cho Latin
            half = ceil(len(c)/2)
            pos = c.rfind(' ', 0, half) if ' ' in c else -1
            if pos == -1 or pos < self.max_line_chars//2:
                pos = half
            line1 = c[:pos].strip(); line2 = c[pos:].strip()
            caption_text = line1 if not line2 else f"{line1}\n{line2}"
            subs.append({"start": cur_t, "end": min(end, cur_t + dur), "text": caption_text})
            cur_t += dur
        if subs:
            subs[-1]["end"] = end
        return subs

    # ---------- public ----------
    def write_srt(self, segments, output_path: str):
        if not segments:
            open(output_path, 'w', encoding='utf-8').close()
            return

        # Lọc & clean text
        cleaned = []
        for seg in segments:
            text = self._clean_text(seg.get("text", ""))
            if self._should_skip_segment(text):
                continue
            cleaned.append({"start": seg["start"], "end": seg["end"], "text": text})

        if not cleaned:
            open(output_path, 'w', encoding='utf-8').close()
            return

        # Gộp nhẹ các segment sát nhau (giữ nhịp ngắn)
        merged = []
        cur = cleaned[0].copy()
        for nxt in cleaned[1:]:
            gap = nxt["start"] - cur["end"]
            if gap < self.merge_short_gap and (len(cur["text"]) + len(nxt["text"]) < self.merge_max_chars):
                cur["end"] = nxt["end"]
                cur["text"] = f"{cur['text']}{'' if self.cjk_mode else ' '}{nxt['text']}"
            else:
                merged.append(cur); cur = nxt.copy()
        merged.append(cur)

        # Chia nhỏ + phân thời lượng
        final_subs = []
        for seg in merged:
            final_subs.extend(self._split_and_time(seg))

        # Ghi file .srt
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(final_subs, 1):
                f.write(f"{i}\n{self._format_timestamp(s['start'])} --> {self._format_timestamp(s['end'])}\n{s['text']}\n\n")
        print(f"Written {len(final_subs)} subtitles to file")
