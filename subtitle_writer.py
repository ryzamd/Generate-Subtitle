# subtitle_writer.py
import re
from math import ceil

class SubtitleWriter:
    """
    Ghi segments thành file .srt với:
    - Lọc noise cơ bản
    - Chia caption theo quy tắc đọc tốt (2 dòng, 42 ký tự mỗi dòng mặc định)
    - Phân bổ thời lượng hợp lý, tránh caption quá dài/ngắn
    """

    def __init__(
        self,
        max_line_chars: int = 42,         # tối đa ký tự mỗi dòng
        max_lines_per_caption: int = 2,   # tối đa số dòng trong 1 caption
        target_cps: float = 15.0,         # chars-per-second mục tiêu
        min_duration: float = 1.0,        # srt tối thiểu cho một caption
        max_duration: float = 6.0,        # srt tối đa cho một caption
        merge_short_gap: float = 1.0,     # gộp segment nếu khoảng cách < 1s
        merge_max_chars: int = 200        # gộp nếu tổng ký tự chưa quá dài
    ):
        self.max_line_chars = max_line_chars
        self.max_lines_per_caption = max_lines_per_caption
        self.target_cps = target_cps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_short_gap = merge_short_gap
        self.merge_max_chars = merge_max_chars

        # Patterns lọc nhiễu
        self._punct_only = re.compile(r"^[\s\W_]+$")
        self._music_pattern = re.compile(r"^[\s\[\(]*(?:♪|♫|music|música|音楽|音乐)[\s\]\)]*$", re.IGNORECASE)
        self._noise_pattern = re.compile(r"^[\s\[\(]*(?:noise|static|silence|inaudible|unintelligible)[\s\]\)]*$", re.IGNORECASE)

    # ---------- utils ----------
    def _format_timestamp(self, seconds: float) -> str:
        total_ms = int(round(seconds * 1000))
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
        if len(text.strip()) < 2:
            return True
        return False

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)   # bỏ space trước dấu câu
        text = re.sub(r'([,.!?;:])(\w)', r'\1 \2', text)  # thêm space sau dấu câu
        # Viết hoa chữ cái đầu nếu cần
        text = text.strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    # ---------- splitting ----------
    def _soft_split_chunks(self, text: str):
        """
        Chia văn bản thành các 'chunk' nhỏ, mỗi chunk sau này sẽ thành một caption (2 dòng).
        Quy tắc: cắt theo dấu câu ưu tiên, sau đó theo khoảng trắng, đảm bảo
        tổng ký tự mỗi caption <= max_line_chars * max_lines_per_caption.
        """
        limit = self.max_line_chars * self.max_lines_per_caption
        words = text.split()
        chunks, current = [], []

        def count_chars(ws):  # đếm với khoảng trắng
            if not ws: return 0
            return sum(len(w) for w in ws) + (len(ws) - 1)

        for w in words:
            if count_chars(current + [w]) <= limit:
                current.append(w)
            else:
                # nếu từ hiện tại quá dài (hiếm), ép xuống caption mới
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

        # Thử tách sâu hơn theo dấu chấm câu để caption tự nhiên hơn
        refined = []
        for c in chunks:
            if len(c) <= limit:
                refined.append(c)
            else:
                # chia theo dấu câu nếu có
                parts = re.split(r'([.?!;:])', c)
                buf = ""
                for p in parts:
                    if p in ".?!;:":
                        buf += p
                        if 0 < len(buf) <= limit:
                            refined.append(buf.strip())
                            buf = ""
                    else:
                        if not buf:
                            buf = p
                        elif len(buf) + len(p) <= limit:
                            buf += p
                        else:
                            refined.append(buf.strip())
                            buf = p
                if buf.strip():
                    refined.append(buf.strip())

        return refined

    def _layout_two_lines(self, chunk: str):
        """
        Từ một chunk (<= limit ký tự), dàn thành tối đa 2 dòng (~ đều ký tự).
        """
        words = chunk.split()
        line1, line2 = [], []
        half = ceil(len(chunk) / 2)
        cur_len = 0
        for w in words:
            if cur_len + (len(w) + (1 if line1 else 0)) <= min(self.max_line_chars, half):
                line1.append(w)
                cur_len = len(" ".join(line1))
            else:
                line2.append(w)
        if not line2 and len(" ".join(line1)) > self.max_line_chars:
            # fallback: cắt cứng theo giới hạn
            s = " ".join(line1)
            return s[:self.max_line_chars], s[self.max_line_chars:]
        return " ".join(line1), " ".join(line2)

    def _split_and_time(self, seg):
        """
        Chia 1 segment thành nhiều caption và phân bổ thời lượng hợp lý.
        """
        start, end, text = seg["start"], seg["end"], seg["text"]
        duration = max(0.0, end - start)
        if duration <= 0 or not text:
            return []

        chunks = self._soft_split_chunks(text)
        if not chunks:
            return []

        # Ước lượng số caption tối thiểu theo tốc độ đọc mục tiêu
        est_min_captions = max(1, ceil(len(text) / (self.target_cps * self.max_duration)))
        if len(chunks) < est_min_captions:
            # nếu text dài mà chia chưa đủ, chia đều hơn
            need = est_min_captions - len(chunks)
            # chia đều các chunk dài nhất
            for _ in range(need):
                longest_idx = max(range(len(chunks)), key=lambda i: len(chunks[i]))
                c = chunks.pop(longest_idx)
                mid = len(c) // 2
                # tách theo khoảng trắng gần mid
                m = re.search(r'\s', c[mid:]) or re.search(r'\s', c[:mid][::-1])
                if m:
                    pos = mid + (m.start() if m.re.pattern == r'\s' and m.string is c[mid:] else -m.start())
                    chunks.insert(longest_idx, c[:pos].strip())
                    chunks.insert(longest_idx + 1, c[pos:].strip())
                else:
                    chunks.insert(longest_idx, c[:mid])
                    chunks.insert(longest_idx + 1, c[mid:])

        # Phân bổ thời lượng theo tỷ lệ ký tự, kẹp theo [min_duration, max_duration]
        total_chars = sum(len(c) for c in chunks)
        # tránh chia 0
        if total_chars == 0:
            return []

        # tính duration sơ bộ
        raw_durs = [max(self.min_duration,
                        min(self.max_duration,
                            max(len(c) / self.target_cps,  # theo CPS
                                duration * (len(c) / total_chars))))  # tỷ lệ seg
                    for c in chunks]

        # scale lại để tổng = duration
        sum_raw = sum(raw_durs)
        if sum_raw > 0:
            scale = duration / sum_raw
            durs = [d * scale for d in raw_durs]
        else:
            durs = [duration / len(chunks)] * len(chunks)

        # Tạo danh sách sub-segments với 2 dòng/ caption
        cur_t = start
        subs = []
        for c, d in zip(chunks, durs):
            line1, line2 = self._layout_two_lines(c)
            caption_text = line1 if not line2 else f"{line1}\n{line2}"
            subs.append({
                "start": cur_t,
                "end": min(end, cur_t + d),
                "text": caption_text.strip()
            })
            cur_t += d

        # đảm bảo cái cuối đúng end (tránh drift)
        if subs:
            subs[-1]["end"] = end

        return subs

    # ---------- public ----------
    def write_srt(self, segments, output_path: str):
        """
        Ghi .srt với:
        - Lọc nhiễu
        - (Tùy) gộp segment quá sát
        - Chia nhỏ caption theo quy tắc đọc
        """
        if not segments:
            open(output_path, 'w', encoding='utf-8').close()
            return

        # Lọc & clean
        cleaned = []
        for seg in segments:
            text = self._clean_text(seg.get("text", ""))
            if self._should_skip_segment(text):
                continue
            cleaned.append({"start": seg["start"], "end": seg["end"], "text": text})

        if not cleaned:
            open(output_path, 'w', encoding='utf-8').close()
            return

        # Gộp segment sát nhau (gap < merge_short_gap) và không quá dài
        merged = []
        cur = cleaned[0].copy()
        for nxt in cleaned[1:]:
            gap = nxt["start"] - cur["end"]
            if gap < self.merge_short_gap and (len(cur["text"]) + len(nxt["text"]) < self.merge_max_chars):
                cur["end"] = nxt["end"]
                cur["text"] = f"{cur['text']} {nxt['text']}"
            else:
                merged.append(cur)
                cur = nxt.copy()
        merged.append(cur)

        # Chia nhỏ + phân bổ thời lượng
        final_subs = []
        for seg in merged:
            final_subs.extend(self._split_and_time(seg))

        # Ghi file
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(final_subs, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_timestamp(s['start'])} --> {self._format_timestamp(s['end'])}\n")
                f.write(s["text"] + "\n\n")

        print(f"Written {len(final_subs)} subtitles to file")
