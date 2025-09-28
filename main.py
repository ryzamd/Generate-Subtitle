# main.py
import sys, os, time
from video_finder import VideoFinder
from subtitle_generator import SubtitleGenerator
from subtitle_writer import SubtitleWriter

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python main.py <thư_mục_video> [language]")
        print("  language: auto (default), zh, en, ja, ko, ...")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Lỗi: '{folder_path}' không phải thư mục hợp lệ."); sys.exit(1)

    language = None
    if len(sys.argv) > 2:
        lang_arg = sys.argv[2].lower()
        language = None if lang_arg == "auto" else lang_arg
        print(f"📌 {'Auto-detect language mode' if language is None else f'Force language: {language}'}")
    else:
        print("📌 Auto-detect language mode (use 'python main.py <folder> zh' for Chinese)")

    finder = VideoFinder(folder_path)
    video_files = finder.find_videos()  # :contentReference[oaicite:6]{index=6}
    if not video_files:
        print("Không tìm thấy file video nào."); sys.exit(0)

    print(f"\n📁 Found {len(video_files)} video(s) to process")
    print("="*60)

    generator = SubtitleGenerator(model_size="small", prefer_gpu=True)  # :contentReference[oaicite:7]{index=7}

    total_start = time.time(); success_count = 0; error_count = 0

    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        out_srt = os.path.splitext(video_path)[0] + ".srt"
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")

        if os.path.exists(out_srt):
            print(f"  ⏭️  Subtitle already exists, skipping: {out_srt}"); success_count += 1; continue

        try:
            start_time = time.time()

            # Nếu ép zh -> bật VAD + word timestamps chính xác hơn cho cắt ngắn
            want_zh = (language or "").startswith("zh")
            segments, detected = generator.generate_segments(
                video_path,
                language=language,                              # None = auto
                vad=True,                                       # bỏ intro im lặng
                beam_size=5,
                word_timestamps=True if want_zh else False      # zh: chính xác tách mốc
            )

            # Quyết định CJK mode theo ngôn ngữ ép hoặc detect
            is_zh = want_zh or (detected and str(detected).startswith("zh"))
            if is_zh:
                writer = SubtitleWriter(
                    cjk_mode=True,
                    max_line_chars=14,          # 1 dòng ~12–16 ký tự
                    max_lines_per_caption=1,
                    target_cps=9.5,             # CPS mục tiêu cho zh (≤11 trần)  :contentReference[oaicite:8]{index=8}
                    min_duration=1.0,
                    max_duration=3.0,
                    merge_short_gap=0.2,
                    merge_max_chars=60
                )
            else:
                # GIỮ NGUYÊN logic cũ cho ngôn ngữ khác
                writer = SubtitleWriter()       # 2 dòng, 42 CPL, CPS~15, 1–6s  :contentReference[oaicite:9]{index=9}

            process_time = time.time() - start_time
            writer.write_srt(segments, out_srt)                 # :contentReference[oaicite:10]{index=10}

            file_size = os.path.getsize(out_srt)
            if file_size > 0:
                print(f"  ✅ Created subtitle: {out_srt}")
                print(f"     Size: {file_size:,} bytes | Segments: {len(segments)} | Time: {format_time(process_time)}")
                success_count += 1
            else:
                print(f"  ⚠️  Created empty subtitle (no speech detected): {out_srt}")
                success_count += 1

        except KeyboardInterrupt:
            print("\n\n⛔ Process interrupted by user!")
            print(f"Progress: {success_count}/{len(video_files)} completed"); sys.exit(1)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            try:
                open(out_srt, 'w').close(); print(f"  → Created empty .srt: {out_srt}")
            except: pass
            error_count += 1

    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    print(f"  ✅ Success: {success_count}/{len(video_files)}")
    if error_count > 0: print(f"  ❌ Errors: {error_count}/{len(video_files)}")
    print(f"  ⏱️  Total time: {format_time(total_time)}")
    if success_count > 0: print(f"  ⚡ Average: {format_time(total_time/success_count)} per video")
    print("="*60)
