# main.py
import sys
import os
import time
from video_finder import VideoFinder
from subtitle_generator import SubtitleGenerator
from subtitle_writer import SubtitleWriter

def format_time(seconds):
    """Format seconds to readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python main.py <thư_mục_video> [language]")
        print("  language: auto (default), zh (Chinese), en (English), ja (Japanese), ko (Korean), etc.")
        print("Ví dụ:")
        print("  python main.py \"D:\\Videos\" auto  # Auto-detect language")
        print("  python main.py \"D:\\Videos\" zh    # Force Chinese")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Lỗi: '{folder_path}' không phải thư mục hợp lệ.")
        sys.exit(1)
    
    # Language parameter (default: auto-detect)
    language = None  # None = auto-detect
    if len(sys.argv) > 2:
        lang_arg = sys.argv[2].lower()
        if lang_arg != 'auto':
            language = lang_arg
            print(f"📌 Force language: {language}")
        else:
            print("📌 Auto-detect language mode")
    else:
        print("📌 Auto-detect language mode (use 'python main.py <folder> zh' for Chinese)")

    # Find videos
    finder = VideoFinder(folder_path)
    video_files = finder.find_videos()
    if not video_files:
        print("Không tìm thấy file video nào.")
        sys.exit(0)
    
    print(f"\n📁 Found {len(video_files)} video(s) to process")
    print("=" * 60)

    # Initialize generator and writer
    generator = SubtitleGenerator(model_size="small", prefer_gpu=True)
    writer = SubtitleWriter()

    # Statistics
    total_start = time.time()
    success_count = 0
    error_count = 0
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        out_srt = os.path.splitext(video_path)[0] + ".srt"
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")
        
        # Check if subtitle already exists
        if os.path.exists(out_srt):
            print(f"  ⏭️  Subtitle already exists, skipping: {out_srt}")
            success_count += 1
            continue
        
        try:
            start_time = time.time()
            
            # Generate segments with optimized settings
            segments = generator.generate_segments(
                video_path, 
                language=language,  # None = auto-detect, or specific like 'zh', 'en'
                vad=True,          # BẬT VAD để bỏ qua im lặng
                beam_size=5,       # Giảm beam_size để nhanh hơn
                word_timestamps=False  # Tắt word timestamps để nhanh hơn
            )
            
            process_time = time.time() - start_time
            
            # Write SRT
            writer.write_srt(segments, out_srt)
            
            # Check result
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
            print(f"Progress: {success_count}/{len(video_files)} completed")
            sys.exit(1)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            # Create empty .srt to mark as processed
            try:
                open(out_srt, 'w').close()
                print(f"  → Created empty .srt: {out_srt}")
            except:
                pass
            error_count += 1

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"  ✅ Success: {success_count}/{len(video_files)}")
    if error_count > 0:
        print(f"  ❌ Errors: {error_count}/{len(video_files)}")
    print(f"  ⏱️  Total time: {format_time(total_time)}")
    if success_count > 0:
        print(f"  ⚡ Average: {format_time(total_time/success_count)} per video")
    print("=" * 60)