# video_finder.py
import os

class VideoFinder:
    """Enhanced video finder with better filtering and sorting."""
    
    def __init__(self, root_path, extensions=None, skip_processed=True):
        """
        Args:
            root_path: Root directory to search
            extensions: List of video extensions (default: common video formats)
            skip_processed: Skip videos that already have .srt files
        """
        self.root_path = root_path
        self.skip_processed = skip_processed
        
        # Comprehensive list of video extensions
        if extensions is None:
            self.extensions = [
                ".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", 
                ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv",
                ".ts", ".mts", ".m2ts", ".vob", ".mp2", ".mpe"
            ]
        else:
            # Normalize extensions
            self.extensions = [
                ext.lower() if ext.startswith('.') else f".{ext.lower()}"
                for ext in extensions
            ]
    
    def _should_skip_file(self, filepath):
        """Check if file should be skipped."""
        # Skip hidden files
        basename = os.path.basename(filepath)
        if basename.startswith('.'):
            return True
        
        # Skip temp files
        if basename.startswith('~') or basename.endswith('.tmp'):
            return True
        
        # Skip if .srt already exists (optional)
        if self.skip_processed:
            srt_path = os.path.splitext(filepath)[0] + ".srt"
            if os.path.exists(srt_path) and os.path.getsize(srt_path) > 0:
                return True
        
        return False
    
    def find_videos(self, sort=True, show_size=True):
        """
        Find all video files in directory tree.
        
        Args:
            sort: Sort files by path
            show_size: Include file size info in results
            
        Returns:
            List of video file paths
        """
        video_files = []
        total_size = 0
        skipped_count = 0
        
        print(f"🔍 Scanning: {self.root_path}")
        
        # Walk through directory tree
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                
                if ext in self.extensions:
                    filepath = os.path.join(dirpath, fname)
                    
                    # Check if should skip
                    if self._should_skip_file(filepath):
                        skipped_count += 1
                        continue
                    
                    # Get file size
                    try:
                        file_size = os.path.getsize(filepath)
                        # Skip very small files (< 100KB, likely corrupted)
                        if file_size < 100 * 1024:
                            continue
                        total_size += file_size
                    except:
                        continue
                    
                    video_files.append(filepath)
        
        # Sort if requested
        if sort:
            video_files.sort()
        
        # Report findings
        if show_size and video_files:
            size_mb = total_size / (1024 * 1024)
            size_gb = size_mb / 1024
            
            if size_gb >= 1:
                size_str = f"{size_gb:.2f} GB"
            else:
                size_str = f"{size_mb:.1f} MB"
            
            print(f"📊 Found: {len(video_files)} videos ({size_str})")
            
            if skipped_count > 0:
                print(f"⏭️  Skipped: {skipped_count} already processed")
        elif video_files:
            print(f"📊 Found: {len(video_files)} videos")
            if skipped_count > 0:
                print(f"⏭️  Skipped: {skipped_count} already processed")
        
        return video_files
    
    def find_videos_without_srt(self):
        """Find only videos that don't have .srt files yet."""
        all_videos = self.find_videos(show_size=False)
        videos_without_srt = []
        
        for video_path in all_videos:
            srt_path = os.path.splitext(video_path)[0] + ".srt"
            if not os.path.exists(srt_path):
                videos_without_srt.append(video_path)
        
        return videos_without_srt