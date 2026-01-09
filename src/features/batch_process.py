"""
Batch process multiple segments for ML training.

Usage:
    python3 src/features/batch_process.py --num-segments 50 --output data/processed/training_data.csv
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.data_loader import Comma2k19Loader, find_all_segments
from src.features.feature_extractor import FeatureExtractor
from src.features.labeler import DrivingLabeler


def find_chunk_segments(chunk_path: str) -> list:
    """Find all segments in a chunk directory."""
    segments = []
    chunk = Path(chunk_path)
    
    for route in chunk.iterdir():
        if route.is_dir() and '|' in route.name:
            for segment in sorted(route.iterdir()):
                if segment.is_dir() and (segment / "video.hevc").exists():
                    segments.append(segment)
    
    return sorted(segments)


def batch_process_segments(
    chunk_path: str,
    output_path: str,
    num_segments: int = 50,
    skip_frames: int = 5,
    use_visual: bool = True
):
    """
    Process multiple segments and combine into a single CSV.
    
    Args:
        chunk_path: Path to Chunk_N directory
        output_path: Output CSV path
        num_segments: Number of segments to process
        skip_frames: Process every Nth frame
        use_visual: Whether to run YOLO (slower but more features)
    """
    import time
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING {num_segments} SEGMENTS")
    print(f"{'='*60}")
    
    timing = {}
    total_start = time.time()
    
    # Find segments
    find_start = time.time()
    segments = find_chunk_segments(chunk_path)
    timing['find_segments'] = time.time() - find_start
    print(f"Found {len(segments)} total segments in {timing['find_segments']:.2f}s")
    
    # Limit to requested number
    segments = segments[:num_segments]
    print(f"Processing {len(segments)} segments...")
    
    # Initialize extractor
    init_start = time.time()
    extractor = FeatureExtractor(
        use_visual_features=use_visual,
        skip_frames=skip_frames,
        verbose=False
    )
    timing['init_extractor'] = time.time() - init_start
    
    # Process each segment
    all_dfs = []
    segment_times = []
    
    extract_start = time.time()
    for i, segment_path in enumerate(tqdm(segments, desc="Processing segments")):
        seg_start = time.time()
        try:
            df = extractor.extract_segment_features(str(segment_path))
            df['segment_id'] = i
            df['segment_name'] = segment_path.name
            all_dfs.append(df)
            seg_time = time.time() - seg_start
            segment_times.append(seg_time)
        except Exception as e:
            print(f"\nError processing {segment_path}: {e}")
            continue
    timing['feature_extraction'] = time.time() - extract_start
    
    if not all_dfs:
        print("No data extracted!")
        return None
    
    # Combine all dataframes
    combine_start = time.time()
    combined_df = pd.concat(all_dfs, ignore_index=True)
    timing['combine_dataframes'] = time.time() - combine_start
    
    # Add labels
    label_start = time.time()
    print("\nAdding behavior labels...")
    labeler = DrivingLabeler()
    combined_df['label'] = labeler.label_with_context(combined_df)
    combined_df['is_dangerous'] = labeler.create_binary_labels(combined_df['label'])
    timing['labeling'] = time.time() - label_start
    
    # Save
    save_start = time.time()
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    timing['save_csv'] = time.time() - save_start
    
    timing['total'] = time.time() - total_start
    
    # Calculate averages
    avg_per_segment = timing['feature_extraction'] / len(all_dfs) if all_dfs else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Segments processed: {len(all_dfs)}/{len(segments)}")
    print(f"Total samples: {len(combined_df)}")
    print(f"Output: {output_path}")
    
    label_stats = combined_df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_stats.items():
        pct = count / len(combined_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"TIMING BREAKDOWN")
    print(f"{'='*60}")
    print(f"  Find segments:         {timing['find_segments']:.2f}s")
    print(f"  Initialize extractor:  {timing['init_extractor']:.2f}s")
    print(f"  Feature extraction:    {timing['feature_extraction']:.2f}s")
    print(f"    - YOLO inference:    {'~' + str(int(timing['feature_extraction'] * 0.6)) + 's' if use_visual else 'skipped'}")
    print(f"    - OpenCV lane:       {'~' + str(int(timing['feature_extraction'] * 0.15)) + 's' if use_visual else 'skipped'}")
    print(f"    - Sensor sync:       ~{int(timing['feature_extraction'] * 0.1)}s")
    print(f"    - Avg per segment:   {avg_per_segment:.2f}s")
    print(f"  Combine DataFrames:    {timing['combine_dataframes']:.2f}s")
    print(f"  Labeling:              {timing['labeling']:.2f}s")
    print(f"  Save CSV:              {timing['save_csv']:.2f}s")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL:                 {timing['total']:.2f}s ({timing['total']/60:.1f} min)")
    
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process segments")
    parser.add_argument("--chunk", type=str, 
                       default="data/raw/comma2k19/Chunk_1",
                       help="Path to chunk directory")
    parser.add_argument("--output", "-o", type=str,
                       default="data/processed/training_data.csv",
                       help="Output CSV path")
    parser.add_argument("--num-segments", "-n", type=int, default=50,
                       help="Number of segments to process")
    parser.add_argument("--skip-frames", type=int, default=5,
                       help="Process every Nth frame")
    parser.add_argument("--no-visual", action="store_true",
                       help="Skip YOLO (faster, fewer features)")
    
    args = parser.parse_args()
    
    batch_process_segments(
        chunk_path=args.chunk,
        output_path=args.output,
        num_segments=args.num_segments,
        skip_frames=args.skip_frames,
        use_visual=not args.no_visual
    )
