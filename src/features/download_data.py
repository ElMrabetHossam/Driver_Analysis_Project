"""
Comma2k19 Dataset Downloader

The Comma2k19 dataset is distributed via Academic Torrents (BitTorrent).
This script provides instructions and utilities for downloading and managing the dataset.

Dataset: ~100GB total, ~10GB per chunk
Torrent: http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb

Alternative: Clone the GitHub repo for a 1-minute sample segment.

Usage:
    python -m src.features.download_data --help
    python -m src.features.download_data --sample  # Download sample from GitHub
    python -m src.features.download_data --list    # List downloaded segments
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


# Academic Torrents info
TORRENT_URL = "http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb"
MAGNET_LINK = "magnet:?xt=urn:btih:65a2fbc964078aff62076ff4e103f18b951c5ddb&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
TORRENT_FILE = "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"

# GitHub repo with sample data
GITHUB_REPO = "https://github.com/commaai/comma2k19.git"


def print_download_instructions():
    """Print instructions for downloading the full dataset."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMMA2K19 DATASET DOWNLOAD INSTRUCTIONS                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

The Comma2k19 dataset (~100GB) is distributed via Academic Torrents.

Option 1: Use a BitTorrent client (Recommended)
───────────────────────────────────────────────
1. Install a BitTorrent client:
   - macOS: Transmission (brew install transmission)
   - Linux: qBittorrent, Transmission
   - Windows: qBittorrent

2. Download the .torrent file:
   curl -O https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent

3. Open the .torrent file in your client and select which chunks to download:
   - Chunk 1-2: RAV4 vehicle (~20GB)
   - Chunk 3-10: Civic vehicle (~80GB)

4. Move downloaded data to: data/raw/


Option 2: Use magnet link
─────────────────────────
Copy this magnet link into your BitTorrent client:

magnet:?xt=urn:btih:65a2fbc964078aff62076ff4e103f18b951c5ddb&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php


Option 3: Download sample from GitHub (Small, for testing)
───────────────────────────────────────────────────────────
python3 -m src.features.download_data --sample

This downloads a ~1 minute sample segment from the comma2k19 GitHub repo.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset Info:
- Total size: ~100GB
- 10 chunks × ~10GB each
- 2019 segments of 1 minute each
- ~33 hours of California highway driving
- Includes: video.hevc, CAN bus data, IMU, GPS, radar
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def download_sample(output_dir: str) -> bool:
    """
    Download the sample segment from the GitHub repository.
    
    The repo contains a ~1 minute sample in the 'Example_1' directory.
    
    Args:
        output_dir: Directory to save the sample
        
    Returns:
        True if successful
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sample_dir = output_path / "sample_segment"
    
    if sample_dir.exists():
        print(f"Sample already exists at {sample_dir}")
        return True
    
    print("Downloading sample segment from GitHub...")
    print("This is a ~1 minute sample for testing the pipeline.\n")
    
    # Clone with sparse checkout to get only the example data
    temp_dir = output_path / "_temp_clone"
    
    try:
        # Clone the repo (shallow, sparse)
        print("Step 1/3: Cloning repository (sparse checkout)...")
        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            GITHUB_REPO, str(temp_dir)
        ], check=True, capture_output=True)
        
        # Set sparse-checkout to get Example_1
        print("Step 2/3: Fetching sample data...")
        subprocess.run([
            "git", "-C", str(temp_dir), "sparse-checkout", "set", "Example_1"
        ], check=True, capture_output=True)
        
        # Move the example to our output directory
        print("Step 3/3: Moving sample to data directory...")
        example_src = temp_dir / "Example_1"
        if example_src.exists():
            shutil.move(str(example_src), str(sample_dir))
            print(f"\n✓ Sample downloaded to: {sample_dir}")
        else:
            print("Warning: Example_1 directory not found in repo")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"Error during git clone: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True


def list_segments(data_dir: str) -> list:
    """
    List all segments in the data directory.
    
    Args:
        data_dir: Path to data/raw directory
        
    Returns:
        List of segment paths
    """
    data_path = Path(data_dir)
    segments = []
    
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return segments
    
    # Look for segments in various structures
    
    # 1. Direct sample_segment directory
    sample = data_path / "sample_segment"
    if sample.exists():
        # Check for segment subdirectories
        for item in sample.iterdir():
            if item.is_dir() and (item / "video.hevc").exists():
                segments.append(item)
            # Or the sample might be the segment itself
            elif (sample / "video.hevc").exists():
                segments.append(sample)
                break
    
    # 2. Chunk_N structure (from full dataset)
    for chunk in data_path.glob("Chunk_*"):
        if chunk.is_dir():
            # Look for route directories (format: dongle_id|timestamp)
            for route in chunk.iterdir():
                if route.is_dir() and '|' in route.name:
                    # Look for segment directories (numbered)
                    for segment in sorted(route.iterdir()):
                        if segment.is_dir() and segment.name.isdigit():
                            if (segment / "video.hevc").exists():
                                segments.append(segment)
    
    return sorted(set(segments))


def print_segment_info(segment_path: Path):
    """Print detailed information about a segment."""
    print(f"\n{'─'*60}")
    print(f"Segment: {segment_path}")
    
    # Check for key files
    video = segment_path / "video.hevc"
    processed_log = segment_path / "processed_log"
    global_pos = segment_path / "global_pos"
    
    print(f"  ├── video.hevc: {'✓' if video.exists() else '✗'} ", end="")
    if video.exists():
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"({size_mb:.1f} MB)")
    else:
        print()
    
    print(f"  ├── processed_log/: {'✓' if processed_log.exists() else '✗'}")
    
    if processed_log.exists():
        for sensor_type in processed_log.iterdir():
            if sensor_type.is_dir():
                sensors = [s.name for s in sensor_type.iterdir() if s.is_dir()]
                if sensors:
                    print(f"  │   └── {sensor_type.name}/: {', '.join(sensors[:4])}", end="")
                    if len(sensors) > 4:
                        print(f" (+{len(sensors)-4} more)")
                    else:
                        print()
    
    print(f"  └── global_pos/: {'✓' if global_pos.exists() else '✗'}")


def download_torrent_file(output_dir: str) -> bool:
    """Download the .torrent file for the full dataset."""
    import requests
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    torrent_path = output_path / "comma2k19.torrent"
    
    print(f"Downloading torrent file to {torrent_path}...")
    
    try:
        response = requests.get(TORRENT_FILE, stream=True)
        response.raise_for_status()
        
        with open(torrent_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"\n✓ Torrent file saved to: {torrent_path}")
        print("\nOpen this file with your BitTorrent client to start downloading.")
        print("You can select which chunks (1-10) to download in the client.")
        return True
        
    except Exception as e:
        print(f"Error downloading torrent: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comma2k19 Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.features.download_data              # Show download instructions
  python -m src.features.download_data --sample     # Download sample from GitHub
  python -m src.features.download_data --torrent    # Download .torrent file
  python -m src.features.download_data --list       # List available segments
        """
    )
    parser.add_argument("--sample", action="store_true",
                       help="Download 1-minute sample from GitHub repo")
    parser.add_argument("--torrent", action="store_true",
                       help="Download the .torrent file")
    parser.add_argument("--list", action="store_true",
                       help="List downloaded segments")
    parser.add_argument("--output", "-o", type=str, default="data/raw",
                       help="Output directory (default: data/raw)")
    
    # Legacy arguments (for backwards compatibility)
    parser.add_argument("--chunk", type=int, help="(Deprecated) Use --sample or download via torrent")
    
    args = parser.parse_args()
    
    if args.chunk:
        print("Note: Direct chunk download is not available.")
        print("The dataset is distributed via Academic Torrents.\n")
        print_download_instructions()
        sys.exit(0)
    
    if args.list:
        segments = list_segments(args.output)
        if segments:
            print(f"Found {len(segments)} segment(s) in {args.output}:")
            for seg in segments:
                print_segment_info(seg)
        else:
            print(f"No segments found in {args.output}")
            print("\nRun with --sample to download a test segment.")
    
    elif args.sample:
        success = download_sample(args.output)
        sys.exit(0 if success else 1)
    
    elif args.torrent:
        success = download_torrent_file(args.output)
        sys.exit(0 if success else 1)
    
    else:
        print_download_instructions()
