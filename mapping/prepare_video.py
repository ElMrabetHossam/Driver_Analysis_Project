import cv2
import os
import sys

def extract_frames():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, '..', 'video.hevc')
    output_dir = os.path.join(base_dir, 'assets', 'frames')
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if already extracted
    if len(os.listdir(output_dir)) > 100:
        print("✅ Frames already extracted.")
        return

    print(f"🎬 Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as JPG
        out_path = os.path.join(output_dir, f"frame_{count+1}.jpg")
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        
        count += 1
        if count % 100 == 0:
            print(f"   Processed {count} frames...")
            
    cap.release()
    print(f"✅ Successfully extracted {count} frames to {output_dir}")

if __name__ == "__main__":
    extract_frames()
