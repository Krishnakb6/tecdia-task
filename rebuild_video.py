import cv2
import numpy as np
import os
import glob
import time

# --- Configuration ---
FRAMES_DIR = 'frames_output'     # Folder with your 300 PNG frames
ORDER_FILE = 'optimal_order.npy' # The final path from your TSP solver
OUTPUT_FILE = 'reconstructed_video.mp4'
FRAME_COUNT = 300

# Video properties (must match the original)
VIDEO_FPS = 30.0
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

print(f"--- SCRIPT STARTED: Video Re-assembly ---")

# 1. Load the optimal frame order
try:
    optimal_order = np.load(ORDER_FILE)
    print(f"--- STEP 1: Loaded optimal order from {ORDER_FILE}.")
except FileNotFoundError:
    print(f"Error: Could not find {ORDER_FILE}.")
    print("Please run the 'solve_tsp.py' script first.")
    exit()

if len(optimal_order) != FRAME_COUNT:
    print(f"Error: Order file has {len(optimal_order)} frames, expected {FRAME_COUNT}.")
    exit()

# 2. Get a list of all frame filenames
# We must read them in the original (jumbled) order first
# so the indices from our 'optimal_order' list match
frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.png')))
if len(frame_files) != FRAME_COUNT:
    print(f"Error: Found {len(frame_files)} PNGs, expected {FRAME_COUNT}.")
    exit()

# 3. Set up the Video Writer
# We use the 'mp4v' codec for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

if not out.isOpened():
    print(f"Error: Could not open video writer for {OUTPUT_FILE}.")
    exit()

# 4. Loop, Re-order, and Write Video
print(f"--- STEP 2: Writing {FRAME_COUNT} frames to {OUTPUT_FILE}...")
start_time = time.time()

for i in range(FRAME_COUNT):
    # Get the index of the *next* correct frame
    frame_index = optimal_order[i]
    
    # Get the filename for that frame
    # e.g., optimal_order[0] might be 112, so we get 'frame_112.png'
    frame_filename = frame_files[frame_index]
    
    # Read the image file from disk
    img = cv2.imread(frame_filename)
    
    if img is None:
        print(f"Warning: Could not read {frame_filename}. Skipping.")
        continue
    
    # Ensure the frame is the correct size (just in case)
    if img.shape[1] != VIDEO_WIDTH or img.shape[0] != VIDEO_HEIGHT:
        print(f"Warning: Resizing {frame_filename} to fit video dimensions.")
        img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    # Write the frame to the video file
    out.write(img)
    
    if (i + 1) % 30 == 0:
        print(f"Written {i + 1} / {FRAME_COUNT} frames...")

# 5. Clean up
out.release()
end_time = time.time()
print("--- Video writing complete. ---")
print(f"Total re-assembly time: {end_time - start_time:.2f} seconds.")
print(f"\nSUCCESS! Final video saved to: {OUTPUT_FILE}")