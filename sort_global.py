import pickle
import os
import cv2
import time
import numpy as np


MATRIX_FILE = "distance_matrix_FLOW.pkl" 
FRAME_DIR = "frames_output/"

OUTPUT_VIDEO_FILE = "reconstructed_video_FINAL.avi" 

def global_sort(frame_keys, distance_matrix):
    """
    Implements the 'Global Sort' algorithm.
    """
    
    groups = {key: [key] for key in frame_keys}
    group_ends = {key: (key, key) for key in frame_keys} 
    
    match_list = []
    for (k1, k2), distance in distance_matrix.items():
        if k1 != k2:
            match_list.append((distance, k1, k2))
            
    
    match_list.sort(key=lambda x: x[0], reverse=False) 
    
    print(f"Starting Global Sort with {len(groups)} groups...")
    
    num_groups = len(groups)
    
    for distance, k1, k2 in match_list:
        if num_groups == 1:
            break
            
        g1 = groups.get(k1)
        g2 = groups.get(k2)
        
        if g1 is None or g2 is None or g1 == g2:
            continue
            
        g1_start, g1_end = group_ends[g1[0]]
        g2_start, g2_end = group_ends[g2[0]]

        merged_group = None
        
        if k1 == g1_end and k2 == g2_start:
            merged_group = g1 + g2
        elif k1 == g1_start and k2 == g2_end:
            merged_group = g2 + g1
        elif k1 == g1_end and k2 == g2_end:
            merged_group = g1 + g2[::-1]
        elif k1 == g1_start and k2 == g2_start:
            merged_group = g1[::-1] + g2
        else:
            continue
            
        new_start, new_end = merged_group[0], merged_group[-1]
        
        for key in merged_group:
            groups[key] = merged_group
            
        group_ends[new_start] = (new_start, new_end)
        group_ends[new_end] = (new_start, new_end)
        
        num_groups -= 1
        
    final_sorted_frames = list(groups.values())[0]
    
    
         
    return final_sorted_frames

if __name__ == "__main__":
    print("--- Step 3: Running Global Sort (Optical Flow Version) ---")
    
    
    start_time_total = time.time()
    
    try:
        with open(MATRIX_FILE, 'rb') as f:
            frame_keys, distance_matrix = pickle.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Matrix file not found: {MATRIX_FILE}")
        print("Please run distance_calculator_flow.py first.")
        exit()
        
    print(f"Loaded Optical Flow distance matrix for {len(frame_keys)} frames.")
    
    
    start_time_sort = time.time()
    sorted_frames = global_sort(frame_keys, distance_matrix)
    end_time_sort = time.time()
    
    print(f"Sorting complete. Final sequence has {len(sorted_frames)} frames.")
    

    
    
    print(f"Reconstructing video to {OUTPUT_VIDEO_FILE}...")
    start_time_reconstruct = time.time()
    
    first_frame_path = os.path.join(FRAME_DIR, sorted_frames[0])
    first_frame_orig = cv2.imread(first_frame_path)
    if first_frame_orig is None:
        print(f"FATAL ERROR: Could not read frame: {first_frame_path}")
        exit()
        
    height, width, _ = first_frame_orig.shape
    
    
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, 
                                    cv2.VideoWriter_fourcc(*'MJPG'), 
                                    30, (width, height))

    for frame_key in sorted_frames:
        frame_path = os.path.join(FRAME_DIR, frame_key)
        frame_img = cv2.imread(frame_path)
        if frame_img is not None:
            video_writer.write(frame_img)
            
    video_writer.release()
    end_time_reconstruct = time.time()
    
    
    end_time_total = time.time()
    
    print(f"\n--- ALL DONE ---")
    print(f"Reconstructed video saved to: {OUTPUT_VIDEO_FILE}")

    
    with open("Execution_Time_Log.txt", "a") as f: # 'a' to append
        f.write(f"2. Global Sort Execution (Merging Groups): {end_time_sort - start_time_sort:.2f} seconds\n")
        f.write(f"3. Video Reconstruction (Writing Frames): {end_time_reconstruct - start_time_reconstruct:.2f} seconds\n")
        f.write(f"4. Total Script Runtime: {end_time_total - start_time_total:.2f} seconds\n")