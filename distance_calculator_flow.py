
import cv2
import os
import glob
import pickle
import numpy as np
import time
import itertools
from multiprocessing import Pool, cpu_count


FRAME_DIR = "frames_output/"
OUTPUT_FILE = "distance_matrix_FLOW.pkl" 
IMG_HEIGHT = 256 
IMG_WIDTH = 256


frames_global = None 

def load_all_frames_to_ram(frame_dir):
    """Loads and resizes all frames into a dictionary."""
    print("Loading all frames into RAM for Optical Flow...")
    frame_paths = glob.glob(os.path.join(frame_dir, "*.png"))
    if not frame_paths:
        print(f"FATAL ERROR: No .png frames found in {frame_dir}")
        exit()
    
    frames_dict = {}
    for path in frame_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            frames_dict[os.path.basename(path)] = img_resized
            
    print(f"Loaded {len(frames_dict)} frames into RAM.")
    return frames_dict

def init_worker(frames_data):
    """Loads the frame data into each worker's memory."""
    global frames_global
    frames_global = frames_data

def calculate_flow_distance(frame_pair):
    """Worker function: Calculates optical flow magnitude between two frames."""
    global frames_global
    key1, key2 = frame_pair
    
    try:
        img1 = frames_global[key1]
        img2 = frames_global[key2]
        
      
        flow = cv2.calcOpticalFlowFarneback(
            prev=img1, 
            next=img2, 
            flow=None, 
            pyr_scale=0.5, 
            levels=3, 
            winsize=15, 
            iterations=3, 
            poly_n=5, 
            poly_sigma=1.2, 
            flags=0
        )
        
       
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
       
        distance = np.mean(magnitude)
        
        return (key1, key2, distance)
        
    except Exception as e:
        return (key1, key2, float('inf')) 

if __name__ == "__main__":
    print("--- Step 1 (NEW): Calculating Optical Flow Distance Matrix (Parallel) ---")
    start_time = time.time()

    
    frames_data = load_all_frames_to_ram(FRAME_DIR)
    frame_keys = list(frames_data.keys())
    N = len(frame_keys)
    all_pairs = list(itertools.combinations(frame_keys, 2))
    print(f"Creating {len(all_pairs)} comparison tasks...")
    
    distance_matrix = {}
    
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(frames_data,)) as pool:
        
        results = pool.map(calculate_flow_distance, all_pairs)
        
        for key1, key2, distance in results:
           
            distance_matrix[(key1, key2)] = distance
            distance_matrix[(key2, key1)] = distance 

    for key in frame_keys:
        distance_matrix[(key, key)] = float('inf') 
        
    print(f"Distance matrix calculation complete.")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump((frame_keys, distance_matrix), f)
    print(f"Distance matrix saved to {OUTPUT_FILE}")
    
    print(f"Step 1 Time: {time.time() - start_time:.2f} seconds")