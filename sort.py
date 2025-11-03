import torch
import torch.nn as nn
import pickle
import numpy as np
import itertools
import cv2
import os
import glob
import time
from multiprocessing import Pool, cpu_count


from model_lstm import TemporalValidator 


EMBEDDINGS_FILE = "frame_embeddings.pkl"
MODEL_WEIGHTS_FILE = "temporal_validator.pth"
FRAME_DIR = "frames_output/"
OUTPUT_VIDEO_FILE = "reconstructed_video.mp4"


model_global = None
embeddings_global = None
device_global = None


def init_worker(model_weights_path, embeddings_path, device_str):
    """
    Loads the model and embeddings *once* per worker process.
    This avoids loading them hundreds of times.
    """
    global model_global, embeddings_global, device_global
    
    
    device_global = torch.device(device_str)
    
   
    model_global = TemporalValidator(input_size=2048)
    model_global.load_state_dict(torch.load(model_weights_path))
    model_global.to(device_global)
    model_global.eval()
    
    
    with open(embeddings_path, 'rb') as f:
        embeddings_global = pickle.load(f)
    


def check_insertion_spot(args):
    """
    Calculates the 'goodness' score for inserting a frame at a specific spot.
    Runs in a separate parallel process.
    """
    
    global model_global, embeddings_global, device_global
    
    j, sorted_frames, new_frame_key = args
    
    
    temp_list = sorted_frames[:j] + [new_frame_key] + sorted_frames[j:]

    
    def get_prob(frame_key_tuple):
        try:
            emb1 = embeddings_global[frame_key_tuple[0]]
            emb2 = embeddings_global[frame_key_tuple[1]]
            emb3 = embeddings_global[frame_key_tuple[2]]
        except KeyError:
            return 0.0
        
        sequence_data = np.stack([emb1, emb2, emb3])
        sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0).to(device_global)
        
        with torch.no_grad():
            return model_global(sequence_tensor).item()

   
    score_before = 0
    if j > 1: 
        triplet_before = (temp_list[j-2], temp_list[j-1], temp_list[j])
        score_before = get_prob(triplet_before)
    
    score_at = 0
    if j > 0 and j < len(temp_list) - 1: 
        triplet_at = (temp_list[j-1], temp_list[j], temp_list[j+1])
        score_at = get_prob(triplet_at)

    score_after = 0
    if j < len(temp_list) - 2: 
        triplet_after = (temp_list[j], temp_list[j+1], temp_list[j+2])
        score_after = get_prob(triplet_after)
        
    total_score = score_before + score_at + score_after
    
    return (j, total_score) 


if __name__ == "__main__":
    
    print("--- Starting Final Video Reconstruction (Parallel Version) ---")
    start_time = time.time()
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    main_device = torch.device(device_str)
    print(f"Main process: Using device {device_str}")

   
    print("Loading model/embeddings in main process for initialization...")
    try:
        main_model = TemporalValidator(input_size=2048)
        main_model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE))
        main_model.to(main_device)
        main_model.eval()
        
        with open(EMBEDDINGS_FILE, 'rb') as f:
            main_embeddings = pickle.load(f)
            
        frame_keys = list(main_embeddings.keys())
        N = len(frame_keys)
        print(f"Loaded {N} embeddings.")
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: File not found. {e}")
        print("Please make sure 'frame_embeddings.pkl' and 'temporal_validator.pth' exist.")
        exit()

   
    def get_prob_local(frame_key_tuple):
        try:
            emb1 = main_embeddings[frame_key_tuple[0]]
            emb2 = main_embeddings[frame_key_tuple[1]]
            emb3 = main_embeddings[frame_key_tuple[2]]
        except KeyError: return 0.0
        data = np.stack([emb1, emb2, emb3])
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(main_device)
        with torch.no_grad():
            return main_model(tensor).item()

    
    print("Starting sorting process...")
    
    
    initial_frames = frame_keys[:3]
    best_score = -1.0
    best_permutation = ()
    
    for perm in itertools.permutations(initial_frames, 3):
        score = get_prob_local(perm)
        if score > best_score:
            best_score = score
            best_permutation = perm
            
    sorted_frames = list(best_permutation)
    unsorted_frames = frame_keys[3:]
    print(f"Initial 3 frames sorted: {sorted_frames}")

    
    init_args = (MODEL_WEIGHTS_FILE, EMBEDDINGS_FILE, device_str)
    
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=init_args) as pool:
    
        for i, new_frame_key in enumerate(unsorted_frames):
            
           
            tasks = []
            for j in range(len(sorted_frames) + 1):
                
                tasks.append((j, list(sorted_frames), new_frame_key))

        
            results = pool.map(check_insertion_spot, tasks)
           
            best_insert_pos, best_insert_score = max(results, key=lambda item: item[1])
            
            
            sorted_frames.insert(best_insert_pos, new_frame_key)
            
            if (i+1) % 10 == 0 or (i+1) == len(unsorted_frames):
                print(f"  ...sorted {i+4}/{N} frames")

    print("Sorting complete.")
    
    
    print(f"Reconstructing video to {OUTPUT_VIDEO_FILE}...")
    
    first_frame_path = os.path.join(FRAME_DIR, sorted_frames[0])
    first_frame_img = cv2.imread(first_frame_path)
    if first_frame_img is None:
        print(f"FATAL ERROR: Could not read first frame: {first_frame_path}")
        print(f"Check FRAME_DIR ('{FRAME_DIR}') and file extension (.png?).")
        exit()
        
    height, width, layers = first_frame_img.shape
    
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   30, (width, height))

    for frame_key in sorted_frames:
        frame_path = os.path.join(FRAME_DIR, frame_key)
        frame_img = cv2.imread(frame_path)
        if frame_img is not None:
            video_writer.write(frame_img)
        else:
            print(f"Warning: Could not read frame {frame_key}, skipping.")
            
    video_writer.release()
    
 
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- ALL DONE ---")
    print(f"Reconstructed video saved to: {OUTPUT_VIDEO_FILE}")
    print(f"Total execution time (sort.py): {total_time:.2f} seconds")

    with open("Execution_Time_Log.txt", "w") as f:
        f.write(f"Total execution time for sort.py: {total_time:.2f} seconds\n")
        f.write(f"Device used: {device_str}\n")