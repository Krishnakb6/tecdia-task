
import torch, cv2, os, glob, time, pickle, numpy as np, itertools
from multiprocessing import Pool, cpu_count, freeze_support


from model_cnn import MotionValidatorCNN
from model_lstm import TemporalValidatorLSTM


CNN_MODEL_FILE = "temporal_validator_CNN.pth"
LSTM_MODEL_FILE = "temporal_validator_LSTM.pth" 
FRAME_EMBEDDINGS_FILE = "frame_embeddings.pkl" 
FRAME_DIR = "frames_output/"
OUTPUT_VIDEO_FILE = "reconstructed_video_ENSEMBLE.mp4"
IMG_HEIGHT = 128
IMG_WIDTH = 128
CNN_WEIGHT = 0.7 
LSTM_WEIGHT = 0.3 


cnn_model_global, lstm_model_global = None, None
frames_global_dict, embeddings_global_dict = None, None
device_global = None


def init_worker(cnn_path, lstm_path, embed_path, device_str, all_frames_dict):
    """Loads BOTH models and features once per worker process."""
    global cnn_model_global, lstm_model_global, frames_global_dict, embeddings_global_dict, device_global
    
    device_global = torch.device(device_str)
    
    try:
        
        cnn_model_global = MotionValidatorCNN(IMG_HEIGHT, IMG_WIDTH)
        cnn_model_global.load_state_dict(torch.load(cnn_path))
        cnn_model_global.to(device_global).eval()
        
      
        lstm_model_global = TemporalValidatorLSTM(input_size=2048)
        lstm_model_global.load_state_dict(torch.load(lstm_path))
        lstm_model_global.to(device_global).eval()
        
        with open(embed_path, 'rb') as f:
            embeddings_global_dict = pickle.load(f)
        
        frames_global_dict = all_frames_dict
    except Exception as e:
        print(f"WORKER INIT ERROR: {e}")


def check_insertion_spot_ensemble(args):
    """Calculates the weighted score (CNN + LSTM) for one insertion spot."""
    global cnn_model_global, lstm_model_global, frames_global_dict, embeddings_global_dict, device_global
    
    j, sorted_frame_keys, new_frame_key = args
    temp_list = sorted_frame_keys[:j] + [new_frame_key] + sorted_frame_keys[j:]

    def get_ensemble_score(key_tuple):
        try:
        
            img1 = frames_global_dict[key_tuple[0]]
            img2 = frames_global_dict[key_tuple[1]]
            img3 = frames_global_dict[key_tuple[2]]
            diff1, diff2 = cv2.absdiff(img2, img1), cv2.absdiff(img3, img2)
            diffs = np.stack([diff1, diff2], axis=0)
            cnn_input = torch.tensor(diffs / 255.0, dtype=torch.float32).unsqueeze(0).to(device_global)
            with torch.no_grad():
                cnn_score = cnn_model_global(cnn_input).item()

          
            emb1 = embeddings_global_dict[key_tuple[0]]
            emb2 = embeddings_global_dict[key_tuple[1]]
            emb3 = embeddings_global_dict[key_tuple[2]]
            lstm_input_data = np.stack([emb1, emb2, emb3])
            lstm_input = torch.tensor(lstm_input_data, dtype=torch.float32).unsqueeze(0).to(device_global)
            with torch.no_grad():
                lstm_score = lstm_model_global(lstm_input).item()
                
            return (cnn_score * CNN_WEIGHT) + (lstm_score * LSTM_WEIGHT)
        except Exception as e:
           
            return 0.0
    
   
    total_score = 0
    if j > 1: total_score += get_ensemble_score((temp_list[j-2], temp_list[j-1], temp_list[j]))
    if j > 0 and j < len(temp_list) - 1: total_score += get_ensemble_score((temp_list[j-1], temp_list[j], temp_list[j+1]))
    if j < len(temp_list) - 2: total_score += get_ensemble_score((temp_list[j], temp_list[j+1], temp_list[j+2]))
    
    return (j, total_score)

def load_all_frames_to_ram(frame_dir, ext="*.png"):
    """Loads all jumbled frames into memory (RAM) for fast access."""
    print(f"Loading all frames from {frame_dir} into RAM (for CNN)...")
    frame_paths = glob.glob(os.path.join(frame_dir, ext))
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


if __name__ == "__main__":
    freeze_support() 
    
    print("--- Starting Final Video Reconstruction (ENSEMBLE Version) ---")
    start_time = time.time()
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Main process: Using device {device_str}")

    
    all_frames_dict = load_all_frames_to_ram(FRAME_DIR)
    frame_keys = list(all_frames_dict.keys())
    N = len(frame_keys)

   
    try:
        main_cnn_model = MotionValidatorCNN(IMG_HEIGHT, IMG_WIDTH).to(device_str)
        main_cnn_model.load_state_dict(torch.load(CNN_MODEL_FILE))
        main_cnn_model.eval()
        
        main_lstm_model = TemporalValidatorLSTM(input_size=2048).to(device_str)
        main_lstm_model.load_state_dict(torch.load(LSTM_MODEL_FILE))
        main_lstm_model.eval()

        with open(FRAME_EMBEDDINGS_FILE, 'rb') as f:
            main_embeddings = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required file was not found. {e}")
        print("Please ensure all .pth and .pkl files exist before running.")
        exit()

    
    def get_prob_local(key_tuple):
        
        img1 = all_frames_dict[key_tuple[0]]
        img2 = all_frames_dict[key_tuple[1]]
        img3 = all_frames_dict[key_tuple[2]]
        diff1, diff2 = cv2.absdiff(img2, img1), cv2.absdiff(img3, img2)
        diffs = np.stack([diff1, diff2], axis=0)
        tensor_cnn = torch.tensor(diffs / 255.0, dtype=torch.float32).unsqueeze(0).to(device_str)
        with torch.no_grad(): cnn_score = main_cnn_model(tensor_cnn).item()
        
      
        emb1 = main_embeddings[key_tuple[0]]
        emb2 = main_embeddings[key_tuple[1]]
        emb3 = main_embeddings[key_tuple[2]]
        data_lstm = np.stack([emb1, emb2, emb3])
        tensor_lstm = torch.tensor(data_lstm, dtype=torch.float32).unsqueeze(0).to(device_str)
        with torch.no_grad(): lstm_score = main_lstm_model(tensor_lstm).item()
        
        return (cnn_score * CNN_WEIGHT) + (lstm_score * LSTM_WEIGHT)

 
    print("Starting sorting process...")
    
    
    initial_frames = frame_keys[:3]
    best_score, best_permutation = -1.0, ()
    for perm in itertools.permutations(initial_frames, 3):
        score = get_prob_local(perm)
        if score > best_score:
            best_score, best_permutation = score, perm
            
    sorted_frames = list(best_permutation)
    unsorted_frames = [k for k in frame_keys if k not in sorted_frames]
    print(f"Initial 3 frames sorted: {sorted_frames}")

   
    init_args = (CNN_MODEL_FILE, LSTM_MODEL_FILE, FRAME_EMBEDDINGS_FILE, device_str, all_frames_dict)
    
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=init_args) as pool:
        for i, new_frame_key in enumerate(unsorted_frames):
            tasks = [(j, list(sorted_frames), new_frame_key) for j in range(len(sorted_frames) + 1)]
            results = pool.map(check_insertion_spot_ensemble, tasks)
            best_insert_pos, _ = max(results, key=lambda item: item[1])
            sorted_frames.insert(best_insert_pos, new_frame_key)
            
            if (i+1) % 10 == 0 or (i+1) == len(unsorted_frames):
                print(f"  ...sorted {i+4}/{N} frames")

    print("Sorting complete.")
    
   
    print(f"Reconstructing video to {OUTPUT_VIDEO_FILE}...")
    first_frame_path = os.path.join(FRAME_DIR, sorted_frames[0])
    first_frame_orig = cv2.imread(first_frame_path)
    height, width, _ = first_frame_orig.shape
    
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   30, (width, height))

    for frame_key in sorted_frames:
        frame_path = os.path.join(FRAME_DIR, frame_key)
        frame_img = cv2.imread(frame_path)
        if frame_img is not None: video_writer.write(frame_img)
            
    video_writer.release()
    
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- ALL DONE ---")
    print(f"Reconstructed video saved to: {OUTPUT_VIDEO_FILE}")
    print(f"Total execution time (sort_ensemble.py): {total_time:.2f} seconds")

    with open("Execution_Time_Log.txt", "w") as f:
        f.write(f"Total execution time for sort_ensemble.py: {total_time:.2f} seconds\n")
        f.write(f"Device used: {device_str}\n")