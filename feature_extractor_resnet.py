import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import cv2
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count


def load_feature_model():
    """Loads the pre-trained ResNet50 model."""
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
   
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval() 
    
  
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    
    return model, transform, device


def extract_embedding_worker(frame_path):
    """Worker function to process one frame."""
    
   
    model, transform, device = load_feature_model()
    
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            return (None, None)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(input_tensor).squeeze().cpu().numpy()
        
        frame_key = os.path.basename(frame_path)
        return (frame_key, embedding)
    except Exception as e:
        print(f"Error on {frame_path}: {e}")
        return (None, None)


if __name__ == "__main__":
    
    FRAME_DIR = "training_frames/"
    OUTPUT_FILE = "train_embeddings_resnet.pkl"
    
    
    print(f"--- ResNet Feature Extraction ---")
    print(f"Looking for frames in: {os.path.abspath(FRAME_DIR)}")
    

    if FRAME_DIR == "training_frames/":
        frame_paths = glob.glob(os.path.join(FRAME_DIR, "*", "*.png"))
    else:
        frame_paths = glob.glob(os.path.join(FRAME_DIR, "*.png"))
    
    if not frame_paths:
        print(f"FATAL ERROR: No .png frames found in '{FRAME_DIR}'.")
        exit()
        
    print(f"Found {len(frame_paths)} frames to process.")
    
    embeddings_dict = {}
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_embedding_worker, frame_paths)
        
        for i, (key, embedding) in enumerate(results):
            if key is not None:
             
                if FRAME_DIR == "training_frames/":
                    
                    full_path = frame_paths[i]
                    video_name = os.path.basename(os.path.dirname(full_path))
                    key = os.path.join(video_name, key)
                
                embeddings_dict[key] = embedding

    if not embeddings_dict:
        print("FATAL ERROR: No embeddings were extracted.")
    else:
        print(f"Successfully extracted {len(embeddings_dict)} embeddings.")
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"All embeddings saved to {OUTPUT_FILE}")