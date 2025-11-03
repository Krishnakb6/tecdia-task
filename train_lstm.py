import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
import pickle


from model_lstm import TemporalValidatorLSTM


FRAME_DIR_TRAIN = "training_frames/"   
EMBEDDINGS_FILE_TRAIN = "train_embeddings_resnet.pkl" 
MODEL_OUTPUT_PATH = "temporal_validator_LSTM.pth" 

NUM_EPOCHS = 50    
BATCH_SIZE = 32
LEARNING_RATE = 0.001


class FrameTripletEmbeddingsDataset(Dataset):
    def __init__(self, frame_dir, embeddings_dict):
        self.embeddings = embeddings_dict
        self.video_folders = glob.glob(os.path.join(frame_dir, "*"))
        self.samples = self._generate_samples()
        print(f"Generated {len(self.samples)} total samples.")

    def _generate_samples(self):
        print("Generating training samples (triplets)...")
        samples = []
        for video_folder in self.video_folders:
            frame_files = sorted(glob.glob(os.path.join(video_folder, "*.png")))
            if len(frame_files) < 3:
                continue
            video_name = os.path.basename(video_folder)
            frame_keys = [os.path.join(video_name, os.path.basename(f)) for f in frame_files]
            
            for _ in range(150): 
                try:
                    idx1, idx2, idx3 = sorted(random.sample(range(len(frame_keys)), 3))
                    positive_triplet = (frame_keys[idx1], frame_keys[idx2], frame_keys[idx3])
                    
                    if all(k in self.embeddings for k in positive_triplet):
                        samples.append((positive_triplet, 1.0))
                    
                    shuffled_triplet = list(positive_triplet)
                    random.shuffle(shuffled_triplet)
                    shuffled_triplet = tuple(shuffled_triplet)
                    
                    if (shuffled_triplet != positive_triplet and
                        all(k in self.embeddings for k in shuffled_triplet)):
                        samples.append((shuffled_triplet, 0.0))
                except:
                    continue
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        triplet_keys, label = self.samples[idx]
        try:
            emb1 = self.embeddings[triplet_keys[0]]
            emb2 = self.embeddings[triplet_keys[1]]
            emb3 = self.embeddings[triplet_keys[2]]
        except KeyError as e:
            return torch.zeros((3, 2048)), torch.tensor(0.0, dtype=torch.float32)

        sequence_tensor = torch.tensor(np.stack([emb1, emb2, emb3]), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return sequence_tensor, label_tensor


if __name__ == "__main__":
    
   
    if not os.path.exists(EMBEDDINGS_FILE_TRAIN):
        print(f"FATAL ERROR: Embeddings file not found: {EMBEDDINGS_FILE_TRAIN}")
        print(f"Please run feature_extractor_resnet.py on the 'training_frames/' folder first.")
        exit()
    else:
        print(f"Loading existing embeddings from {EMBEDDINGS_FILE_TRAIN}...")
        with open(EMBEDDINGS_FILE_TRAIN, 'rb') as f:
            embeddings = pickle.load(f)
            
    
    dataset = FrameTripletEmbeddingsDataset(FRAME_DIR_TRAIN, embeddings)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TemporalValidatorLSTM(input_size=2048).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  
    print("Starting LSTM training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"  [Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(dataloader)}] loss: {avg_loss:.4f}")
                running_loss = 0.0
                
    print("Training finished.")
    
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")