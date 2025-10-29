import cv2
import numpy as np
import os
import glob
import time # To log the execution time

# --- Configuration ---
FRAMES_DIR = 'frames_output'
FRAME_COUNT = 300

# --- Functions from previous step ---
def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

def compare_histograms(hist1, hist2):
    # Correlation: 1.0 is a perfect match
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# --- New Procedure ---
if __name__ == "__main__":
    print("Loading frame paths...")
    # 1. Get a sorted list of all our extracted frame files
    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.png')))
    
    if len(frame_files) != FRAME_COUNT:
        print(f"Error: Expected {FRAME_COUNT} frames, but found {len(frame_files)}")
    else:
        # ----------------------------------------------------
        # STEP 1: Calculate all 300 histograms (fingerprints)
        # ----------------------------------------------------
        print("Calculating histograms for all 300 frames...")
        start_time = time.time()
        
        all_histograms = []
        for i in range(FRAME_COUNT):
            img = cv2.imread(frame_files[i])
            if img is None:
                print(f"Error reading {frame_files[i]}")
                continue
            all_histograms.append(calculate_histogram(img))
            
        hist_time = time.time()
        print(f"Histogram calculation took: {hist_time - start_time:.2f} seconds")
        
        # ----------------------------------------------------
        # STEP 2: Build the 300x300 Similarity Matrix
        # ----------------------------------------------------
        print("Building 300x300 similarity matrix...")
        
        # Create an empty 300x300 matrix, filled with zeros
        # We use float32 for the scores (e.g., 0.9876)
        similarity_matrix = np.zeros((FRAME_COUNT, FRAME_COUNT), dtype=np.float32)

        # Iterate over every possible pair of frames
        for i in range(FRAME_COUNT):
            if i % 30 == 0:
                print(f"Processing matrix row {i}...")
                
            for j in range(FRAME_COUNT):
                if i == j:
                    # A frame's similarity with itself is a perfect 1.0
                    similarity_matrix[i, j] = 1.0
                    continue
                
                # Get the pre-calculated histograms
                hist1 = all_histograms[i]
                hist2 = all_histograms[j]
                
                # Calculate and store the score
                score = compare_histograms(hist1, hist2)
                similarity_matrix[i, j] = score

        matrix_time = time.time()
        print(f"Matrix calculation took: {matrix_time - hist_time:.2f} seconds")
        print(f"Total execution time: {matrix_time - start_time:.2f} seconds")
        
        # Optional: Save the matrix to a file so we don't have to
        # re-calculate it every time we test our path-finding algorithm
        np.save('similarity_matrix.npy', similarity_matrix)
        print("Similarity matrix saved to 'similarity_matrix.npy'")