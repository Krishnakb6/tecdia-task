import cv2
import os

VIDEO_FILE = 'jumbled_video.mp4' 
OUTPUT_DIR = 'frames_output'     
FRAME_COUNT = 300               

def extract_frames(video_path, output_folder):
    """
    Extracts frames from a video file and saves them to a directory.
    """
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_index = 0
    print("Starting frame extraction...")
    
   
    while cap.isOpened() and frame_index < FRAME_COUNT:
       
        ret, frame = cap.read()

        if ret:
            
            frame_filename = os.path.join(output_folder, f'frame_{frame_index:03d}.png')
            
          
            cv2.imwrite(frame_filename, frame)
            
            if (frame_index + 1) % 30 == 0:
                 print(f"Extracted {frame_index + 1} frames...")
                 
            frame_index += 1
        else:
            
            break

    cap.release()
    print(f"\nExtraction complete. Total frames saved: {frame_index}")


if __name__ == "__main__":

    extract_frames(VIDEO_FILE, OUTPUT_DIR)