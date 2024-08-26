import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from utils import get_face_landmarks

data_dir = './dataset'
output_file = 'data.txt'

def process_image(image_path, emotion_indx):
    """Process a single image, extract face landmarks, and append emotion label."""
    image = cv2.imread(image_path)
    face_landmarks = get_face_landmarks(image)

    if len(face_landmarks) == 1404:
        face_landmarks.append(int(emotion_indx))
        return face_landmarks
    return None

def process_emotion_directory(emotion, emotion_indx):
    """Process all images in a given emotion directory."""
    emotion_output = []
    emotion_dir = os.path.join(data_dir, emotion)
    for image_path_ in os.listdir(emotion_dir):
        image_path = os.path.join(emotion_dir, image_path_)
        result = process_image(image_path, emotion_indx)
        if result:
            emotion_output.append(result)
    return emotion_output

def main():
    output = []
    emotions = sorted(os.listdir(data_dir))
    
    # Use ProcessPoolExecutor to parallelize processing across emotion directories
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_emotion_directory, emotion, idx) for idx, emotion in enumerate(emotions)]
        
        for future in futures:
            output.extend(future.result())

    np.savetxt(output_file, np.asarray(output))

if __name__ == "__main__":
    main()
