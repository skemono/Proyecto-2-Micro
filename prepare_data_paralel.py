import os
import cv2
import numpy as np
import threading
from queue import Queue
from utils import get_face_landmarks

import time

start_time = time.time()

data_dir = './dataset'
output_file = 'data.txt'

def process_image(image_path, emotion_indx):
    """Process a single image, extract face landmarks, and append emotion label."""
    image = cv2.imread(image_path)
    try:
        face_landmarks = get_face_landmarks(image)
    except:
        return None

    if len(face_landmarks) == 1404:
        face_landmarks.append(int(emotion_indx))
        return face_landmarks
    return None

def process_emotion_directory(emotion, emotion_indx, output_queue):
    """Process all images in a given emotion directory."""
    emotion_output = []
    emotion_dir = os.path.join(data_dir, emotion)
    for imagepath in os.listdir(emotion_dir):
        image_path = os.path.join(emotion_dir, imagepath)
        result = process_image(image_path, emotion_indx)
        if result:
            emotion_output.append(result)
    output_queue.put(emotion_output)

def main():
    output = []
    emotions = sorted(os.listdir(data_dir))
    threads = []
    output_queue = Queue()

    # Create and start a thread for each emotion directory
    for idx, emotion in enumerate(emotions):
        thread = threading.Thread(target=process_emotion_directory, args=(emotion, idx, output_queue))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results from the queue
    while not output_queue.empty():
        output.extend(output_queue.get())

    np.savetxt(output_file, np.asarray(output))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken:", elapsed_time, "seconds")

if __name__ == "__main__":
    main()