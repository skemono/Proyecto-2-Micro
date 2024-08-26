import pickle
import cv2
import os
import sys
import csv
from utils import get_face_landmarks
import random

import time

start_time = time.time()

emotions = ['HAPPY', 'SAD', 'SURPRISED']

def test_model(input_dir, output_file='output.csv', model_path='./model'):
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Open the CSV file in append mode
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                # Process each image and make a prediction
                frame = cv2.imread(file_path)
                if frame is not None:
                    # Get face landmarks
                    face_landmarks = get_face_landmarks(frame, draw=False, static_image_mode=True)

                    # Predict emotion
                    try:
                        output = model.predict([face_landmarks])
                    except:
                        continue

                    # Write the result to the CSV file
                    csv_writer.writerow([file_path, emotions[int(output[0])]])

                    # Optionally, display the image with the predicted emotion
                    cv2.putText(frame,
                                emotions[int(output[0])],
                                (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3,
                                (0, 255, 0),
                                5)

                    cv2.imshow('frame', frame)
                    cv2.waitKey(25)
                

    # Release resources
    cv2.destroyAllWindows()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken:", elapsed_time, "seconds")
                

test_model('./testData', 'output.csv', './model')
