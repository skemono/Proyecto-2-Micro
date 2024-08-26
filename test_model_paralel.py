import pickle
import cv2
import os
import sys
import csv
from utils import get_face_landmarks
import random


# Define emotions
emotions = ['HAPPY', 'SAD', 'SURPRISED']

def process_batch(image_folder, output_csv):
    # Load the model
    with open('./model', 'rb') as f:
        model = pickle.load(f)

    # Open the CSV file for appending results
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file is empty, write the header
        if os.stat(output_csv).st_size == 0:
            writer.writerow(['Image Path', 'Emotion'])

        # Loop over all images in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)

                # Read the image
                frame = cv2.imread(image_path)
                
                if frame is not None:
                    # Get face landmarks
                    face_landmarks = get_face_landmarks(frame, draw=False, static_image_mode=True)

                    # Predict emotion
                    try:
                        output = model.predict([face_landmarks])
                    except:
                        continue

                    # Write the result to the CSV file
                    writer.writerow([image_path, emotions[int(output[0])]])

                    # Optionally, display the image with the predicted emotion
                    cv2.putText(frame,
                                emotions[int(output[0])],
                                (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3,
                                (0, 255, 0),
                                5)

                    # Create a window with random position and size
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.moveWindow('frame', 120*(int(image_folder[-1])), 500)
                    cv2.resizeWindow('frame', 100, 100)

                    cv2.imshow('frame', frame)
                    cv2.waitKey(25)


    # Release resources
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_model.py <batch_dir> <output_file>")
        sys.exit(1)

    batch_dir = sys.argv[1]
    output_file = sys.argv[2]

    process_batch(batch_dir, output_file)
