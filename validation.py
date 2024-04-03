import os
import cv2
from training import *
import mediapipe as mp
import numpy as np
import pickle

def increase_bbox(bbox, scale_factor):
    x, y, w, h = bbox
    delta_w = int((scale_factor - 1) * w / 2)
    delta_h = int((scale_factor - 1) * h / 2)
    return x - delta_w, y - delta_h, w + 2 * delta_w, h + 2 * delta_h

# path : directory containing test images
def validate(classifierFilePath, valid_open_path, valid_closed_path) : 
    # Load the classifier
    # clf = joblib.load(classifierFilePath)
    with open(classifierFilePath, 'rb') as f:
        clf = pickle.load(f)

    mpHands = mp.solutions.hands
    hands = mp.solutions.hands.Hands(static_image_mode=True, 
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    # positive : closed
    # negative : open

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    for filename in os.listdir(valid_open_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(valid_open_path, filename))
            imgCopy = img.copy()
            # img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
            results = hands.process(imgRGB)
            # hands = []
            if results.multi_hand_landmarks:
                num = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    landmark_points = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        landmark_points.append([x, y])

                
                    landmark_points = np.array(landmark_points)  
                    x_old, y_old, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x_old, y_old, w, h), scale_factor)
                    if x<0 or y<0 or x+w>img.shape[1] or y+h>img.shape[0]:
                        # set img_cropped such that the part which is not fitting is black
                        img_cropped = np.zeros((h, w, 3), np.uint8)
                        h_cropped, w_cropped = img_cropped.shape[:2]
                        x = max(0, x)
                        y = max(0, y)
                        h = min(img.shape[0]-y, h)
                        w = min(img.shape[1]-x, w)
                        img_cropped[:h, :w] = img[y:y+h, x:x+w]
                    else : img_cropped = img[y:y+h, x:x+w]
            
                    test_image = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                    test_image = cv2.resize(test_image, (64, 128))
                    test_image = descriptor_scratch(test_image)
                    # hands.append(test_image)
                    if clf.predict([test_image])[0] == 0: # open
                        true_negative += 1
                        # write "Open" on the image
                        # cv2.putText(imgCopy, "Open", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif clf.predict([test_image])[0] == 1: # closed
                        false_positive += 1
                        # write "Closed" on the image
                        cv2.putText(imgCopy, "Closed", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("img", imgCopy)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

            # cv2.imwrite(os.path.join(output_path, filename), imgCopy)
            # cv2.imshow("img", imgCopy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
                        

    for filename in os.listdir(valid_closed_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(valid_closed_path, filename))
            imgCopy = img.copy()
            # img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            results = hands.process(imgRGB)
            # hands = []
            if results.multi_hand_landmarks:
                num = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    landmark_points = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        landmark_points.append([x, y])

                
                    landmark_points = np.array(landmark_points)  
                    x_old, y_old, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x_old, y_old, w, h), scale_factor)
                    if x<0 or y<0 or x+w>img.shape[1] or y+h>img.shape[0]:
                        # set img_cropped such that the part which is not fitting is black
                        img_cropped = np.zeros((h, w, 3), np.uint8)
                        h_cropped, w_cropped = img_cropped.shape[:2]
                        x = max(0, x)
                        y = max(0, y)
                        h = min(img.shape[0]-y, h)
                        w = min(img.shape[1]-x, w)
                        img_cropped[:h, :w] = img[y:y+h, x:x+w]
                    else : img_cropped = img[y:y+h, x:x+w]
            
                    test_image = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                    test_image = cv2.resize(test_image, (64, 128))
                    test_image = descriptor_scratch(test_image)
                    # hands.append(test_image)
                    if clf.predict([test_image])[0] == 0: # open
                        false_negative += 1
                        # write "Open" on the image
                        # cv2.putText(imgCopy, "Open", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif clf.predict([test_image])[0] == 1: # closed
                        true_positive += 1
                        # write "Closed" on the image
                        # cv2.putText(imgCopy, "Closed", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print("False Positive: ", false_positive)
    print("False Negative: ", false_negative)
    print("True Positive: ", true_positive)
    print("True Negative: ", true_negative)
    print("Accuracy: ", (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)) 
    print("Precision: ", true_positive / (true_positive + false_positive))
    print("Recall: ", true_positive / (true_positive + false_negative))
    print("F1 Score: ", 2 * true_positive / (2 * true_positive + false_positive + false_negative))

    

import sys
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 validation.py <classifier_file_path> <valid_open_path> <valid_closed_path>")
        sys.exit(1)
    validate(sys.argv[1], sys.argv[2], sys.argv[3])

# test("valid")