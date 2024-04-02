import cv2
import mediapipe as mp
import time
import numpy as np
from training import *

def increase_bbox(bbox, scale_factor):
    x, y, w, h = bbox
    delta_w = int((scale_factor - 1) * w / 2)
    delta_h = int((scale_factor - 1) * h / 2)
    return x - delta_w, y - delta_h, w + 2 * delta_w, h + 2 * delta_h

def open_camera():
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return None
    
    return cap

def close_camera(cap):
    # Release the camera
    cap.release()
    print("Camera closed.")

def take_picture(cap):
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to capture frame.")
        return None
    
    return frame

def main():
    # Open the camera
    cap = open_camera()
    if cap is None:
        return
    
    # Create a MediaPipe Hands object
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    # Initialize the drawing utilities
    mp_drawing = mp.solutions.drawing_utils

    start_time = None
    take_picture_flag = False

    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        isHand = False
        isOpen = False

        # If hands are detected, draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                img = frame_rgb
                landmark_points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    landmark_points.append([x, y])

              
                landmark_points = np.array(landmark_points)  
                x, y, w, h = cv2.boundingRect(landmark_points) 
                scale_factor = 1.3
                x, y, w, h = increase_bbox((x, y, w, h), scale_factor)
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
                test_image = descriptor(test_image)
                
                isHand = True
                if (clf.predict([test_image])[0] == 0): 
                    print("Open")
                    isOpen = True
                elif (clf.predict([test_image])[0] == 1): 
                    print("Closed")
                # Start the timer if not started
                if start_time is None and isOpen:
                    start_time = time.time()
                
        # Check if 5 seconds have passed
        if start_time is not None:
            current_time = time.time()
            if current_time - start_time >= 5:
                take_picture_flag = True

        # Display the frame
        cv2.imshow('Camera', frame)

        # Take a picture if the flag is set
        if take_picture_flag:
            # Take a picture
            picture = take_picture(cap)
            if picture is not None:
                # Save the picture
                cv2.imwrite('picture.jpg', picture)
                print("Picture saved as 'picture.jpg'")
                take_picture_flag = False
                start_time = None

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            close_camera(cap)
            break

        if not isOpen and isHand:
            close_camera(cap)
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
