import cv2
from sklearn import svm
import os
import random
import numpy as np
from hog_scratch import *
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# def descriptor_scratch(img) :
#     # compute gradient
#     gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
#     gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
#     gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
#     # angle 
#     gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
#     L = []
#     # for each 8x8 block, compute histogram and append to L
#     for i in range(0, 128, 8):
#         for j in range(0, 64, 8):
#             histogram = [0] * 9
#             for x in range(i, i+8):
#                 for y in range(j, j+8):
#                     angle = gradient_angle[x][y]
#                     magnitude = gradient_magnitude[x][y]
#                     if angle >= 180: angle -= 180
#                     bins = [10, 30, 50, 70, 90, 110, 130, 150, 170]
#                     bin = int(angle/20) # bin would always be between 0 and 8 (inclusive)
#                     c1 = bins[bin]
#                     c2 = bins[(bin+1)%9]
#                     f1 = (c2 - angle)/20
#                     f2 = (angle - c1)/20
#                     # print(bin)
#                     bin%=9
#                     histogram[bin] += magnitude*f1
#                     histogram[(bin+1)%9] += magnitude*f2
#             L.append(histogram)
#     pool = []
#     # combine 4 histograms of 4 8x8 blocks to form a 36x1 vector (image size is 128x64, so 128 8x8 blocks, 15x7=105 pools)
#     for i in range(15):
#         for j in range(7):
#             vector = []
#             vector += L[j+i*8]
#             vector += L[j+1+i*8]
#             vector += L[j+8+i*8]
#             vector += L[j+9+i*8]
#             pool.append(vector)
#     # normalize the 36x1 vector
#     for i in range(len(pool)):
#         magnitude = np.linalg.norm(pool[i])
#         if magnitude != 0:
#             y = [x/magnitude for x in pool[i]]
#             pool[i] = y
            
#     # print(L)
#     # print("len(L)", len(L))
#     # print(len(pool))
#     # combine 105 36x1 vectors to form a 3780x1 vector
#     p = []
#     for x in pool :
#         for y in x :
#             p.append(y)
#     return p        
#     # print(gradient_angle)

def descriptor(img) :    
    hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    h = hog.compute(img)
    return h

def train(open_dataset, closed_dataset) :
    images = []
    labels = []
    for filename in os.listdir(open_dataset):
        if filename.endswith(".jpg"):
            img_path = os.path.join(open_dataset, filename)
            img = cv2.imread(img_path)
            # with probability 0.5 flip the image
            # if random.random() > 0.5: img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 128))
            h = descriptor_scratch(img)
            images.append(h)
            labels.append(0)
    print(len(images), len(labels))
    for filename in os.listdir(closed_dataset):
        if filename.endswith(".jpg"):
            img_path = os.path.join(closed_dataset, filename)
            img = cv2.imread(img_path)
            if random.random() > 0.5: img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 128))
            h = descriptor_scratch(img)
            images.append(h)
            labels.append(1)
    clf = svm.SVC(probability=True)
    print(len(images), len(labels))
    clf.fit(images, labels)
    print(clf.score(images, labels))

    # Predict probabilities for each class
    probs = clf.predict_proba(images)
    # Keep probabilities for the positive class only
    probs_positive = probs[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, probs_positive)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()    

    return clf


# clf = train("Final/open1/train/output", "Final/closed1/train/output")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 training.py <open_dataset_folder> <closed_dataset_folder>")
        sys.exit(1)
    open_dataset = sys.argv[1]
    closed_dataset = sys.argv[2]
    clf = train(open_dataset, closed_dataset)    
    print("Training completed successfully.")
    # save the classifier
    import pickle
    with open("clf.pkl", "wb") as f:
        pickle.dump(clf, f, protocol=2)

    print("Classifier saved as clf.pkl.")