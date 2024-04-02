import cv2
from sklearn import svm
import os

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 128))
            h = descriptor(img)
            images.append(h)
            labels.append(0)
    print(len(images), len(labels))
    for filename in os.listdir(closed_dataset):
        if filename.endswith(".jpg"):
            img_path = os.path.join(closed_dataset, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 128))
            h = descriptor(img)
            images.append(h)
            labels.append(1)
    clf = svm.SVC()
    print(len(images), len(labels))
    clf.fit(images, labels)
    print(clf.score(images, labels))
    # print summary of clf

    return clf

clf = train("Final/open1/train/output", "Final/closed1/train/output")
# clf = train("MyHands/open/output", "MyHands/closed/output")

# closed_validation = []
# for filename in os.listdir("Final/closed1/valid/output"):
#     if filename.endswith(".jpg"):
#         test_image = cv2.imread("Final/closed1/valid/output/" + filename)
#         # print(len(closed_validation))
#         test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#         test_image = cv2.resize(test_image, (64, 128))
#         test_image = descriptor(test_image)
#         closed_validation.append(test_image)
# print(clf.predict(closed_validation))

# open_validation = []
# for filename in os.listdir("Final/open1/valid/output"):
#     if filename.endswith(".jpg"):
#         test_image = cv2.imread("Final/open1/valid/output/" + filename)
#         test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#         test_image = cv2.resize(test_image, (64, 128))
#         test_image = descriptor(test_image)
#         open_validation.append(test_image)
# print(clf.predict(open_validation))

# print clf svm summary



# open_validation = []
# file_names = []
# for filename in os.listdir("MyHands/output"):
#     if filename.endswith(".jpg"):
#         test_image = cv2.imread("MyHands/output/" + filename)
#         test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#         test_image = cv2.resize(test_image, (64, 128))
#         test_image = descriptor(test_image)
#         open_validation.append(test_image)
#         file_names.append(filename)
# print(file_names)
# print(clf.predict(open_validation))



# test_image = cv2.imread("Final/open1/valid/output/image_2688_1.jpg")
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# test_image = cv2.resize(test_image, (64, 128))
# test_image = descriptor(test_image)
# print(clf.predict([test_image]))

    



#             H.append(h)
#             L.append(1)
#         except : continue
#     else:
#         continue

# print(H)
# print(L)
# clf = svm.SVC()
# clf.fit(H, L)

    