import cv2
import numpy as np
import pytesseract
import os

# percent of the best compare matches
percent = 25;

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# load query image as compare default
imQ = cv2.imread('sampleData/Query.jpg')
h, w, c = imQ.shape

# image detection
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imQ, None)

path = 'compareData'
myPictureList = os.listdir(path)
print(myPictureList)
for j, y in enumerate(myPictureList):
    image = cv2.imread(path + "/" + y)
    kp2, des2 = orb.detectAndCompute(image, None)
    bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bruteForce.match(des2, des1)
    # lower the distance equals better that result
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (percent / 100))]
    # imageMatch = cv2.drawMatches(image, kp2, imQ, kp1, good[:50], None, flags=2)
    # cv2.imshow(y, imageMatch)
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destinationPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, destinationPoints, cv2.RANSAC, 5.0)
    imageScan = cv2.warpPerspective(image, M, (w, h))
    imageScan = cv2.resize(imageScan, (w // 3, h // 3))
    cv2.imshow(y, imageScan)

# out put image
# cv2.imshow("Keypoints", imKp1)
# cv2.imshow("Output", imQ)
cv2.waitKey(0)
