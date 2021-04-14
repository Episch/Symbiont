import cv2
import numpy as np
import pytesseract
import os

# percent of the best compare matches
percent = 25
pixelThreshold = 160

roi = [[(123, 65), (259, 76), 'text', 'Name'],
       [(71, 214), (90, 224), 'box', 'Vehicletype']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# load query image as compare default
imQ = cv2.imread('sampleData/Query.jpg')
h, w, c = imQ.shape

# image detection
orb = cv2.ORB_create(1100)
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

    M, _ = cv2.findHomography(srcPoints, destinationPoints, cv2.RANSAC, 2.0)
    imageScan = cv2.warpPerspective(image, M, (w, h))
    # imageScan = cv2.resize(imageScan, (w // 3, h // 3))
    # cv2.imshow(y, imageScan)
    imageShow = imageScan.copy()
    imageMask = np.zeros_like(imageShow)

    myData = []
    print(f' Extracting from {myPictureList[j]}')

    for x, r in enumerate(roi):
        cv2.rectangle(imageMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
        imageShow = cv2.addWeighted(imageShow, 0.99, imageMask, 0.1, 0)

        imageCrop = imageScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(str(x), imageCrop)

        if r[2] == 'text':
            print(f'{r[3]}: {pytesseract.image_to_string(imageCrop)}')
            text = myData.append(pytesseract.image_to_string(imageCrop))
        if r[2] == 'box':
            imageGray = cv2.cvtColor(imageCrop, cv2.COLOR_BGR2GRAY)
            imageThresh = cv2.threshold(imageGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imageThresh)
            if totalPixels>pixelThreshold: totalPixels=1
            else: totalPixels=0
            print(f'{r[3]}: {totalPixels}')
            myData.append(totalPixels)

    cv2.imshow(y + "2", imageShow)
# out put image
# cv2.imshow("Keypoints", imKp1)
# cv2.imshow("Output", imQ)
cv2.waitKey(0)
