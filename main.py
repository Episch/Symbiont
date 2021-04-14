import cv2
import numpy
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# load query image as compare default
imQ = cv2.imread('sampleData/Query.jpg')

# resize image scale
h, w, c = imQ.shape
imQ = cv2.resize(imQ, (w // 3, h // 3))

# out put image
cv2.imshow("Output", imQ)
cv2.waitKey(0)
