import cv2
from pathlib import Path
import numpy as np

def detect_dominoes(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   # blurred_image = cv2.medianBlur(gray, 5)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 1) # примерно одинаковые
   # blurred_image = cv2.medianBlur(gray, 5)
  #  blurred_image = cv2.GaussianBlur(blurred_image, (3, 3), 0)

 #   _, binary = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blurred_image, 50, 150, L2gradient=True)
    cv2.imshow('found', edges)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if 150 > w > 40 and 150 > h > 40 and 2.5 > w / h > 0.5:
          #  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if 8 >= len(approx) >= 3: 
            #print(len(approx))
                cv2.drawContours(img, [approx], 0, (0, 255, 0), 2) 
    
    cv2.imshow('found_faces', img)
    cv2.waitKey(0)

folder = Path("./train_data/train_data")
files = [f for f in folder.iterdir() if f.is_file()]

for f in files:
    if f.suffix.lower() == '.bmp':
        detect_dominoes(f)
