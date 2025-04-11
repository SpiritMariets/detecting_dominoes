import cv2
from pathlib import Path
import numpy as np
import itertools

font = cv2.FONT_HERSHEY_SIMPLEX

def find_triangles(image, approx_per):
    cnt = 0
    edges = cv2.Canny(image, 50, 150, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if 150 > w > 40 and 150 > h > 40 and 1.5 > w / h > 0.5:
            epsilon = approx_per * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 3: 
               cnt += 1 
    
    return cnt

def maximaze_triangles(image):
    approx_per, median_ker, gaussian_ker, sigmaX, sigmaY, flag = 0.02, 1, 1, 0, 0, "m"
    max_triangles = -1
    for sgX, sgY in itertools.product(range(-1, 2), range(-1, 2)):
        for approx in range(2, 11):
            for mk in range(1, 9, 2):
                blurred_image = cv2.medianBlur(image, mk)
                cnt = find_triangles(blurred_image, approx / 100)

                if (cnt > max_triangles):
                    max_triangles = cnt
                    approx_per = approx / 100
                    median_ker = mk
                    flag = "m"

                for gk in range(1, 9, 2):
                    blurred_image = cv2.GaussianBlur(blurred_image, (gk, gk), sigmaX=sgX, sigmaY=sgY)
                    cnt = find_triangles(blurred_image, approx / 100)

                    if (cnt > max_triangles):
                        max_triangles = cnt
                        approx_per = approx / 100
                        median_ker = mk
                        gaussian_ker = gk
                        sigmaX = sgX
                        sigmaY = sgY 
                        flag = "mg"
                    
            for gk in range(1, 9, 2):
                blurred_image = cv2.GaussianBlur(image, (gk, gk), sigmaX=sgX, sigmaY=sgY)
                cnt = find_triangles(blurred_image, approx / 100)

                if (cnt > max_triangles):
                    max_triangles = cnt
                    approx_per = approx / 100
                    gaussian_ker = gk
                    sigmaX = sgX
                    sigmaY = sgY
                    flag = "g"

                for mk in range(1, 9, 2):
                    blurred_image = cv2.medianBlur(blurred_image, mk)
                    cnt = find_triangles(blurred_image, approx / 100)

                    if (cnt > max_triangles):
                        max_triangles = cnt
                        approx_per = approx / 100
                        median_ker = mk
                        gaussian_ker = gk
                        sigmaX = sgX
                        sigmaY = sgY
                        flag = "gm"

    return approx_per, median_ker, gaussian_ker, sigmaX, sigmaY, flag

def detect_dominoes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    approx_per, median_ker, gaussian_ker, sigmaX, sigmaY, flag = maximaze_triangles(gray)

    if flag == "m":
        blurred_image = cv2.medianBlur(gray, median_ker)
    elif flag == "mg":
        blurred_image = cv2.medianBlur(gray, median_ker)
        blurred_image = cv2.GaussianBlur(blurred_image, (gaussian_ker, gaussian_ker), sigmaX=sigmaX, sigmaY=sigmaY)
    elif flag == "g":
        blurred_image = cv2.GaussianBlur(gray, (gaussian_ker, gaussian_ker), sigmaX=sigmaX, sigmaY=sigmaY)
    elif flag == "gm":
        blurred_image = cv2.GaussianBlur(gray, (gaussian_ker, gaussian_ker), sigmaX=sigmaX, sigmaY=sigmaY)
        blurred_image = cv2.medianBlur(blurred_image, median_ker)

    edges = cv2.Canny(blurred_image, 50, 150, L2gradient=True)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if 150 > w > 40 and 150 > h > 40 and 1.5 > w / h > 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            epsilon = approx_per * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 3: 
              cv2.drawContours(img, [approx], 0, (0, 255, 0), 2) 
    
    cv2.imshow('found_faces', img)
    cv2.waitKey(0)


folder = Path("./train_data/train_data")
#folder = Path("./train_data/train_data/expert_train")
files = [f for f in folder.iterdir() if f.is_file()]

for f in files:
    if f.suffix.lower() == '.bmp':
        detect_dominoes(f)
