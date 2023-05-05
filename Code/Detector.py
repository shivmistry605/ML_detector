# example: python detector.py --image_path C:/Source/ML.Detection/Data/Identity/75c905df157de24e9cadd48b010f362e.jpg --model_path "C:/Source/ML.Detection/ML models/haarcascade_frontalface_default.xml"

import pandas as pd
import cv2
import pytesseract
from IPython.core.interactiveshell import InteractiveShell
import warnings
import argparse

def _detect_face(path_to_model,image_path):
    # Load the classifier and image
    face_cascade = cv2.CascadeClassifier(path_to_model)
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If a face is detected, return True, otherwise return False
    if len(faces) > 0:
        return True
    else:
        return False
    
def _detect_text(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to remove noise and enhance text
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(thresh)
    
    # If text is detected, return the text, otherwise return False
    if text:
        return text
    else:
        return False
    
def detector(path_to_model,image_path):
    print("> Starting to detect from the image provided ...")
    if _detect_face(path_to_model,image_path):
        print("\t > A face has been detected in the image ...")

        if _detect_text(image_path):
            text = _detect_text(image_path).split()
            print(f"\t > Text identified from the imgae..")
            print("> We conclude that the image is an official document ...")
            return "Identity"
        else:
            print("\t > No text extracted from this image ...")
            print("> We conclude that the image is just a head shot ...")
            return "Headshot"
    else:
        print("\t > Faces or text not detected in the image ...") 
        return "Unknown"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to image', required=True)
    parser.add_argument('--model_path', type=str, help='Path to model', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f'> Reading face detection model from {args.model_path} ...')
    print(f'> Reading image from {args.image_path} ...')
    print('\n')
    outcome = detector(args.model_path,args.image_path) 
    print(f"\nOutcome: {outcome}")
    return

if __name__ == '__main__':
    main()