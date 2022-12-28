import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

# Load the image of the tree using OpenCV
image = cv2.imread('input.tif')

plt.tight_layout()
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Otsu's thresholding method to convert the image to binary
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Use opening followed by closing to obtain a mask for the trunk area
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

plt.subplot(2, 3, 2)
plt.imshow(opening)
plt.title("Opening")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=10)
closing = cv2.bitwise_not(closing)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the biggest contour by area
c = max(contours, key = cv2.contourArea)

image_contour = image.copy()
cv2.drawContours(image_contour, [c], -1, 255, 3)

x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(image_contour,(x,y),(x+w,y+h),(0,255,0),2)

plt.subplot(2, 3, 3)
plt.imshow(image_contour)
plt.title("Contour")

# Cut out the biggest contour
mask = np.zeros((image.shape[0],image.shape[1],1), np.uint8)
cv2.drawContours(mask, [c], -1, (255,255,255), -1)

cut = np.zeros(image.shape, np.uint8)
cut = cv2.bitwise_and(image, image, mask = mask)
cut = cut[y:y+h,x:x+w]

plt.subplot(2, 3, 4)
plt.imshow(cut)
plt.title("Cropped")

cut_transform = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)

ret, cut_thresh = cv2.threshold(cut_transform, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

blur = cv2.medianBlur(cut_transform, 11)

plt.subplot(2,3,5)
plt.imshow(blur)
plt.title("Blurred")

adjusted = adjust_contrast_brightness(cut_transform, contrast=5, brightness=-30)

plt.subplot(2,3,6)
plt.imshow(adjusted)
plt.title("Adjusted")

plt.show()