import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Modify the contrast and brightness
def adjust_contrast_brightness(img, contrast=1.0, brightness=0):
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

# Count the amount of contiguous white pixels in 
# a specific column in an image
def count_column(img, col):
    counter = 0
    isBlob = False

    for i in range(img.shape[0]):
        if img[i][col] > 0:
            isBlob = True
        if img[i][col] < 1:
            if isBlob == True:
                counter += 1
            isBlob = False

    return counter


# Load the image
image = cv2.imread('input.tif')

plt.tight_layout()
plt.subplot(2,3,1)
plt.imshow(image)
plt.title("Original")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Otsu thresholding to convert to binary
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Use opening followed by closing to obtain a mask for the trunk area
# - Since the background is usually composed of large objects out of focus,
# while the trunk area is composed of fine details, the details
# can be eroded away by using opening, which preserves the larger structures,
# and then the larger structures are expanded and the image inverted
# to obtain a good approximation of the trunk area
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(49,49))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=10)
closing = cv2.bitwise_not(closing)

plt.subplot(2,3,2)
plt.imshow(cv2.bitwise_not(closing))
plt.title("Opening + Closing")

# Find the biggest contour by area - this is the trunk
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cont = max(contours, key = cv2.contourArea)

image_contour = image.copy()
cv2.drawContours(image_contour, [cont], -1, 255, 3)

x,y,w,h = cv2.boundingRect(cont)
cv2.rectangle(image_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.subplot(2,3,3)
plt.imshow(image_contour)
plt.title("Trunk Contour")

# Crop out the trunk (corresponding to the biggest contour)
mask = np.zeros((image.shape[0],image.shape[1],1), np.uint8)
cv2.drawContours(mask, [cont], -1, (255,255,255), -1)

crop = np.zeros(image.shape, np.uint8)
crop = cv2.bitwise_and(image, image, mask = mask)
crop = crop[y:y+h, x:x+w]

plt.subplot(2,3,4)
plt.imshow(crop)
plt.title("Cropped")

crop_transform = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# Increase the contrast and decrease the brightness to make the separate rings stand out
contrast = adjust_contrast_brightness(crop_transform, contrast=5, brightness=-30)
bw = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

plt.subplot(2,3,5)
plt.imshow(bw)
plt.title("Enhanced Contrast")

# Use a rectangular kernel to remove most vertical structures
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
horizontal = cv2.erode(bw, horizontalStructure)

plt.subplot(2,3,6)
plt.imshow(horizontal)
plt.title("Vertical Erosion and Sampling")

# Sample 10 evenly spread out columns from the image and count the contiguous
# white pixels in each one
# - The column nearest to the center will almost always have the highest
# ring count, which is useful for finding the center
ringSamples = []
highestSample = 0
iHighestSample = 0

for i in range(10):
    ring = count_column(horizontal, math.floor(horizontal.shape[1] * ((i / 10) + 0.05)))
    ringSamples.append(ring)

    if ring > highestSample:
        highestSample = ring
        iHighestSample = i
    
    # Draw a line at each sampled column
    x1, y1 = [horizontal.shape[1] * ((i / 10) + 0.05), horizontal.shape[1] * ((i / 10) + 0.05)], [0, horizontal.shape[0]]
    plt.axline((x1[0], y1[0]), (x1[1], y1[1]), color="orange")

# Draw a red line at the sampled column nearest to the trunk
# which will be used for calculating the actual number of rings
x1, y1 = [horizontal.shape[1] * ((iHighestSample / 10) + 0.05), horizontal.shape[1] * ((iHighestSample / 10) + 0.05)], [0, horizontal.shape[0]]
plt.axline((x1[0], y1[0]), (x1[1], y1[1]), color="red")

# Print the rings found at each of the sampled column
for i in range(10):
    print("Col: " + str(i) + " | " + str(ringSamples[i]))

# Sample 100 columns near the center of the trunk
centerRings = []
for i in range(-50, 50):
    j = ((iHighestSample / 10) + 0.05) + (i / 1000)

    age = count_column(horizontal, math.floor(horizontal.shape[1] * j))
    centerRings.append(age)

# Calculate the average ring count of 100 pixel columns near the trunk center,
# to account for noise and irregularities in the trunk
rings = 0
for i in range(100):
    rings = rings + centerRings[i]

rings = math.floor(rings / 100)

print()
print("Tree rings: " + str(rings))
plt.suptitle("Tree rings: " + str(rings))

plt.show()