import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

dir='stitches/stitched-'
name = 'glacier_test.jpeg'
# Load the image 
image = cv2.imread(dir + name) 
  
  
# Create the sharpening kernel 
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
  
# Sharpen the image 
sharpened_image = cv2.filter2D(image, -1, kernel)
#Save the image 
cv2.imwrite('sharpened-' + name, sharpened_image) 
