import cv2 as cv
img = cv.imread('data/Testing Data/Glacier/Glacier (311).jpeg')

print(img.shape)
h,w = img.shape[0:2]
rec_w = w//5
# Make middle black
img[:, rec_w*2:rec_w*3, :] = 0

cv.imwrite('result.png', img)
