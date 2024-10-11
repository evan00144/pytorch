import cv2

frame = cv2.imread('4.jpg')


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)
lapacian = cv2.Laplacian(gray, cv2.CV_64F)
cv2.imshow('lapacian', lapacian)
cv2.waitKey(0)
laplacian_var = lapacian.var()
print(laplacian_var)