import cv2 


# img = cv2.imread("normal5.png")
# downscaled = cv2.resize(img, (64, 64))
# cv2.imwrite("downscaled.png", downscaled)

img = cv2.imread("ndf.png")

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        print(img[x,y])

cv2.imshow("test", cv2.resize((img/10.0),(256,256)))
cv2.waitKey(0)