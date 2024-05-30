from PIL import Image

import cv2
import numpy as np
import pytesseract

image_file = "Cifi_test2.png"
cradle_icon = "icon/cradle.PNG"
zeus_icon = "icon/Zeus.PNG"

im = cv2.imread(image_file)
template_cradle = cv2.imread(cradle_icon, 0)
template_zeus = cv2.imread(zeus_icon, 0)

# cv2.imshow("original image", im)
# cv2.waitKey(2000)
#
# inverted_image = cv2.bitwise_not(im)
# cv2.imwrite("temp/inverted.jpg", inverted_image)
#
# cv2.imshow("inverted", inverted_image)
# cv2.waitKey(2000)
#
gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#
cv2.imwrite("temp/gray.jpg", gray_image)

result_cradle = cv2.matchTemplate(gray_image, template_cradle, cv2.TM_CCOEFF_NORMED)
result_zeus = cv2.matchTemplate(gray_image, template_zeus, cv2.TM_CCOEFF_NORMED)

cv2.imshow("gray", gray_image)
cv2.waitKey(2000)

thresh, im_bw = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("temp/bw_image.jpg", im_bw)


# Create a copy for visualization
img_copy = im.copy()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_cradle)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_zeus)

# Extract width and height of the template
template_w, template_h = template_cradle.shape[::-1]
template_w, template_h = template_zeus.shape[::-1]

# Draw a rectangle around the match
cv2.rectangle(img_copy, max_loc, (max_loc[0] + template_w, max_loc[1] + template_h), (0, 255, 0), 2)  # Green bounding box

# Display the image with contours
cv2.imshow("Image with Contours", img_copy)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()



# ocr_result = pytesseract.image_to_string(gray_image)

# print(ocr_result)

# cv2.imshow("bw", im_bw)
# cv2.waitKey(0)

