from PIL import Image

import cv2
import numpy as np
import pytesseract

image_file = "Cifi_test2.jpg"
zeus_icon = "icon/zeus.jpg"
dem_icon = "icon/dem.jpg"
heph_icon = "icon/heph.jpg"
koios_icon = "icon/koios.jpg"

def get_match_area(template, result):
    """
    This function finds the minimum and maximum values, minimum location,
    and width and height of the template match result.

    :param template:
    :param result:

    :return: min_val, max_val, min_loc, max_loc, template_w, template_h:
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    template_w, template_h = template.shape[::-1]  # Extract width and height from result shape
    return min_val, max_val, min_loc, max_loc, template_w, template_h


def draw_contour_match(template, result):

    min_confidence = 0.6 # Minimum confidence level for a match

    # Get data from the template match result
    min_val, max_val, min_loc, max_loc, template_w, template_h = get_match_area(template, result)

    if max_val >= min_confidence:
        # Calculate bottom right corner coordinates (corrected for zero-based indexing)
        bottom_right = (max_loc[0] + template_w - 1, max_loc[1] + template_h - 1)

        # Draw a green rectangle around the match
        cv2.rectangle(img_copy, max_loc, bottom_right, (0, 255, 0), 2)
    else:
        print("not found")


# files
im = cv2.imread(image_file)
template_zeus = cv2.imread(zeus_icon, 0)
template_dem = cv2.imread(dem_icon, 0)
template_heph = cv2.imread(heph_icon, 0)
template_koios = cv2.imread(koios_icon, 0)

# Create a copy for visualization
img_copy = im.copy()

# Gray scale image creation
gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite("temp/gray.jpg", gray_image)

# Match results
result_zeus = cv2.matchTemplate(gray_image, template_zeus, cv2.TM_CCOEFF_NORMED)
result_dem = cv2.matchTemplate(gray_image, template_dem, cv2.TM_CCOEFF_NORMED)
result_heph = cv2.matchTemplate(gray_image, template_heph, cv2.TM_CCOEFF_NORMED)
result_koios = cv2.matchTemplate(gray_image, template_koios, cv2.TM_CCOEFF_NORMED)

draw_contour_match(template_zeus, result_zeus)
draw_contour_match(template_dem, result_dem)
draw_contour_match(template_heph, result_heph)
draw_contour_match(template_koios, result_koios)


cv2.imshow("gray", gray_image)
cv2.waitKey(2000)

# shows other image for testing
cv2.imshow("image", cv2.imread("Cifi_test.jpg", 0))
cv2.waitKey(2000)

# thresh, im_bw = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imwrite("temp/bw_image.jpg", im_bw)




# Display the image with contours
cv2.imshow("Image with Contours", img_copy)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

# ocr_result = pytesseract.image_to_string(gray_image)

# print(ocr_result)

# cv2.imshow("bw", im_bw)
# cv2.waitKey(0)
