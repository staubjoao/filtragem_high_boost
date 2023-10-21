import cv2
import numpy as np


def show(name, img):
    # shows the image
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# image loading and prepping
image = cv2.imread(
    'img1.tif',
    cv2.IMREAD_GRAYSCALE
    )
image = np.array(image)
image = image.astype("int16")  # we need negative values for the mask

# 5x5 smoothing (averaging) filter mask (Gonzalez uses Gaussian filter)
smoothing_mask = (1 / 25) * np.ones((5, 5))

# main part as explained in the book (Woods and Gonzalez) ---------------------

# 1. blur the original image
blurred_img = cv2.filter2D(image, -1, smoothing_mask)

# 2. subtract the blurred image from the original
mask = image - blurred_img

# 3. add the mask (weighted) to the original
k = 4.5  # k > 1 for highboost filtering
sharpened = image + k * mask

# end main part ---------------------------------------------------------------

# clip everybody into [0, 255]
sharpened = np.clip(sharpened, 0, 255)

show("Original", image.astype("uint8"))
show("Blurred", blurred_img.astype("uint8"))
show("Mask", mask.astype("uint8"))
show("Highboost", sharpened.astype("uint8"))
show("Original (again)", image.astype("uint8"))
