import cv2
import numpy as np

def padding(image, border_width):
    padded = cv2.copyMakeBorder(
        image,
        border_width,  # top
        border_width,  # bottom
        border_width,  # left
        border_width,  # right
        cv2.BORDER_REFLECT
    )
    return padded

def crop(image, x_0, x_1, y_0, y_1):
    height, width = image.shape[:2]
    cropped = image[y_0:height - y_1, x_0:width - x_1]
    return cropped

def resize(image, width, height):
    resized = cv2.resize(image, (width, height))
    return resized

def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]

    return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    shifted = image.astype(np.int16) + hue

    shifted = np.mod(shifted, 256).astype(np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = shifted[y, x, c]

    return emptyPictureArray

def smoothing(image):
    smoothed = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    return smoothed

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("rotation_angle må være 90 eller 180")
    return rotated


if __name__ == "__main__":

    image = cv2.imread("iris-1.png")

    if image is None:
        raise FileNotFoundError(
            "Fant ikke iris-1.png."
        )

    padded_img = padding(image, border_width=100)
    cv2.imwrite("padded.png", padded_img)

    cropped_img = crop(image, x_0=200, x_1=130, y_0=200, y_1=130)
    cv2.imwrite("cropped.png", cropped_img)

    resized_img = resize(image, width=200, height=200)
    cv2.imwrite("resized.png", resized_img)

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copied_img = copy(image, emptyPictureArray)
    cv2.imwrite("copied.png", copied_img)

    gray_img = grayscale(image)
    cv2.imwrite("grayscale.png", gray_img)

    hsv_img = hsv(image)
    cv2.imwrite("hsv.png", hsv_img)

    emptyPictureArray2 = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted_img = hue_shifted(image, emptyPictureArray2, hue=50)
    cv2.imwrite("hue_shifted.png", hue_shifted_img)

    smoothed_img = smoothing(image)
    cv2.imwrite("smoothed.png", smoothed_img)

    rotated_90_img = rotation(image, 90)
    cv2.imwrite("rotated_90.png", rotated_90_img)

    rotated_180_img = rotation(image, 180)
    cv2.imwrite("rotated_180.png", rotated_180_img)

    print("Alle bilder er behandlet og lagret")