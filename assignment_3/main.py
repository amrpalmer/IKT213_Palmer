import cv2
import numpy as np
import os

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

    sobelxy = cv2.Sobel(src=blurred, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)

    sobel_8u = cv2.convertScaleAbs(sobelxy)
    cv2.imwrite("sobel_edge_detection.png", sobel_8u)
    return sobel_8u


def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

    edges = cv2.Canny(blurred, threshold1=threshold_1, threshold2=threshold_2)

    cv2.imwrite("canny_edge_detection.png", edges)
    return edges


def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    result = image.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("template_match.png", result)
    return result


def resize(image, scale_factor: int, up_or_down: str):
    result = image.copy()
    factor = scale_factor

    while factor > 1:
        if up_or_down == "up":
            result = cv2.pyrUp(result)
        elif up_or_down == "down":
            result = cv2.pyrDown(result)
        else:
            raise ValueError('up_or_down must be "up" or "down"')
        factor //= 2

    cv2.imwrite(f"resize_{up_or_down}.png", result)
    return result


def main():

    base = os.path.dirname(os.path.abspath(__file__))
    image = cv2.imread(os.path.join(base, "lambo.png"))
    shapes = cv2.imread(os.path.join(base, "shapes-1.png"))
    template = cv2.imread(os.path.join(base, "shapes_template.jpg"))
    if image is None or shapes is None or template is None:
        raise FileNotFoundError(
            "Legg lambo.png, shapes-1.png og shapes_template.jpg i samme mappe som main.py"
        )

    sobel_edge_detection(image)
    canny_edge_detection(image, threshold_1=50, threshold_2=50)
    template_match(shapes, template)
    resize(image, scale_factor=2, up_or_down="up")
    resize(image, scale_factor=2, up_or_down="down")


if __name__ == "__main__":
    main()