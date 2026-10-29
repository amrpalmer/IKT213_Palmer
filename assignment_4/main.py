"""
IKT213 - Lab assignment 4
Harris corner detection + feature-based image alignment.

Alignment approach used: SIFT (2.1) with FLANN matcher + homography,
following https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

Place reference_img.png and align_this.jpg in this folder, then run:
    python main.py
Output: harris.png, aligned.png, matches.png and assignment_4.pdf
"""

import cv2
import numpy as np
from PIL import Image


def harris(reference_image, out_path="harris.png"):
    """Detect Harris corners in the image, mark them in red and save the result."""
    img = reference_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)  # make the marks more visible

    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite(out_path, img)
    return img


def align_images(image_to_align, reference_image, max_features, good_match_precent,
                 aligned_path="aligned.png", matches_path="matches.png"):
    """
    Align image_to_align to reference_image using SIFT + FLANN + homography.

    max_features:       minimum number of good matches required to compute the
                        homography (MIN_MATCH_COUNT in the OpenCV SIFT tutorial)
    good_match_precent: Lowe's ratio-test threshold
    """
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_align, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < good_match_precent * n.distance:
                good.append(m)
    good = sorted(good, key=lambda x: x.distance, reverse=False)

    print(f"Good matches: {len(good)}")

    if len(good) < max_features:
        print(f"Not enough matches found - {len(good)}/{max_features}")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    h, w = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, M, (w, h))
    cv2.imwrite(aligned_path, aligned)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matches_mask,
                       flags=2)
    matches_img = cv2.drawMatches(image_to_align, kp1, reference_image, kp2,
                                  good, None, **draw_params)
    cv2.imwrite(matches_path, matches_img)

    return aligned, matches_img


def make_pdf(image_paths, pdf_path="assignment_4.pdf"):
    """Combine the output images into one PDF, one image per page."""
    pages = [Image.open(p).convert("RGB") for p in image_paths]
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    print(f"Saved {pdf_path}")


def main():
    reference_image = cv2.imread("reference_img.png")
    image_to_align = cv2.imread("align_this.jpg")
    if reference_image is None or image_to_align is None:
        raise FileNotFoundError("Put reference_img.png and align_this.jpg in this folder.")

    harris(reference_image, "harris.png")

    aligned, _ = align_images(image_to_align, reference_image,
                              max_features=10, good_match_precent=0.7)

    if aligned is not None:
        make_pdf(["harris.png", "aligned.png", "matches.png"])


if __name__ == "__main__":
    main()