import cv2
import os

def print_image_information(image):
    height, width, channels = image.shape
    size = image.size
    dtype = image.dtype

    print(f"height: {height}")
    print(f"width: {width}")
    print(f"channels: {channels}")
    print(f"size: {size}")
    print(f"dtype: {dtype}")

def save_camera_information():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    fps = camera.get(cv2.CAP_PROP_FPS)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    camera.release()

    output_dir = "solutions"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "camera_outputs.txt")

    with open(output_path, "w") as f:
        f.write(f"fps: {int(fps)}\n")
        f.write(f"height: {int(height)}\n")
        f.write(f"width: {int(width)}\n")

    print(f"Camera info saved to {output_path}")

def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)

    save_camera_information()

if __name__ == "__main__":
    main()
