import cv2
import sys
import os

def read_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img = cv2.imread(os.path.join(folder_path, file))
            images.append(img)
    return images

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    stitched = stitcher.stitch(images)
    if stitched[0] == 0:
        return stitched[1]
    else:
        return None

def main(folder_path):
    images = read_images(folder_path)
    stitched_image = stitch_images(images)

    if stitched_image is not None:
        cv2.imwrite("stitched.jpg", stitched_image)
        print("Imagen combinada guardada como 'stitched.jpg'")
    else:
        print("No se pudo realizar el stitching de las imágenes")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python stitch.py /path/to/folder")
        sys.exit(1)
    folder_path = sys.argv[1]
    main(folder_path)
