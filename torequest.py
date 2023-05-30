import cv2
import base64
import numpy as np

def resize_image(input_image_path, output_image_path, size):
    img = cv2.imread(input_image_path)
    h, w, _ = img.shape
    if h > w:
        new_h, new_w = size * h / w, size
    else:
        new_h, new_w = size, size * w / h
    img_resized = cv2.resize(img, (int(new_w), int(new_h)))
    cv2.imwrite(output_image_path, img_resized)

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def main(input_image_path, output_image_path, size, output_base64_path):
    #resize_image(input_image_path, output_image_path, size)
    img_base64 = img_to_base64(input_image_path)
    with open(output_base64_path, "w") as text_file:
        text_file.write(img_base64)

# Ejemplo de uso
main('inputs/uvas/IMG_1736.JPG', 'output.jpg', 1536, 'b64.txt')
