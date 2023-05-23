import cv2
import json
import os
import numpy as np
import re
import torchvision
import torch
import math


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Inicializar las dimensiones de la imagen y obtener el tamaño original
    dim = None
    (h, w) = image.shape[:2]

    # Si se proporcionan ambos, el ancho y el alto, se ignorará la relación de aspecto
    if width is not None and height is not None:
        dim = (width, height)

    # Si solo se proporciona el ancho, calcular el alto manteniendo la relación de aspecto
    elif width is not None:
        r = width / float(w)
        dim = (width, int(h * r))

    # Si solo se proporciona el alto, calcular el ancho manteniendo la relación de aspecto
    elif height is not None:
        r = height / float(h)
        dim = (int(w * r), height)

    # Si no se proporciona ninguna dimensión, retornar la imagen original
    if dim is None:
        return image

    # Redimensionar la imagen y devolverla
    resized_image = cv2.resize(image, dim, interpolation=inter)
    return resized_image


def non_max_suppression(detections, iou_thres=0.2):
    bboxes, confs, clss = detections  # bboxes in xyxy format
    
    idx = torchvision.ops.nms(bboxes, confs, iou_thres)  # NMS
    idx_np = idx.cpu().numpy()
    
    idx_np = postprocess_nms(idx_np, bboxes, inclusion_threshold=0.2)
    
    return idx_np

def postprocess_nms(idx_np, bboxes, inclusion_threshold=0.9):
    def relative_area(box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / box1_area

    filtered_idxs = []

    for i in idx_np:
        box1 = bboxes[i]
        remove = False
        for j in idx_np:
            if i == j:
                continue
            box2 = bboxes[j]
            if relative_area(box1, box2) >= inclusion_threshold:
                remove = True
                break
        if not remove:
            filtered_idxs.append(i)

    return np.array(filtered_idxs)

def load_annotations(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def save_annotations(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
        
def find_homography(img_s, img_d):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img_s, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_d, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography
        
def find_homography_fast(img_s, img_d):
    # Crear un objeto ORB
    orb = cv2.ORB_create(1000)

    # Detectar y calcular características y descriptores
    keypoints1, descriptors1 = orb.detectAndCompute(img_s, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_d, None)

    # Crear un objeto FLANN Matcher con índices de árboles kd
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Realizar la correspondencia entre los descriptores utilizando FLANN Matcher
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Aplicar el criterio de proporción de Lowe para filtrar las correspondencias buenas
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    return homography

def find_homography_slow(img_s, img_d):
    # Crear un objeto AKAZE
    akaze = cv2.AKAZE_create()

    # Detectar y calcular características y descriptores
    keypoints1, descriptors1 = akaze.detectAndCompute(img_s, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(img_d, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    return homography

def transform_bbox(bbox, homography):
    x_min, y_min, x_max, y_max = bbox
    points = np.float32([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, homography)

    x_min_transformed = round(min(transformed_points[:, 0, 0]),0)
    y_min_transformed = round(min(transformed_points[:, 0, 1]),0)
    x_max_transformed = round(max(transformed_points[:, 0, 0]),0)
    y_max_transformed = round(max(transformed_points[:, 0, 1]),0)

    return [x_min_transformed, y_min_transformed, x_max_transformed, y_max_transformed]

def coco_to_bbox(bbox, img_size):
    x_min_rel, y_min_rel, x_max_rel, y_max_rel = bbox
    img_width, img_height = img_size

    x_min = x_min_rel * img_width
    y_min = y_min_rel * img_height
    x_max = x_max_rel * img_width
    y_max = y_max_rel * img_height

    return [x_min, y_min, x_max, y_max]

def bbox_to_coco(bbox, img_size):
    x_min, y_min, x_max, y_max = bbox
    img_width, img_height = img_size

    x_min_rel = x_min / img_width
    y_min_rel = y_min / img_height
    x_max_rel = x_max / img_width
    y_max_rel = y_max / img_height

    return [x_min_rel, y_min_rel, x_max_rel, y_max_rel]

def stitch_images(img1, img2):
    #img1_resized = resize_image(np.copy(img1), width=1024, height=1024)
    #img2_resized = resize_image(np.copy(img2), width=1024, height=1024)
    
    #homography = find_homography(img2_resized, img1_resized)
    homography = find_homography(img2, img1)

    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    stitched_img = cv2.warpPerspective(img2, homography, (int(width1*1.5), int(height1*1.5)))

    # Encontrar la ubicación donde img2 se mezclará en stitched_img
    mask = np.all(stitched_img[:height1, :width1] == 0, axis=2)
    y_offset, x_offset = np.argwhere(mask).T

    # Mezclar img1 en la posición adecuada de stitched_img
    stitched_img[y_offset, x_offset] = img1[y_offset, x_offset]

    return stitched_img, homography

def draw_annotations_on_image(image, annotations):
    for annotation in annotations:
        bbox = coco_to_bbox(annotation["bbox"], (annotation["width"],annotation["height"]))
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]

        color = (0, 0, 255) # Color verde en formato BGR
        thickness = 4

        # Dibuja el rectángulo delimitador (bounding box) en la imagen
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    return image

def filter_bboxes(image, annotations):
    bboxes = []
    confs = []
    clss = []
    for annotation in annotations:
        bboxes.append(coco_to_bbox(annotation["bbox"], (annotation["width"],annotation["height"])))
        confs.append(annotation["score"])
        clss.append(annotation["category_id"])
    detections = torch.FloatTensor(bboxes), torch.FloatTensor(confs), torch.FloatTensor(clss)
    idxs = non_max_suppression(detections, iou_thres=0.15)
    nms_annotations = []
    for ii, annotation in enumerate(annotations):
        if ii in idxs:
            nms_annotation = annotation.copy()
            bbox = coco_to_bbox(annotation["bbox"], (annotation["width"],annotation["height"]))
            nms_annotation["width"] = image.shape[1]
            nms_annotation["height"] = image.shape[0]
            nms_annotation["bbox"] = bbox_to_coco(bbox,((nms_annotation["width"],nms_annotation["height"])))
            nms_annotations.append(nms_annotation)
            
    nms_img = draw_annotations_on_image(np.copy(image), nms_annotations)
    
    return nms_img, nms_annotations

def crop_image(img):
    mask = np.all(img == 0, axis=2)
    ys, xs = np.where(~mask)

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    img_cropped = img[y_min:y_max, x_min:x_max]
    return img_cropped


def main():
    in_folder_path = "inputs/test_stitch_dos"
    
    output_path = "output/stitching"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    images = [f for f in os.listdir(in_folder_path) if f.endswith(".jpg") or f.endswith(".png")]
    images = sorted(images, key=lambda x: int(re.findall(r'\d+', x)[0]))

    num_images = len(images)
    #num_iters = num_images - 1
    num_iters = round(math.log2(num_images))
    
    folder_path = in_folder_path
    
    #for iter in range(num_iters -1):
    for iter in range(num_iters):
        
        images = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]
        images = [f for f in images if not f.startswith("joined_") and not f.startswith("nms_")]
        images = sorted(images, key=lambda x: int(re.findall(r'\d+', x)[0]))
        num_images = len(images)
        
        output_path_iter = os.path.join(output_path,"iter" + str(iter)) 
        if not os.path.exists(output_path_iter):
            os.makedirs(output_path_iter)
        
        #for i in range(num_images - 1):
        for i in range(round(num_images/2)):
            if i == (round(num_images/2)-1) and (num_images%2) !=0:
                file1_name = images[i*2].split(".")[0]
                img1_path = os.path.join(folder_path, images[i])
                img1TXT_path = file1_path + ".txt"
                file1_path = img1_path.split(".")[0]
                img1 = cv2.imread(img1_path)
                stitched_img = img1
                with open(img1TXT_path) as f:
                    imgfixed_annotations = json.load(f)
                transformed_annotations = []
                for annotation in imgfixed_annotations["annotations"]:
                    transformed_annotation = annotation.copy()
                    transformed_annotation["bbox"] = annotation["bbox"] if type(annotation["bbox"]) is list else eval(annotation["bbox"])
                    transformed_annotations.append(transformed_annotation)
                img_stitch_path = os.path.join(output_path_iter, str(i) + ".png")
                cv2.imwrite(img_stitch_path, stitched_img)
                out_annotations = {"annotations": []}
                out_annotations["annotations"] = transformed_annotations
                txt_nms_path = os.path.join(output_path_iter, str(i) + ".txt")
                save_annotations(txt_nms_path, out_annotations)
            else:    
                file1_name = images[i*2].split(".")[0]
                file2_name = images[i*2+1].split(".")[0]
                img1_path = os.path.join(folder_path, images[i*2])
                img2_path = os.path.join(folder_path, images[i*2+1])
                file1_path = img1_path.split(".")[0]
                file2_path = img2_path.split(".")[0]
                img1TXT_path = file1_path + ".txt"
                img2TXT_path = file2_path + ".txt"
                
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)

                stitched_img, homography = stitch_images(img1, img2)
                
                if stitched_img is None:
                    print(f"Error: No se pudo realizar el stitching de las imágenes {i} y {i+1}.")
                    continue

                # Guardar homography
                homography_file_path = os.path.join(output_path_iter, f"homography_{i}_{i+1}.npy")
                np.save(homography_file_path, homography)

                with open(img2TXT_path) as f:
                    imgwarpped_annotations = json.load(f)
                with open(img1TXT_path) as f:
                    imgfixed_annotations = json.load(f)

                transformed_annotations = []
                for annotation in imgfixed_annotations["annotations"]:
                    transformed_annotation = annotation.copy()
                    transformed_annotation["bbox"] = annotation["bbox"] if type(annotation["bbox"]) is list else eval(annotation["bbox"])
                    transformed_annotations.append(transformed_annotation)
                    
                for annotation in imgwarpped_annotations["annotations"]:
                    bbox = coco_to_bbox(annotation["bbox"] if type(annotation["bbox"]) is list else eval(annotation["bbox"]), (annotation["width"], annotation["height"]))
                    transformed_bbox = transform_bbox(bbox, homography)
                    transformed_bbox_coco = bbox_to_coco(transformed_bbox, (stitched_img.shape[1], stitched_img.shape[0]))
                    transformed_annotation = annotation.copy()
                    transformed_annotation["bbox"] = transformed_bbox_coco
                    transformed_annotation["width"] = stitched_img.shape[1]
                    transformed_annotation["height"] = stitched_img.shape[0]
                    transformed_annotations.append(transformed_annotation)
                
                stitched_img = crop_image(stitched_img)
                annotated_img = draw_annotations_on_image(np.copy(stitched_img), transformed_annotations)
            
                nms_img, nms_annotations = filter_bboxes(np.copy(stitched_img), transformed_annotations)
                nms_img = crop_image(nms_img)
                
                img_stitch_path = os.path.join(output_path_iter, str(i) + ".png")
                img_join_path = os.path.join(output_path_iter, "joined_" + str(i) + ".png")
                img_nms_path = os.path.join(output_path_iter, "nms_" + str(i) +  ".png")
                txt_nms_path = os.path.join(output_path_iter, str(i) + ".txt")
                cv2.imwrite(img_stitch_path, stitched_img)
                #cv2.imwrite(img_join_path, annotated_img)
                if iter == (num_iters-1):
                    cv2.imwrite(img_nms_path, nms_img)
                
                out_annotations = {"annotations": []}
                out_annotations["annotations"] = nms_annotations
                save_annotations(txt_nms_path, out_annotations)
        folder_path = output_path_iter


if __name__ == "__main__":
    main()





