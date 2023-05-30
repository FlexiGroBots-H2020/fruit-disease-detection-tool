import argparse
from detectron2.config import get_cfg
from Detic.third_party.CenterNet2.centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from xdcoder_utils import load_config_dict_to_opt, load_opt_from_config_files, load_opt_command
import json
import random
import time
import torchvision
import logging

from Detic.detic.predictor import VisualizationDemo

import os
import torch

from PIL import Image
import numpy as np
np.random.seed(27)

from torchvision import transforms

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_colors
from X_Decoder.xdecoder.BaseModel import BaseModel
from X_Decoder.xdecoder import build_model
from X_Decoder.utils.distributed import init_distributed

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from skimage.draw import polygon2mask

from detic_utils import detic_proccess_img
from yolo_utils import yolo_proccess_img

def load_detic(args):
    cfg = setup_cfg_detic(args)

    # Instance Detic Predictor
    try:
        detic_predictor = VisualizationDemo(cfg, args)
    except:
        # second time it works
        detic_predictor = VisualizationDemo(cfg, args)
        logging.warning("w: 'CustomRCNN' was already registered")
        
    return detic_predictor

def load_xdecoder(args):
    
    opt, cmdline_args= setup_cfg_xdecoder(args)
    opt = init_distributed(opt)

    vocabulary_xdec = args.vocabulary_xdec

    model = BaseModel(opt, build_model(opt)).from_pretrained(args.xdec_pretrained_pth).eval().cuda()
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background", "background"], is_eval=False)

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    metadata = MetadataCatalog.get('ade20k_panoptic_train')
    model.model.metadata = metadata
    
    return model, transform, metadata, vocabulary_xdec

def setup_cfg_detic(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file_detic)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

def setup_cfg_xdecoder(args):
    cmdline_args = args

    opt = load_opt_from_config_files(cmdline_args.config_file_xdec)

    if cmdline_args.config_overrides:
        config_overrides_string = ' '.join(cmdline_args.config_overrides)
        logging.warning(f"Command line config overrides: {config_overrides_string}")
        config_dict = json.loads(config_overrides_string)
        load_config_dict_to_opt(opt, config_dict)

    if cmdline_args.overrides:
        assert len(cmdline_args.overrides) % 2 == 0, "overrides arguments is not paired, required: key value"
        keys = [cmdline_args.overrides[idx*2] for idx in range(len(cmdline_args.overrides)//2)]
        vals = [cmdline_args.overrides[idx*2+1] for idx in range(len(cmdline_args.overrides)//2)]
        vals = [val.replace('false', '').replace('False','') if len(val.replace(' ', '')) == 5 else val for val in vals]

        types = []
        for key in keys:
            key = key.split('.')
            ele = opt.copy()
            while len(key) > 0:
                ele = ele[key.pop(0)]
            types.append(type(ele))
        
        config_dict = {x:z(y) for x,y,z in zip(keys, vals, types)}
        load_config_dict_to_opt(opt, config_dict)

    # combine cmdline_args into opt dictionary
    for key, val in cmdline_args.__dict__.items():
        if val is not None:
            opt[key] = val

    return opt, cmdline_args


def get_parser():
    
    parser = argparse.ArgumentParser(description="Xdcoder and Detectron2 setup for builtin configs")
    
    # COMMON
    parser.add_argument(
        "--input",
        default="",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg' or a path to a video",
    )
    parser.add_argument(
        "--output",
        default="output/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    
    parser.add_argument("--debug", default=False, action='store_true', help="Bool indicating if debug")
    parser.add_argument("--print_health", default=False, action='store_true', help="Bool indicating if print health over image")
    parser.add_argument("--full_pipeline", default=True, action='store_true', help="Bool indicating if use full pipeline approach")
    
    # DETIC SETUP    
    parser.add_argument(
        "--config-file-detic",
        default="Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")

    
    parser.add_argument(
        "--vocabulary",
        default="custom",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="grape, botrytis_grape",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS",
                "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument('--nms_max_overlap', type=float, default=0.7,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    
    
    parser.add_argument("--patching", default=True, action='store_true', help="To proccess the image in patches.")
    parser.add_argument("--patch_size", default=640, type=int, help="Patch of the patches.")
    parser.add_argument("--overlap", default=0.2, type=float, help="Overlap of the patches")
    parser.add_argument("--seg_step", default=True, action='store_true', help="To proccess the image in patches.")
    
    # XDECODER SETUP 
    
    parser.add_argument('--command', default="evaluate", help='Command: train/evaluate/train-and-evaluate')
    parser.add_argument('--config_file_xdec', default=['X_Decoder/configs/xdecoder/svlp_focalt_lang.yaml'], nargs='+', help='Path(s) to the config file(s).')
    parser.add_argument('--user_dir', help='Path to the user defined module for tasks (models, criteria), optimizers, and lr schedulers.')
    parser.add_argument('--config_overrides', nargs='*', help='Override parameters on config with a json style string, e.g. {"<PARAM_NAME_1>": <PARAM_VALUE_1>, "<PARAM_GROUP_2>.<PARAM_SUBGROUP_2>.<PARAM_2>": <PARAM_VALUE_2>}. A key with "." updates the object in the corresponding nested dict. Remember to escape " in command line.')
    parser.add_argument('--overrides', help='arguments that used to override the config file in cmdline', nargs=argparse.REMAINDER)
    parser.add_argument('--xdec_pretrained_pth', default='X_Decoder/models/xdecoder_focalt_last.pt', help='Path(s) to the weight file(s).')
    parser.add_argument('--xdec_img_size', type=int, default=512 ,help='reshape size for the image to be proccessed wit x-decoder')
    parser.add_argument('--vocabulary_xdec', nargs='+', default=['closer vineyard'], help='Concepts to segmentate')
    parser.add_argument('--det_model', default="YOLO", help='Select the model use for detection: Detic or YOLO')
    
    return parser


def non_max_suppression_compose(prediction, iou_thres=0.45):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Returns:
         Dict of predictions with the not overlapped instances 
    """
    bbxs_out, confs_out, clss_out, masks_out = [], [], [], []
    bboxes, confs, clss, masks = prediction #bboxes in xyxy
    masks_out = masks
    
    idx = torchvision.ops.nms(bboxes, confs, iou_thres)  # NMS
    idx_np=idx.cpu().numpy()
    for idx in idx_np:
        bbxs_out.append(bboxes[idx]) 
        confs_out.append(confs[idx])
        clss_out.append(clss[idx]) 
    keep_predictions = bbxs_out, confs_out, clss_out, masks_out
    return keep_predictions


def kmeans(mask, max_clusters):
    points_array = np.argwhere(mask == 1)
    
    # Si no se especifica el número máximo de clusters, usar un valor por defecto
    if max_clusters is None:
        max_clusters = 10
    
    # Calcular el número óptimo de clusters para K-means.
    points_array = np.float32(points_array)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    dists = []
    for k in range(1, max_clusters):
        compactness, labels, centers = cv2.kmeans(points_array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dists.append(compactness)
    
    # Create figure and axes
    fig_kopt, ax = plt.subplots()
    # Generate 2D plot
    ax.plot(dists)
    # Configure axes
    ax.set_xlabel('N clusters')
    ax.set_ylabel('Value')
    ax.set_title('kmeans values')
    
    elbow_2 = np.gradient(np.gradient(dists))
    elbow_1 = np.gradient(dists)
    ax.plot(elbow_1)
    ax.plot(elbow_2)
    optimal_k = elbow_2.argmax() + 1
    print("num_cluster: ", str(optimal_k))

    # Aplicar K-means en la imagen para encontrar clusters de objetos segmentados.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(points_array, optimal_k, None, criteria, 10, flags)

    # Asignar a cada pixel el valor correspondiente al centroide más cercano.
    centers = np.uint8(centers)
    labels = labels.flatten()
    img_clusters = np.zeros((mask.shape[0], mask.shape[1], 3))
    colors = random_colors(k)
    
    for ii in range(points_array.shape[0]):
        img_clusters[int(points_array[ii][0])][int(points_array[ii][1])] = colors[labels[ii]]
        
    return img_clusters, k, fig_kopt


def process_mask(mask, save=False, save_path=None, max_clusters=20):
    # Apply dilation to close gaps between segmented instances.
    kernel_op = np.ones((10, 10),np.uint8)
    kernel_b = np.ones((5, 5),np.uint8)
    
    # Apply opening to reduce noise and smooth the edges of segmented instances.
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_op)
    
    dilated_mask = cv2.dilate(closed_mask, kernel_b, iterations=3)
    
    # Apply clossing to avoid gaps.
    opened_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel_op)
    
    # Erode to quit extra margins
    proccessed_mask = cv2.erode(opened_mask, kernel_b, iterations=3)
    
    #kmeans_centroids_img, num_c, fig_optK = kmeans(proccessed_mask, max_clusters)
    
    if save:
        #cv2.imwrite(os.path.join(save_path,"dilated_mask.png"), dilated_mask*255)
        cv2.imwrite(os.path.join(save_path,"closed_mask.png"), closed_mask*255)
        cv2.imwrite(os.path.join(save_path,"opened_mask.png"), opened_mask*255)
        #cv2.imwrite(os.path.join(save_path,"kmeans.png"), kmeans_centroids_img)
        # Save plot to disk
        #fig_optK.savefig(os.path.join(save_path,"k_values.png"))
    
    num_clusters = count_contiguous_groups(proccessed_mask)
    print("num bunches aprox: " + str(num_clusters))
    
    return proccessed_mask

def count_contiguous_groups(mask):
    # Convert the mask to boolean type (if it is not already)
    mask = np.array(mask, dtype=bool)

    # Get the contiguous groups of pixels with value 1 in the mask
    groups, count = label(mask)

    # Return the number of contiguous groups found
    return count


def mask_metrics(mask_true, mask_pred):
    """Compute the Dice coefficient, Jaccard index, and Hausdorff distance for two binary masks."""
    
    # Ensure that the input masks have the same size
    if mask_pred.shape != mask_true.shape:
        raise ValueError('Input masks must have the same size.')
    
    # Compute the Dice coefficient and Jaccard index
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)
    iou = intersection.sum() / union.sum()
    dice = 2 * intersection.sum() / (mask_true.sum() + mask_pred.sum())
    jaccard = intersection.sum() / (mask_true.sum() + mask_pred.sum() - intersection.sum())
    
    # Compute the Hausdorff distance
    distances_true = distance_transform_edt(mask_true)
    distances_pred = distance_transform_edt(mask_pred)
    hausdorff = np.max([np.percentile(distances_true[distances_pred > 0], 95),
                        np.percentile(distances_pred[distances_true > 0], 95)])
    
    # Return the metrics
    return iou, dice, jaccard, hausdorff


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_detections(img, prediction, clss_name):
    img_draw = np.copy(img).astype("float")
    bboxes, confs, clss, masks = prediction #bboxes in xyxy
    for ii, bbox in enumerate(bboxes):
        box = bbox.cpu().numpy().astype('int')
        label_box =  str(clss_name[int(clss[ii].cpu().numpy())]) + " " + str(np.round(confs[ii].cpu().numpy(),2))
        img_draw = plot_one_box(box,img_draw,label=label_box)
    return img_draw


def lists2img(img, predictions_nms, classes_names, fruit_zone):
    bboxes, confs, clss, masks = predictions_nms
    # Draw detections over image and save it 
    out_img = draw_detections(img, predictions_nms, classes_names)
    
    # Generate masks
    out_img_masked, total_mask = generate_final_mask(masks, img, fruit_zone)
    
    return out_img, out_img_masked, total_mask

def bbox_to_coco(bbox, img_size):
    x_min, y_min, x_max, y_max = bbox
    img_width, img_height = img_size

    # Convert to relative coords
    x_min_rel = x_min / img_width
    y_min_rel = y_min / img_height
    x_max_rel = x_max / img_width
    y_max_rel = y_max / img_height

    return [x_min_rel, y_min_rel, x_max_rel, y_max_rel]


def pred2COCOannotations(img, mask_final, img_health, predictions_nms, out_folder="", file_name=""):
    bboxes, confs, clss, masks = predictions_nms
    height, width, _ = img.shape
    
    txt_path_file = os.path.join(out_folder, file_name +".txt")

    coco_annotations = {
        "info": {
            "description": "COCO-style annotations",
            "version": "1.0"
        },
        "images": [
            {
                "id": out_folder,
                "width": width,
                "height": height,
                }
        ],
        "detections": [],
        "segmentations": [],
    }

    for i, (bbox, conf, cls, msk) in enumerate(zip(bboxes, confs, clss, masks)):
        h, w, _ = img.shape
        coco_bbox = bbox_to_coco(np.asarray(bbox), (w,h))
        annotation = {
            "id": i + 1,
            "image_id": out_folder + file_name,
            "width": width,
            "height": height,
            "category_id": int(np.asarray(cls)),
            "bbox": str(coco_bbox),
            "score": float(np.asarray(conf))
        }
        coco_annotations["detections"].append(annotation)
        
    # Generate segmentation annotation
    seg_annotations = mask_to_coco_segmentation(mask_final, img_health)
    coco_annotations["segmentations"].append(seg_annotations)

    if out_folder != "":
        with open(txt_path_file, "w") as outfile:
            annotations_json = json.dump(coco_annotations, outfile, indent=2)
    else:
        annotations_json = json.dumps(coco_annotations)
    return annotations_json


def generate_final_mask(masks, img, fruit_zone=(0,0,0,0)):
    total_mask = np.zeros((img.shape[0], img.shape[1]))
    if fruit_zone == (0,0,0,0):
        fruit_zone = (0, 0, img.shape[0], img.shape[1])

    for jj in range(len(masks)):
        total_mask[fruit_zone[0]:fruit_zone[2], fruit_zone[1]:fruit_zone[3]] = total_mask[fruit_zone[0]:fruit_zone[2], fruit_zone[1]:fruit_zone[3]] + masks[jj].cpu().numpy().astype(int)[0:(fruit_zone[2]-fruit_zone[0]), 0:(fruit_zone[3]-fruit_zone[1])]
        #total_mask = total_mask + masks[jj].cpu().numpy().astype(int)[0:total_mask.shape[0], 0:total_mask.shape[1]]
    if np.max(total_mask)>1:
        total_mask[total_mask > 1] = 1
        
    out_img_masked = cv2.bitwise_and(img, img,  mask=total_mask.astype("uint8"))
    return out_img_masked, total_mask


def zoom_on_non_zero(image, black_threshold=0.8):
    # Convert the image to grayscale to facilitate non-zero pixel detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the percentage of black pixels
    total_pixels = gray.size
    black_pixels = np.count_nonzero(gray == 0)
    black_ratio = black_pixels / total_pixels

    if black_ratio > black_threshold:
        # Find the indices of the non-zero elements
        rows, cols = np.nonzero(gray)

        # Find the minimum and maximum coordinates of the non-zero rows and columns
        row_min, row_max = np.min(rows), np.max(rows)
        col_min, col_max = np.min(cols), np.max(cols)

        # Calculate the region of interest (ROI) using the minimum and maximum coordinates
        roi = image[row_min:row_max, col_min:col_max]

        # Calculate the desired non-black pixel percentage in the output image
        target_non_black_ratio = 1 - black_threshold

        # Calculate the scaling factor needed to achieve the desired non-black pixel percentage
        input_non_black_pixels = total_pixels - black_pixels
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_non_black_pixels = np.count_nonzero(roi_gray)
        scale_factor = np.sqrt(input_non_black_pixels / roi_non_black_pixels / target_non_black_ratio)

        # Resize the output image to have a non-black pixel percentage equal to the black_threshold value
        resized_roi = cv2.resize(roi, (int(roi.shape[1] * scale_factor), int(roi.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
        resized_roi = resize_image(resized_roi, max_dim=max(image.shape))
        # Create a black image of the same size as the original image
        zoomed_img = np.zeros_like(image)

        # Calculate the coordinates where the resized image will be placed in the output image
        start_row = (image.shape[0] - resized_roi.shape[0]) // 2
        start_col = (image.shape[1] - resized_roi.shape[1]) // 2

        # Place the resized image in the output image
        zoomed_img[start_row:start_row + resized_roi.shape[0], start_col:start_col + resized_roi.shape[1]] = resized_roi

        return zoomed_img
    else:
        return image


def patch_image(image, patch_size, overlap):
    image_tiles = []
    step = round(patch_size*(1-overlap))
    h, w, n_channels = image.shape
    if h == patch_size:
        step_h = patch_size
    else:
        step_h = step
    if w == patch_size:
        step_w = patch_size
    else:
        step_w = step
    
    for y in range(0, image.shape[0], step_h):
        for x in range(0, image.shape[1], step_w):
            image_tile = image[y:y + patch_size, x:x + patch_size]
            if image_tile.shape != (patch_size,patch_size, n_channels):
                image_tile= cv2.copyMakeBorder(image_tile,0,int(patch_size-image_tile.shape[0]), 0, int(patch_size-image_tile.shape[1]),cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_tiles.append(image_tile)
    return image_tiles
            

def im2patches(img, patch_size=640, overlap=0.2):
    h,w,channels= img.shape
    step = round(patch_size*(1-overlap))
    n_h, n_w, _ = np.ceil(np.array(img.shape) / step).astype(int) * step
    
    if h < patch_size:
        n_h = patch_size
    if w < patch_size:
        n_w = patch_size
    
    img_border= cv2.copyMakeBorder(img,0,int(n_h-h), 0, int(n_w-w),cv2.BORDER_CONSTANT, value=[0, 0, 0]) 
    patches = patch_image(img_border, patch_size, overlap)
    empty_mask = np.zeros((img_border.shape[0], img_border.shape[1]))
    patches_np = np.reshape(patches, (int(n_h/step), int(n_w/step), patch_size, patch_size, channels))
    return patches_np, empty_mask


def check_health(health_model, img_patch, masks_p, health_thres, save=False, save_path="", patch_id=""):
    health_mask = np.zeros(img_patch.shape)
    health_flag = False
    for ii,msk in enumerate(masks_p):
        msk_np = msk.cpu().numpy().astype("uint8")
       
        if msk_np.shape != img_patch.shape:
            msk_np = cv2.resize(msk_np, (640,640)) # fix image to model input dimensions 
        img_crop_health = cv2.bitwise_and(img_patch, img_patch, mask=msk_np)    
        img_health_in = cv2.resize(img_crop_health, (640,640)) # fix image to model input dimensions
        img_zoomed = zoom_on_non_zero(img_health_in, black_threshold=0.6) 
        results = health_model(img_zoomed, imgsz=640)
        disease_score = float(results[0].probs[0].cpu().detach().numpy())
        healthy_score =float(results[0].probs[1].cpu().detach().numpy())
        if disease_score > health_thres:
            health_flag = True
            health_msg_d = "disease detected, conf: {:.2f}".format(disease_score)
            color = (0,0,255)
            color_mask = cv2.cvtColor(msk_np, cv2.COLOR_GRAY2RGB) * color
            health_mask = health_mask + color_mask
        else:
            health_msg_h ="healthy, conf: {:.2f}".format(healthy_score)
            color = (255,0,0)
            color_mask = cv2.cvtColor(msk_np, cv2.COLOR_GRAY2RGB) * color
            health_mask = health_mask + color_mask
        if save:
            img_save_path = os.path.join(save_path,"in_health_{}_{}.jpg".format(patch_id,ii))
            txt_position = (10, img_zoomed.shape[0]-10)
            if health_flag:
                health_msg = health_msg_d
            else: 
                health_msg = health_msg_h
            img_health_single_bunch = cv2.putText(np.copy(img_zoomed), health_msg, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  
            cv2.imwrite(img_save_path, img_health_single_bunch)
        
    if health_flag:
        health_msg = health_msg_d
    else: 
        health_msg = health_msg_h
    txt_position = (10, health_mask.shape[0]-10)
    img_health = cv2.putText(np.copy(health_mask), health_msg, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 
    
    return img_health, health_mask, health_msg, health_flag
  
  
def patchmask2imgmask(mask_agg, mask, row, column, patch_size, overlap, fruit_zone=(0,0,0,0)):
    # from patches mask to full dimension mask
    mask_agg_patch = mask_agg[round(row*patch_size*(1 - overlap)):round(row*patch_size*(1 - overlap) + patch_size), round(column*patch_size*(1 - overlap)):round(column*patch_size*(1 - overlap) + patch_size)]
    window_shape = mask_agg_patch.shape
    if mask_agg_patch.shape != (patch_size,patch_size,3):
            mask_agg_patch = cv2.copyMakeBorder(mask_agg_patch,0,int(patch_size-mask_agg_patch.shape[0]), 0, int(patch_size-mask_agg_patch.shape[1]),cv2.BORDER_CONSTANT, value=[0, 0, 0])
    mask_agg_patch = mask_agg_patch + mask
    if mask_agg_patch.shape != window_shape:
            mask_agg_patch = mask_agg_patch[0:window_shape[0], 0:window_shape[1]]
    mask_agg[(round(row*patch_size*(1 - overlap))):(round(row*patch_size*(1 - overlap) + patch_size)), (round(column*patch_size*(1 - overlap))):(round(column*patch_size*(1 - overlap) + patch_size))] = mask_agg_patch
    if np.max(mask_agg)>255:
        mask_agg[mask_agg > 255] = 255

    return mask_agg

def predict_img(img_p, args, model_predictor, save=True, save_path="", path="", fruit_zone=(0,0,0,0), img_o=None, health_model=None, health_thres=0.5, cont_det=0):
    health_flag = False # False indicates that there is not disease
    # If image to be proccessed is a patch of the original both have to be pass as input
    if img_o is None:
        img_o=img_p
    
    # Predict image patches   
    if args.patching:
        patch_size = args.patch_size
        overlap = args.overlap
        patches, empty_mask = im2patches(img_p, patch_size, overlap)
        n_row, n_col, _, _, _ = patches.shape
        logging.info("{} patches: {} rows and {} columns".format(str(n_row*n_col), str(n_row), str(n_col)))
        bboxs_t, confs_t, clss_t, masks_t = ([] for i in range(4))
        health_mask_agg = np.zeros((empty_mask.shape[0], empty_mask.shape[1], 3))
        for ii in range(n_row):
            for jj in range(n_col):
                # Detect over each patch
                start_time = time.time()
                img_patch = patches[ii][jj]
                #simg_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
                if args.det_model == "Detic":
                    detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = detic_proccess_img(img_patch, model_predictor, args, fruit_zone, empty_mask, (ii, jj))
                    bboxs_p, confs_p, clss_p, masks_p, mask_p_agg = detections_p
                else:
                    detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = yolo_proccess_img(img_patch, model_predictor, args, fruit_zone, empty_mask, (ii, jj))
                    bboxs_p, confs_p, clss_p, masks_p, mask_p_agg = detections_p
                        
                bboxs_t = bboxs_t + bboxs_p.tolist()
                confs_t = confs_t + confs_p.tolist()
                clss_t = clss_t + clss_p.tolist()
                masks_t.append(mask_p_agg) 
                
                logging.info("{} {}: {} in {:.2f}s".format(path, "patch " + str(ii) + " " + str(jj) , pred_str, time.time() - start_time)) 
                
                # Search diseases over images if there are detections and model to do it
                if health_model is not None and np.max(img_out_mask) > 0:
                    img_health, health_mask, health_msg, health_flag = check_health(health_model, img_patch, masks_p, health_thres, save, save_path, "patch_{}_{}".format(ii,jj))
                    logging.info("Patch {} {}: {}".format(ii,jj,health_msg))
                    health_mask_agg = patchmask2imgmask(health_mask_agg, health_mask, ii, jj, patch_size, overlap, fruit_zone)
                else:
                    health_mask = np.zeros(img_patch.shape)
                    health_msg = "Unknown health status" 
                    txt_position = (10, health_mask.shape[0]-10)
                    img_health = cv2.putText(np.copy(health_mask), health_msg, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)                          
                
                # Save results   
                if save:
                    if save_path != "":
                        if os.path.isdir(save_path):
                            assert os.path.isdir(save_path), save_path
                            out_filename = os.path.join(save_path, os.path.basename(path))
                        else:
                            assert len(args.input) == 1, "Please specify a directory with save_path"
                            out_filename = save_path
                    else:
                        out_filename=""
                        
                    if args.print_health:                  
                        cv2.imwrite(out_filename + "_p_" + str(ii) + str(jj) +"_health.jpg", img_health)
                    
                    cv2.imwrite(out_filename + "_p_" + str(ii) + str(jj) +".jpg", img_patch)
                    cv2.imwrite(out_filename + "_p_" + str(ii) + str(jj) +"_vis.jpg", img_out_bbox)
                    cv2.imwrite(out_filename + "_p_" + str(ii) + str(jj) +"_mask.jpg", img_out_mask)
                    
        # Join all the predictions over patches
        pred_compose = torch.FloatTensor(np.array(bboxs_t)), torch.FloatTensor(np.array(confs_t)), torch.FloatTensor(np.array(clss_t)), torch.FloatTensor(np.array(masks_t))
        if len(pred_compose[0]):
            pred_compose_nms = non_max_suppression_compose(pred_compose, iou_thres=args.nms_max_overlap)
        else:
            pred_compose_nms = pred_compose
            
        cont_det += len(pred_compose_nms[0])
            
        # Load classes names
        if args.det_model == "Detic":
            classes_names = model_predictor.metadata.thing_classes
        else:
            classes_names = []
            for ii in range(len(model_predictor.names)):
                classes_names.append(model_predictor.names[ii])
        
        # Draw output img     
        img_out_bbox, img_out_mask, mask_tot = lists2img(np.asarray(img_o), pred_compose_nms, classes_names, fruit_zone)
        
        # Print health status over img
        health_mask_agg_final = np.zeros(img_o.shape)
        health_mask_agg_p = health_mask_agg[0:img_o.shape[0], 0:img_o.shape[1]]
        health_mask_agg_final[fruit_zone[0]:fruit_zone[2],fruit_zone[1]:fruit_zone[3]] = health_mask_agg_p[0:(fruit_zone[2]-fruit_zone[0]), 0:(fruit_zone[3]-fruit_zone[1])]
        if args.print_health and (health_model is not None):
            health_msg = "Disease detected: {}".format(str(health_flag))
            txt_position = (10, health_mask_agg_final.shape[0]-10)
            img_health = cv2.putText(np.copy(health_mask_agg_final), health_msg, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
        elif (health_model is not None):
            img_health = np.copy(health_mask_agg_final)

        # Save results
        if save:
            if save_path:
                if os.path.isdir(save_path):
                    assert os.path.isdir(save_path), save_path
                    out_filename = os.path.join(save_path, os.path.basename(path))
                else:
                    assert len(save_path) == 1, "Please specify a directory with args.output"
                    out_filename = save_path
            else:
                out_filename=""
                
            if args.print_health and (health_model is not None):
                cv2.imwrite(out_filename +"_health.jpg", img_health)
            cv2.imwrite(out_filename +"_vis.jpg", img_out_bbox)
            cv2.imwrite(out_filename +"_mask.jpg", img_out_mask)    
    
    # Predict full image        
    else:
        start_time = time.time()
        empty_mask = np.zeros((img_p.shape[0], img_p.shape[1]))
        if args.det_model == "Detic":
            detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = detic_proccess_img(img_p, model_predictor, args, fruit_zone, empty_mask)
        else:
            detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = yolo_proccess_img(img_p, model_predictor, args, fruit_zone, empty_mask)
                    
        cont_det += len(detections_p[0])
        logging.info("{}: {} in {:.2f}s".format(path, pred_str, time.time() - start_time)) 
        
        # Search diseases over images
        pred_compose_nms = detections_p
        _, _, _, masks, mask_agg = detections_p
        pred_health_as_block = True # select if want to predict health over each detection individually or only once fro the full mask
        if pred_health_as_block:
            masks = torch.from_numpy(np.expand_dims(mask_agg,0))
        
        if health_model is not None and np.max(img_out_mask) > 0:
            img_health_in = cv2.resize(img_out_mask, (640,640)) # fix img to model input dimensions
            img_health, health_mask, health_msg, health_flag = check_health(health_model, img_health_in, masks, health_thres)
            img_health = cv2.resize(img_health, (img_o.shape[1],img_o.shape[0])) # fix img to original dimensions
            logging.info("Im: {}".format(health_msg))
        else:
            health_mask = np.zeros(img_p.shape)
            health_msg = "Unknown health status" 
            txt_position = (10, health_mask.shape[0]-10)
            img_health = cv2.putText(np.copy(health_mask), health_msg, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)                          
                
        # Save results
        if save:
            if save_path:
                if os.path.isdir(save_path):
                    assert os.path.isdir(save_path),save_path
                    out_filename = os.path.join(save_path, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with save_path"
                    out_filename = save_path
            else:
                out_filename=""
                
            if args.print_health and (health_model is not None):
                cv2.imwrite(out_filename + "_health.jpg", img_health)
            
            cv2.imwrite(out_filename + "_vis.jpg", img_out_bbox)
            cv2.imwrite(out_filename + "_mask.jpg", img_out_mask)  
            
    return img_out_bbox, img_out_mask, mask_tot, img_health, health_flag, cont_det, pred_compose_nms


import numpy as np
import pycocotools.mask as mask_util
from scipy import ndimage

def mask_to_coco_segmentation(mask, health_mask, small_object_threshold=0.15):
    """Convert a binary segmentation mask to COCO 'segmentation' annotation format, excluding small blobs.

    Args:
        mask (ndarray): a 2D Numpy array of shape (H, W), where H is the height and W is the width of the image.
                        Each pixel is either 0 (background) or 1 (object).
        small_object_threshold (float): if the size of a blob is less than this fraction of the average blob size, it is removed.

    Returns:
        segmentation_data (dict): a dictionary containing the converted data in COCO format.
    """
    
    # Make sure the mask is in the correct format
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a 2D Numpy array")

    # Create a copy of the mask
    mask_copy = mask.copy()

    # Label different blobs in the mask
    labeled_mask, num_labels = ndimage.label(mask_copy)

    # Compute the size of each blob
    blob_sizes = ndimage.sum(mask_copy, labeled_mask, range(num_labels + 1))

    # Compute the average blob size
    average_blob_size = blob_sizes.mean()

    # Create a mask for small blobs
    small_blobs = np.isin(labeled_mask, np.where(blob_sizes < small_object_threshold * average_blob_size))

    # Remove small blobs
    mask_copy[small_blobs] = 0

    # Asegúrate de que la máscara esté en uint8
    mask_copy = mask_copy.astype(np.uint8)
    kernel = np.ones((5,5),np.uint8) # Need to NOT have intering contours
    mask_copy = cv2.morphologyEx(mask_copy, cv2.MORPH_CLOSE, kernel)

    # Encuentra contornos en la máscara
    contours = measure.find_contours(mask_copy, 0.5)

    # Inicializa la lista de polígonos
    polygons = []

    # Inicializa el área total
    area = 0

    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        # Para manejar casos donde después de la simplificación nos quedamos con un polígono multiparte ("MultiPolygon")
        if poly.geom_type == 'MultiPolygon':
            # Unimos todos los polígonos en uno solo
            allparts = [p.buffer(0) for p in poly]
            poly = unary_union(allparts)

        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.coords.xy
            poly_points = [(x[i], y[i]) for i in range(len(x))]
            polygons.append(poly_points)
            area += poly.area
            
    # Compute bbox
    health_status = []
    for polygon in polygons:
        health_status.append(check_polygon_color(polygon, health_mask))

    segmentation_data = {
        "segmentation": polygons,
        "health_status": health_status

    }

    return segmentation_data


def check_polygon_color(polygon, color_mask):
    """Check the color of the pixels within a polygon in a color segmentation mask.

    Args:
        polygon (list): a list of (x, y) pairs defining the vertices of the polygon.
        color_mask (ndarray): a 3D Numpy array of shape (H, W, 3), where H is the height and W is the width of the image,
                              and the third dimension represents the color channels (in RGB order).

    Returns:
        label (int): 0 if all non-black pixels within the polygon are blue in the color mask, 1 otherwise.
    """

    # Convert polygon coordinates to a 2D array (if not already)
    if isinstance(polygon[0], tuple):
        polygon = np.array(polygon)

    # The polygon coordinates should be in (row, col) format (y, x)
    polygon = np.fliplr(polygon)

    # Create a binary mask with the same shape as the color mask
    poly_mask = polygon2mask(color_mask.shape[:2], polygon)

    # Now, use this mask to get the pixels within the polygon from the color mask
    poly_pixels = color_mask[poly_mask]

    # Remove black pixels (consider them as background)
    non_black_pixels = poly_pixels[~np.all(poly_pixels == [0, 0, 0], axis=-1)]

    # Check if all these non-black pixels are blue
    if np.all(non_black_pixels == [255, 0, 0]):
        return 0
    else:
        return 1


def resize_image(img, max_dim=640):

    # Get the current dimensions of the image
    height, width = img.shape[:2]

    # Calculate the ratio
    if max(height, width) == height:
        r = max_dim / height
        dim = (int(width * r), max_dim)
    else:
        r = max_dim / width
        dim = (max_dim, int(height * r))

    # Perform the resizing
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized