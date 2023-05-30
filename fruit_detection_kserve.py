import numpy as np
import sys
import logging
import time
from fruit_detection_utils import get_parser, load_detic, load_xdecoder, process_mask, predict_img, pred2COCOannotations
import sys
from ultralytics import YOLO
from xdcoder_utils import refseg_single_im
import cv2
import torch
from PIL import Image

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, 'X_Decoder/')

# constants
ENCODING = 'utf-8'
WINDOW_NAME = "Detic"

def init_model(self):
    
    self.args = get_parser().parse_args()
    
    logging.info("Arguments: " + str(self.args))
    
    if torch.cuda.is_available():
        self.device = torch.device('cuda')  # Si CUDA está disponible, se usa la GPU
    
    else:
        self.device = torch.device('cpu')  # Si no, se usa la CPU
    
    # Load the model
    logging.info("Init model")
    
    
def load_model(self):
    # generate a random image to make a first inference
    numpy_image = np.random.rand(640, 640, 3) * 255
    numpy_image = numpy_image.astype(np.uint8)
    # Convert NumPy to PIL
    pil_image = Image.fromarray(numpy_image)
    
    if self.args.det_model == "Detic":
            self.model_predictor = load_detic(self.args)
            _ = self.model_predictor.run_on_image(numpy_image) 
    else:
        #self.model_predictor = YOLO("yolov8x-seg.pt")  # load an official model
        self.model_predictor = YOLO("models/best_grapebrunch_model_X_150_epochs.pt")  # load a custom model
        _ = self.model_predictor(numpy_image, imgsz=640, conf=self.args.confidence_threshold, iou=self.args.nms_max_overlap)
        
    logging.info("Detection model loaded")
    
    if self.args.full_pipeline:
        self.model_seg, self.transform, self.metadata, self.vocabulary_xdec = load_xdecoder(self.args)
        width = pil_image.size[0]
        height = pil_image.size[1]
        image = self.transform(pil_image)
        image = np.asarray(image)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
        
        batch_inputs = [{'image': images, 'height': height, 'width': width, 'groundings': {'texts': self.vocabulary_xdec}}]
        outputs = self.model_seg.model.evaluate_grounding(batch_inputs, None)
    logging.info("Segmentation model loaded")
        
    # Load classification model: clss 0 (Botrytis/Damaged) and clss 1 (Healthy)
    #self.yolo_clss_health = YOLO("yolov8l-cls.pt")  # load an official model
    self.yolo_clss_health = YOLO("models/best_health_cls_v2.pt")  # load a custom model
    _ = self.yolo_clss_health(numpy_image , imgsz=640)
    logging.info("Health model loaded")


def infer(self,img, id=0, frame_id=0):
    # Request reception
    logging.info("Infer step -------------------------------------------------------")
    
    start_time = time.time()
    logging.info("Infer over: {} {}; from user: {}; and frame_id {}".format(str(type(img)), str(img.dtype), str(id), str(frame_id)))
    logging.info("Device: {}".format(str(self.device)))
    logging.info("Original dim: {}".format(str(img.shape)))
    
    img_ori_np = img.copy() # original dimensions
    img_ori_np = cv2.cvtColor(img_ori_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_ori_np).convert("RGB")
    img_ori_np = cv2.cvtColor(img_ori_np, cv2.COLOR_RGB2BGR)
    
    logging.info("set-up time: {:.4f} seconds".format(time.time() - start_time))
    
    # Referring segmentation
    seg_start_time = time.time()
    try:
        logging.info("first pixel color check bf seg: {}".format(np.asarray(pil_image)[0][0]))
        img_grapes_crop, fruit_bbox, img_seg = refseg_single_im(pil_image, self.vocabulary_xdec,  self.transform,  self.model_seg,  self.metadata, output_root="", file_name="", save=False)
    except Exception as e:
        out_img = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
        fruit_zone = (0,0,out_img.shape[0],out_img.shape[1]) # top left down right
        fruit_bbox = fruit_zone
        img_seg= cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
        img_grapes_crop = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
        logging.info("error segmentating: {}".format(e))
    
    logging.info("Segmentation time: {:.2f} seconds".format(time.time() - seg_start_time))
    
    # Detection
    det_start_time = time.time()
    # Predict with detection model over patches or full image
    logging.info("first pixel color check bf det grapes crop: {}".format(img_grapes_crop[0][0]))
    logging.info("first pixel color check bf det img_ori: {}".format(img_ori_np[0][0]))
    img_out_bbox, img_out_mask, mask, img_health, health_flag, cont_det, pred = predict_img(img_grapes_crop, self.args, self.model_predictor, save=True, save_path="", img_o=img_ori_np, fruit_zone=fruit_bbox, health_model=self.yolo_clss_health, cont_det=0)
    logging.info("Detection time: {:.2f} seconds".format(time.time() - det_start_time))
    
    post_start_time = time.time()
    # Generate final mask --> post-process
    if self.args.det_model == "Detic":
        mask_final = process_mask(mask, save=False, save_path="", max_clusters=30)
    else:
        mask_final = mask
        
    if self.args.det_model == "Detic":
        img_out_mask = cv2.bitwise_and(img_ori_np, img_ori_np,  mask=mask_final.astype("uint8"))
        
    logging.info("Post-processing time: {:.2f} seconds".format(time.time() - post_start_time))
    
            
    # Save preds as annotations.txt
    logging.info("img_ori_np shape, color: {}, {}".format(img_ori_np.shape, img_ori_np[0][0]))
    logging.info("mask_final shape, color: {}, {}".format(mask_final.shape, mask_final[0][0]))
    logging.info("img_health shape, color: {}, {}".format(img_health.shape, img_health[0][0]))
    annotations_json = pred2COCOannotations(img_ori_np, mask_final, img_health, pred)
    
    logging.info("Total processing time: {:.2f} seconds".format(time.time() - start_time))
        
    return annotations_json, cont_det
                