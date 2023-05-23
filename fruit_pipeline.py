import multiprocessing as mp

from fruit_detection_utils import get_parser, load_detic, load_xdecoder, process_mask, mask_metrics, predict_img, pred2COCOannotations, mask_to_coco_segmentation

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, 'X_Decoder/')

from detectron2.utils.logger import setup_logger
from xdcoder_utils import refseg_single_im
from PIL import Image
import cv2
import os
import numpy as np
import glob
import json
import time

from ultralytics import YOLO

# constants
WINDOW_NAME = "Detic"

if __name__ == "__main__":
    
    gt_data = False
    
    # Set-up models and variables
    setup_logger(name="fvcore")
    logger = setup_logger()
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    
    logger.info("Arguments: " + str(args))
    
    if args.det_model == "Detic":
        model_predictor = load_detic(args)
    else:
        model_predictor = YOLO("yolov8x-seg.pt")  # load an official model
        model_predictor = YOLO("models/best_grapebrunch_model_X_150_epochs.pt")  # load a custom model
    
    if args.full_pipeline:
        model, transform, metadata, vocabulary_xdec = load_xdecoder(args)
    
    list_images_paths = [] 
    for input in args.input:
        list_images_paths = list_images_paths + glob.glob(input)
        
            
    # Generate experiment folder 
    list_existing_exp = glob.glob(os.path.join(args.output, "exp*"))
    exist_exp_idx = np.zeros(len(list_existing_exp),dtype=int)
    for ii in range(len(list_existing_exp)):
        exist_exp_idx[ii] = int(list_existing_exp[ii].split("exp")[1])
    for jj in range(len(list_existing_exp)+1):
        if jj not in exist_exp_idx:
            exp_name= "exp" + str(jj)
    exp_folder = os.path.join(args.output, exp_name)
    if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
    
    # Generate experiment json file
    variables = {
        'path': exp_folder,
        'patching': args.patching,
        'patch_size': args.patch_size,
        'overlap': args.overlap, 
        'vocabulary_xdec': args.vocabulary_xdec,
        'vocabulary_detic': args.custom_vocabulary,
        'conf_threshold': args.confidence_threshold,
        'detection_model' : args.det_model, 
        'full_pipeline' : args.full_pipeline,
        'segmentation_step': args.seg_step
    }
    json_path = os.path.join(exp_folder,'variables.json')
    with open(json_path, 'w') as f:
        f.write(json.dumps(variables))
        
    # Load classification model: clss 0 (Botrytis/Damaged) and clss 1 (Healthy)
    yolo_clss_health = YOLO("yolov8l-cls.pt")  # load an official model
    yolo_clss_health = YOLO("models/best_health_cls_v2.pt")  # load a custom model
    
    metrics = []    
    cont_healthy = 0
    cont_unhealthy = 0
    cont_det = 0
    # Process each input image
    for img_path in list_images_paths:
        start_time = time.time()
        # Set the paths to the images/outputs and GT data
        gt_path = os.path.join(img_path.split("/images")[0], "instances")
        file_name = img_path.split('/')[-1]
        base_name = file_name.split('.')[-2]
        if os.path.exists(gt_path): 
            row_id = img_path.split("/")[-4] # keep dataq format
        else: 
            row_id = "img"
          
        output_folder = os.path.join(exp_folder, row_id + "_" + base_name) 
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Load img in PILL and CV2
        img = Image.open(img_path).convert("RGB")
        img_ori_np = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        
        # Load GT data if exists
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(os.path.join(gt_path,file_name),cv2.IMREAD_GRAYSCALE)
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"gtMask_" + file_name), gt_mask*255)
            gt_data = True
        else:
            gt_data = False
        
        if args.full_pipeline:
            seg_start_time = time.time()
            if args.seg_step:
                try:
                    img_grapes_crop, fruit_bbox, img_seg = refseg_single_im(img, vocabulary_xdec, transform, model, metadata, output_folder, base_name, save=False)
                except:
                    out_img = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                    fruit_zone = (0,0,out_img.shape[0],out_img.shape[1]) # top left down right
                    fruit_bbox = fruit_zone
                    img_seg= cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                    img_grapes_crop = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                    print("error segmentating")
                if args.debug:
                    cv2.imwrite(os.path.join(output_folder,"crop_" + file_name), img_grapes_crop)
                    cv2.imwrite(os.path.join(output_folder,"seg_" + file_name), img_seg)
                logger.info("Segmentation time: {:.2f} seconds".format(time.time() - seg_start_time))
            else:
                out_img = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                fruit_zone = (0,0,out_img.shape[0],out_img.shape[1]) # top left down right
                fruit_bbox = fruit_zone
                img_seg= cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                img_grapes_crop = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                
            det_start_time = time.time()
            # Predict with detection model over patches or full image
            img_out_bbox, img_out_mask, mask, img_health, health_flag, cont_det, pred = predict_img(img_grapes_crop, args, model_predictor, save=False, save_path=output_folder, img_o=img_ori_np, fruit_zone=fruit_bbox, health_model=yolo_clss_health, cont_det=cont_det)
            logger.info("Detection time: {:.2f} seconds".format(time.time() - det_start_time))
            
            post_start_time = time.time()
            # Update health conts
            if health_flag == True:
                cont_unhealthy+=1
            else:
                cont_healthy+=1
                
            # Generate final mask --> post-process
            if args.det_model == "Detic":
                mask_final = process_mask(mask, save=False, save_path=output_folder, max_clusters=30)
            else:
                mask_final = mask
                
            logger.info("Post-processing time: {:.2f} seconds".format(time.time() - post_start_time))
            
            
            # Compare results with GT data if exists
            if gt_data and cv2.countNonZero(mask_final):
                iou, dice, jaccard, hausdorff = mask_metrics(gt_mask, mask_final)
                logger.info("IoU:" + str(iou) + " F1-Dice:" + str(dice) + " Jacc:" + str(jaccard) + " Haus:" + str(hausdorff))
                metrics.append([{"iou":iou}, {"Dice": dice}, {"Jaccard": jaccard}, {"Haus": hausdorff}, {"Time": (time.time()-start_time)}])
            
            # Save output images
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"final_" + file_name), img_out_bbox)
                cv2.imwrite(os.path.join(output_folder,"finalMask_" + file_name), img_out_mask)
                if args.print_health:
                     cv2.imwrite(os.path.join(output_folder,"health_" + file_name), img_health)
                if args.det_model == "Detic":
                    out_img_masked = cv2.bitwise_and(img_ori_np, img_ori_np,  mask=mask_final.astype("uint8"))
                    cv2.imwrite(os.path.join(output_folder,"finalMaskProccessed_" + file_name), out_img_masked)
                
                # EXTRA CODE TO SAVE MASKS AS DATASET
                output_folder.split("/")[:-1][0]
                extra_folder = os.path.join(output_folder.split("/")[:-1][0], output_folder.split("/")[:-1][1], "mask_data")
                if not os.path.exists(extra_folder):
                    os.makedirs(extra_folder)
                if np.max(img_out_mask) > 0:
                    cv2.imwrite(os.path.join(extra_folder, file_name), img_out_mask)
                    
                # Generate metrics json file
                json_path = os.path.join(extra_folder,'stats.json')
                with open(json_path, 'w+') as f:
                    f.write("Healthy {}/{} and Unhealthy{}/{}\n".format(cont_healthy,cont_healthy+cont_unhealthy,cont_unhealthy,cont_healthy+cont_unhealthy))
                    f.write("Num detections {}\n".format(cont_det))
                    f.write("Time: {}\n".format((time.time()- seg_start_time)))
                
        else:
            # Predict with detection model over patches or full image
            det_start_time = time.time()
            img_out_bbox, img_out_mask, mask, img_health, health_flag, cont_det, pred = predict_img(img_ori_np, args, model_predictor, save=False, save_path=output_folder, health_model=yolo_clss_health, cont_det=cont_det)
            logger.info("Detection time: {:.2f} seconds".format(time.time() - det_start_time))
            
            post_start_time = time.time()
            if health_flag == True:
                cont_unhealthy+=1
            else:
                cont_healthy+=1
            

            # Generate final mask --> post-process
            if args.det_model == "Detic":
                mask_final = process_mask(mask, save=False, save_path=output_folder, max_clusters=30)
            else:
                mask_final = mask
            logger.info("Post-processing time: {:.2f} seconds".format(time.time() - post_start_time))
            
            # Compare results with GT data if exists
            if gt_data and cv2.countNonZero(mask_final):
                iou, dice, jaccard, hausdorff = mask_metrics(gt_mask, mask_final)
                logger.info("IoU:" + str(iou) + " F1-Dice:" + str(dice) + " Jacc:" + str(jaccard) + " Haus:" + str(hausdorff))
                metrics.append([{"iou":iou}, {"Dice": dice}, {"Jaccard": jaccard}, {"Haus": hausdorff}])
            
            # Save output images
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"final_" + file_name), img_out_bbox)
                cv2.imwrite(os.path.join(output_folder,"finalMask_" + file_name), img_out_mask)
                if args.det_model == "Detic":
                    out_img_masked = cv2.bitwise_and(img_ori_np, img_ori_np,  mask=mask_final.astype("uint8"))
                    cv2.imwrite(os.path.join(output_folder,"finalMaskProccessed_" + file_name), out_img_masked)
                if args.print_health:
                    cv2.imwrite(os.path.join(output_folder,"health_" + file_name), img_health)
                    
                # EXTRA CODE TO SAVE MASKS AS DATASET
                output_folder.split("/")[:-1][0]
                extra_folder = os.path.join(output_folder.split("/")[:-1][0], output_folder.split("/")[:-1][1], "mask_data")
                if not os.path.exists(extra_folder):
                    os.makedirs(extra_folder)
                cv2.imwrite(os.path.join(extra_folder, file_name), out_img_masked)
                # Generate metrics json file
                json_path = os.path.join(extra_folder,'stats.json')
                with open(json_path, 'w+') as f:
                    f.write("Healthy {}/{} and Unhealthy{}/{}\n".format(cont_healthy,cont_healthy+cont_unhealthy,cont_unhealthy,cont_healthy+cont_unhealthy))
                    f.write("Num detections {}\n".format(cont_det))
                    f.write("Time: {}\n".format((time.time()-det_start_time)))
                    
        logger.info("Total processing time for {}: {:.2f} seconds".format(file_name, time.time() - start_time))
                
        # Save preds as annotations.txt
        txt_path = os.path.join(exp_folder,"annotations") 
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)
        annotations = pred2COCOannotations(np.asarray(img_ori_np), mask_final, img_health, pred, txt_path, base_name)     
    # Obtain mean metrics and save if GT data exists
    if gt_data: 
        # Calculate and save mean metrics values
        m_iou = 0
        m_f1 = 0 
        m_jc = 0
        m_hf = 0
        m_t = 0
        for ii in range(len(metrics)):
            m_iou = m_iou + metrics[ii][0]["iou"]
            m_f1 = m_f1 + metrics[ii][1]["Dice"]
            m_jc = m_jc + metrics[ii][2]["Jaccard"]
            m_hf = m_hf + metrics[ii][3]["Haus"]
            m_t = m_t + metrics[ii][4]["Time"]
            
        m_iou = m_iou / (ii+1)
        m_f1 = m_f1 / (ii+1)
        m_jc = m_jc / (ii+1)
        m_hf = m_hf / (ii+1)
        m_t = m_t / (ii+1)
        
        logger.info("mean IoU:" + str(m_iou) + " mean F1-Dice:" + str(m_f1) + " mean Jacc:" + str(m_jc) + " mean Haus:" + str(m_hf) + " mean time:" + str(m_t))
        metrics.append([{"mean iou":m_iou}, {"mean Dice": m_f1}, {"mean Jaccard": m_jc}, {"mean Haus": m_hf}, {"mean time": m_t}])
            
        # Generate metrics json file
        json_path = os.path.join(exp_folder,'metrics.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(metrics))