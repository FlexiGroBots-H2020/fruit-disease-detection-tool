import multiprocessing as mp

from fruit_detection_utils import get_parser, load_detic, load_xdecoder
from detic_utils import detic_single_im

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

# constants
WINDOW_NAME = "Detic"

if __name__ == "__main__":
    
    # Set-up models and variables
    setup_logger(name="fvcore")
    logger = setup_logger()
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    
    logger.info("Arguments: " + str(args))
    
    detic_predictor = load_detic(args, logger)
    
    model, transform, metadata, vocabulary_xdec = load_xdecoder(args, logger)
    
    list_images_paths = [] 
    for input in args.input:
        list_images_paths = list_images_paths + glob.glob(input)
        
    test_info = ""
    if args.patching:
        test_info = test_info + "_patch" + str(args.patch_size)
    else:
        test_info = test_info + "_full"
        
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
        'conf_threshold': args.confidence_threshold
    }
    json_path = os.path.join(exp_folder,'variables.json')
    with open(json_path, 'w') as f:
        f.write(json.dumps(variables))
    
        
    # Proccess images
    for img_path in list_images_paths:
        file_name = img_path.split('/')[-1]
        base_name = file_name.split('.')[-2]
        
        output_folder = os.path.join(exp_folder, base_name + test_info) 
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        #output_folder= exp_folder
    
        img = Image.open(img_path).convert("RGB")
        img_or_np = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        
        if args.patching:
            img_grapes_crop, fruit_bbox, img_seg = refseg_single_im(img, vocabulary_xdec, transform, model, metadata, output_folder, base_name, save=False)
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"crop_" + file_name), img_grapes_crop)
            img_out_bbox, img_out_mask = detic_single_im(args, detic_predictor, logger, save=False, save_path=output_folder, img_original=img_or_np, img_processed=img_grapes_crop, fruit_zone=fruit_bbox)
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"final_" + file_name), img_out_bbox)
                cv2.imwrite(os.path.join(output_folder,"finalMask_" + file_name), img_out_mask)
        else:
            img_out_bbox, img_out_mask = detic_single_im(args, detic_predictor, logger, save=False, save_path=output_folder, img_processed=img_or_np)
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"final_" + file_name), img_out_bbox)
                cv2.imwrite(os.path.join(output_folder,"finalMask_" + file_name), img_out_mask)