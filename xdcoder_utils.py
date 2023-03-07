import sys
sys.path.insert(0, 'X_Decoder/')

import os

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from X_Decoder.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
import cv2

import yaml
import json
import argparse
import logging
import time

logger = logging.getLogger(__name__)


def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        if ori_value:
            logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")


def load_opt_from_config_files(conf_files):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


def load_opt_command(args):
    parser = argparse.ArgumentParser(description='Pretrain or fine-tune models for NLP tasks.')
    parser.add_argument('--command', default="evaluate", help='Command: train/evaluate/train-and-evaluate')
    parser.add_argument('--conf_files', default=['X_Decoder/configs/xdecoder/svlp_focalt_lang.yaml'], nargs='+', help='Path(s) to the config file(s).')
    parser.add_argument('--user_dir', help='Path to the user defined module for tasks (models, criteria), optimizers, and lr schedulers.')
    parser.add_argument('--config_overrides', nargs='*', help='Override parameters on config with a json style string, e.g. {"<PARAM_NAME_1>": <PARAM_VALUE_1>, "<PARAM_GROUP_2>.<PARAM_SUBGROUP_2>.<PARAM_2>": <PARAM_VALUE_2>}. A key with "." updates the object in the corresponding nested dict. Remember to escape " in command line.')
    parser.add_argument('--overrides', help='arguments that used to override the config file in cmdline', nargs=argparse.REMAINDER)
    parser.add_argument('--pretrained_pth', default='X_Decoder/models/xdecoder_focalt_last.pt', help='Path(s) to the weight file(s).')
    parser.add_argument('--output_path', default='./output', help='Path(s) to the output directory.')
    parser.add_argument('--input', help='Path(s) to the input file.')
    parser.add_argument('--xdec_img_size', type=int, default=512 ,help='reshape size for the image to be proccessed wit x-decoder')
    parser.add_argument('--vocabulary_xdec', nargs='+', default=['weed','soil'], help='Concepts to segmentate')

    cmdline_args = parser.parse_args() if not args else parser.parse_args(args)

    opt = load_opt_from_config_files(cmdline_args.conf_files)

    if cmdline_args.config_overrides:
        config_overrides_string = ' '.join(cmdline_args.config_overrides)
        logger.warning(f"Command line config overrides: {config_overrides_string}")
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


def save_opt_to_json(opt, conf_file):
    with open(conf_file, 'w', encoding='utf-8') as f:
        json.dump(opt, f, indent=4)


def save_opt_to_yaml(opt, conf_file):
    with open(conf_file, 'w', encoding='utf-8') as f:
        yaml.dump(opt, f)


def semseg_video(video_pth, transform, model, metadata, output_root):
    # set video input parameters
    video = cv2.VideoCapture(video_pth)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_pth)
    
    # Create Videowriters to generate video output
    file_ext = ".avi"
    path_out_vis = os.path.join(output_root, basename.split(".")[0] + file_ext)
    output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                        (width, height))
    
    frame_count = 0
    # Processing loop
    while (video.isOpened()):
        start_time = time.time()
        # read frame
        ret, frame = video.read()
        if frame is None:
            break
        # predict segmentation with X-DECODER
        frame = Image.fromarray(np.uint8(frame)).convert('RGB')
        img_out = semseg_single_im(frame, transform, model, metadata, output_root, save=False)
        output_file_vis.write(np.uint8(img_out))
        frame_count = frame_count + 1
        end_time = time.time() - start_time
        print("Detection finished in " + str(round(end_time, 2)) + "s")

        
    # Release VideoCapture and VideoWriters
    video.release()
    output_file_vis.release() 
        
        
def semseg_single_im(image_ori, transform, model, metadata, output_root, file_name = "sem.png", save=True):
    with torch.no_grad():
        start_time = time.time()
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        try:
            outputs = model.forward(batch_inputs)
            visual = Visualizer(image_ori, metadata=metadata)

            sem_seg = outputs[-1]['sem_seg'].max(0)[1]
            img_out = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

            if not os.path.exists(output_root):
                os.makedirs(output_root)
            if save:
                img_out.save(os.path.join(output_root, file_name))
            end_time = time.time() - start_time
            print("Detection finished in " + str(round(end_time, 2)) + "s")
        except Exception as e:
            print('Failed in predict step: '+ str(e)) 
            
    return img_out.get_image()


def refseg_video(video_pth, text, transform, model, metadata, output_root):
    # set video input parameters
    video = cv2.VideoCapture(video_pth)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_pth)
    
    # Create Videowriters to generate video output
    file_ext = ".avi"
    path_out_vis = os.path.join(output_root, basename.split(".")[0] + file_ext)
    output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                        (width, height))
    
    frame_count = 0
    # Processing loop
    while (video.isOpened()):
        start_time = time.time()
        # read frame
        ret, frame = video.read()
        if frame is None:
            break
        # predict segmentation with X-DECODER
        frame = Image.fromarray(np.uint8(frame)).convert('RGB')
        img_out = refseg_single_im(frame, text, transform, model, metadata, output_root, save=False)
        output_file_vis.write(np.uint8(img_out))
        frame_count = frame_count + 1
        end_time = time.time() - start_time
        print("Detection finished in " + str(round(end_time, 2)) + "s")

        
    # Release VideoCapture and VideoWriters
    video.release()
    output_file_vis.release() 


def refseg_single_im(image_ori, text, transform, model, metadata, output_root, file_name = "refsem.png", save=True):
    with torch.no_grad():
        start_time = time.time()
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
        
        batch_inputs = [{'image': images, 'height': height, 'width': width, 'groundings': {'texts': text}}]
        outputs = model.model.evaluate_grounding(batch_inputs, None)
        visual = Visualizer(image_ori, metadata=metadata)

        grd_mask = (outputs[0]['grounding_mask'] > 0).float().cpu().numpy() 
        for idx, mask in enumerate(grd_mask):
            img_out_overlay = visual.draw_binary_mask(mask, color=random_color(rgb=True, maximum=1).astype(np.int).tolist(), text=text[idx], alpha=0.5)

        output_folder = os.path.join(os.path.join(output_root))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if save:
            cv2.imwrite(os.path.join(output_folder, "mask_" + file_name + ".png"),grd_mask[0].astype('int8')*255)
        end_time = time.time() - start_time
        print("Segmentation finished in " + str(round(end_time, 2)) + "s")
        
        if np.max(grd_mask) > 0:
            out_img, fruit_zone = mask_cropping(image_ori, grd_mask)
        else: # if not segmentations detected use the full image
            out_img = cv2.cvtColor(np.asarray(image_ori), cv2.COLOR_BGR2RGB)
            fruit_zone= (0,0,out_img.shape[0],out_img.shape[1]) # top left down right
            print("No segmentation results")  

    return out_img, fruit_zone, cv2.cvtColor(img_out_overlay.get_image(), cv2.COLOR_BGR2RGB)

def mask_cropping(img_original, grd_mask, margin_factor=0.1):
    mask = grd_mask[0]
    kernel = np.ones((10,10),np.uint8)
    iterations = 10
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    dilatated_mask = cv2.dilate(opening, kernel, iterations)
    
    positions = np.nonzero(dilatated_mask)
    top = positions[0].min()
    bottom = positions[0].max()
    #left = positions[1].min()
    #right = positions[1].max()
    # fix the right and left position to the entire image
    left = 0
    right = img_original.shape[1]
    
    # cut the upper part of the images (no grapes) and give margin in the botton for grape bunches
    top_grapes = round((top+bottom)/2)
    margin_y = round((bottom-top)*margin_factor)
    margin_x = round((right-left)*margin_factor)
    
    if (bottom + margin_y) < img_original.shape[0]:
        bottom = (bottom + margin_y) 
    else:
        bottom = img_original.shape[0]
        
    if (top_grapes - margin_y) > 0 :
        top_grapes = (top_grapes-margin_y) 
    else:
        top_grapes = top
    
    # fix the right and left position to the entire image    
    '''
    if (right + margin_x) < img_original.shape[1]:
        right = (right + margin_x) 
    else:
        right = img_original.shape[1]
        
    if (left - margin_x) > 0 :
        left = (left-margin_x) 
    else:
        left = 0
    '''
    
    mask_crop = mask[top_grapes:bottom, left:right]
    mask_crop = mask_crop.astype("uint8")*255
    
    img_original_crop =  cv2.cvtColor(img_original[top_grapes:bottom, left:right], cv2.COLOR_BGR2RGB)
    
    #out_img = cv2.bitwise_and(img_original, img_original, mask=mask_crop)
    out_img = img_original_crop
    fruit_zone = (top_grapes, left, bottom, right)
    
    return out_img, fruit_zone