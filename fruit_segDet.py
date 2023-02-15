import multiprocessing as mp

from fruit_detection_utils import get_parser, load_detic, load_xdecoder
from detic_utils import detic_single_im

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, 'X_Decoder/')

from detectron2.utils.logger import setup_logger
from xdcoder_utils import refseg_single_im, refseg_video, load_opt_command
from PIL import Image
import cv2
import os
import numpy as np
import time

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
    
    frame_count = 0
    
    for input in args.input:
    
        file_name = input.split('/')[-1]
        base_name = file_name.split('.')[-2]

        # Infer in single video
        if input.endswith("mp4") or input.endswith("MP4") or input.endswith("avi"):
            print("no video yet")
            """
            # set video input parameters
            video = cv2.VideoCapture(input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename_path = os.path.basename(base_name)

            # Create Videowriters to generate video output
            file_ext = ".avi"
            path_out_vis = os.path.join(args.output, base_name + file_ext)
            path_out_mask = os.path.join(args.output, base_name + "_mask" + file_ext)
            output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                                (width, height))
            output_file_mask = cv2.VideoWriter(path_out_mask, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                                (width, height))

            frame_count = 0
            # Processing loop
            while (video.isOpened()):
                # read frame
                ret, frame = video.read()
                if frame is None:
                    break
                # predict detections DETIC
                start_time = time.time()
                id = base_name + "_fr" + str(frame_count) + "_"
                img_grapes_crop, increments, img_seg = refseg_single_im(Image.fromarray(frame), vocabulary_xdec, transform, model, metadata, args.output, id, save=True)
                if args.debug:
                        cv2.imwrite(os.path.join(args.output,"crop_" + id +".png"), img_grapes_crop)
                        cv2.imwrite(os.path.join(args.output,"overlap_" + id +".png"), img_seg)
                img_out_bbox, img_out_mask = detic_single_im(args, detic_predictor, logger, save=True, img_original=frame, img_processed=img_grapes_crop, increments=increments)
                    
                # write results to video output
                output_file_vis.write(np.uint8(img_out_bbox))
                output_file_mask.write(np.uint8(img_out_mask))
                frame_count = frame_count + 1
                end_time = time.time() - start_time
                print("Detection finished in " + str(round(end_time, 2)) + "s")

                
            # Release VideoCapture and VideoWriters
            video.release()
            output_file_vis.release() 
            output_file_mask.release()      
            """
            

        # Infer in image
        else:
            img = Image.open(input).convert("RGB")
            img_or_np = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
            img_grapes_crop, fruit_bbox, img_seg = refseg_single_im(img, vocabulary_xdec, transform, model, metadata, args.output, base_name, save=True)
            if args.debug:
                cv2.imwrite(os.path.join(args.output,"crop_" + file_name), img_grapes_crop)
            img_out_bbox, img_out_mask = detic_single_im(args, detic_predictor, logger, save=False, img_original=img_or_np, img_processed=img_grapes_crop, fruit_zone=fruit_bbox)
            if args.debug:
                cv2.imwrite(os.path.join(args.output,"final_" + file_name), img_out_bbox)
                cv2.imwrite(os.path.join(args.output,"finalMask_" + file_name), img_out_mask)