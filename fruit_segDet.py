import multiprocessing as mp

from fruit_detection_utils import setup_cfg, get_parser 
from detic_utils import detic_single_im, detic_video

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')

from detectron2.utils.logger import setup_logger
from Detic.detic.predictor import VisualizationDemo

# constants
WINDOW_NAME = "Detic"

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Instance Detic Predictor
    try:
        detic_predictor = VisualizationDemo(cfg, args)
    except:
        # second time it works
        detic_predictor = VisualizationDemo(cfg, args)
        logger.warning("w: 'CustomRCNN' was already registered")

    frame_count = 0

    # Infer in single video
    if args.input[0].endswith("mp4") or args.input[0].endswith("MP4") or args.input[0].endswith("avi"):
        detic_video(args, detic_predictor, logger)

    # Infer in image
    else:
        detic_single_im(args, detic_predictor, logger)
        