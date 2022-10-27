from unicodedata import name
import torch

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os
import os.path as osp
import cv2
import sys
from tqdm import tqdm
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import nuScenes-devkit
from nuscenes.nuscenes import NuScenes

# IDs for the following classes of interest: ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "bird", "cat", "dog"]
# Essentially, we only want to keep moving entities that may occur in nuscenes images
# see https://github.com/facebookresearch/detectron2/issues/147#issuecomment-645958806
CLASSES_OF_INTEREST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16]

# Define nuScenes camera channels
NUSCENES_CAMERA_CHANNELS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]

# Get Model's Config and instantiate the predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

def create_mask(image, mask_filename):
    # Generate outputs
    outputs = predictor(image)

    # Only keep detected instances of our classes of interest
    instances = outputs["instances"]
    if instances.pred_classes.size(dim=0) != 0:
        instances = instances[torch.as_tensor([elem in CLASSES_OF_INTEREST for elem in instances.pred_classes])]

    # Use `Visualizer` to draw blue masks on the black image
    v = Visualizer(np.zeros_like(image[:, :, ::-1]), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    masks = instances.pred_masks.cpu().data.numpy()
    for m in masks:
        v.draw_binary_mask(m, color="b")
    output = v.get_output()

    # Save the freshly created mask
    mask_file = output.get_image()[:, :, ::-1]
    cv2.imwrite(mask_filename, mask_file)

def process_nuscenes_record(nusc, record):
    for camera_channel in NUSCENES_CAMERA_CHANNELS:
        camera_token = record['data'][camera_channel]
        cam = nusc.get('sample_data', camera_token)
        filename = cam['filename'].split("/")[2][:-4]
        im = cv2.imread(osp.join(nusc.dataroot, cam['filename']))
        mask_filename = f'masks/dynamic_mask_{filename}.png'
        create_mask(im, mask_filename)

def create_masks_for_nuscenes_sequence(nuscenes_root_dir, scene_index):
    # Process nuscenes images
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_root_dir, verbose=True)
    scene = nusc.scene[scene_index]
    current_token = scene["first_sample_token"]
    last_sample_token = scene["last_sample_token"]
    n_samples = scene["nbr_samples"]

    print("Start creating masks for nuscenes images...")

    pbar = tqdm(total=n_samples)
    while current_token != last_sample_token:
        sample_record = nusc.get('sample', current_token)
        process_nuscenes_record(nusc, sample_record)
        current_token = sample_record["next"]
        pbar.update(1)
    pbar.close()

    last_record = nusc.get('sample', last_sample_token)
    process_nuscenes_record(nusc, last_record)

    print("Masks successfully created!")

if __name__ == "__main__":
    # Create mask directories
    masks_path = "masks"
    if not osp.exists(masks_path):
        os.makedirs(masks_path)

    nuscenes_root_dir = sys.argv[1]
    nuscenes_scene_index = int(sys.argv[2])

    # Create and save masks
    create_masks_for_nuscenes_sequence(nuscenes_root_dir, nuscenes_scene_index)