from __future__ import absolute_import

import argparse
import json
import os

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression_face,
    scale_coords,
    xyxy2xywh,
)
from utils.torch_utils import select_device


class FaceDetBiasDataset(Dataset):
    """A PyTorch Dataset that loads the images in the given image list."""

    def __init__(self, image_file: str) -> None:
        super().__init__()
        self.image_list = pd.read_csv(image_file, header=None)[0]

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        image_path = self.image_list[idx]
        image = cv2.imread(image_path)
        return image, image_path


def collate_fn(data: List[Tuple[np.ndarray, str]]) -> Tuple[np.ndarray, str]:
    return data[0]


def output_file_namer(output_dir: Path, dataset_path: str, image_name: str) -> Path:
    subdirectories = os.path.dirname(image_name).replace(dataset_path, "").lstrip("/")
    image_name = os.path.splitext(os.path.basename(image_name))[0]
    return output_dir / subdirectories / f"{image_name}.json"


def dynamic_resize(shape: Tuple[int], stride: int = 64) -> int:
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride
    return max_size


def detect(model: nn.Module, img0: np.ndarray) -> List[List[int]]:
    imgsz = args.img_size
    if imgsz <= 0:  # original size
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]
    # Convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(
        2, 0, 1
    )  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = (img / 255.0).astype(np.float32)  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=args.augment)[0]
    # Apply NMS
    pred = non_max_suppression_face(pred, args.conf_thres, args.iou_thres)[0]
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
    boxes = []
    h, w, _ = img0.shape
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for j in range(pred.size(0)):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.detach().cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            boxes.append([x1, y1, x2, y2, conf])
    return boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=str, help="Pretrained model checkpoint")
    parser.add_argument(
        "--image_file",
        required=True,
        type=str,
        help="Input file containing the absolute paths to images to get face detection predictions on.",
    )
    parser.add_argument(
        "--dataset_path", required=True, type=str, help="The path to the dataset root."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory to save detection JSONs to (default: same subdirectory as those for the input image).",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.02, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    args = parser.parse_args()
    print(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(args.device)
    model = attempt_load(args.weights, map_location=device)  # load FP32 model

    dataset = FaceDetBiasDataset(args.image_file)
    dataloader = DataLoader(dataset, collate_fn=collate_fn)

    with torch.no_grad():
        for image, image_path in tqdm(dataloader):
            if image is None:
                print(f"{image_path} is a problematic image")
                continue

            output_path = output_file_namer(
                args.output_dir, args.dataset_path, image_path
            )

            if output_path.is_file():
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            boxes = detect(model, image)
            # --------------------------------------------------------------------
            reformatted_result = [
                {
                    "xmin": bbox[0],
                    "ymin": bbox[1],
                    "xmax": bbox[2],
                    "ymax": bbox[3],
                    "confidence": min(bbox[4], 1),
                }
                for bbox in boxes
            ]

            with open(output_path, "w") as f:
                json.dump(reformatted_result, f)

        print(f"Done processing everything in {args.dataset_path}")
