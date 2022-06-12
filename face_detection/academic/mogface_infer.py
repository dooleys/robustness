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

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from core.workspace import create, load_config
from data import anchor_utils, BasePreprocess, DataAugSettings, GeneartePriorBoxes
from utils.nms.nms_wrapper import nms


class FaceDetBiasDataset(Dataset):
    """A PyTorch Dataset that loads the images in the given image list."""

    def __init__(self, image_file: str, phase: str, transform: BasePreprocess) -> None:
        super().__init__()
        self.image_list = pd.read_csv(image_file, header=None)[0]
        self.phase = phase
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        image_path = self.image_list[idx]
        image = cv2.imread(image_path).astype(np.float32)
        image = self.transform(image, phase=self.phase)
        return image, image_path


def collate_fn(data: List[Tuple[np.ndarray, str]]) -> Tuple[np.ndarray, str]:
    return data[0]


def output_file_namer(output_dir: Path, dataset_path: str, image_name: str) -> Path:
    subdirectories = os.path.dirname(image_name).replace(dataset_path, "").lstrip("/")
    image_name = os.path.splitext(os.path.basename(image_name))[0]
    return output_dir / subdirectories / f"{image_name}.json"


def detect_face(image: np.ndarray, shrink: float) -> np.ndarray:
    x = image
    if shrink != 1:
        x = cv2.resize(
            image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR
        )

    width = x.shape[1]
    height = x.shape[0]

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.cuda()

    out = net(x)

    anchors = anchor_utils.transform_anchor(generate_anchors_fn(height, width))
    anchors = torch.FloatTensor(anchors).cuda()
    decode_bbox = anchor_utils.decode(out[1].squeeze(0), anchors)
    boxes = decode_bbox
    scores = out[0].squeeze(0)

    top_k = args.pre_nms_top_k
    v, idx = scores[:, 0].sort(0)
    idx = idx[-top_k:]
    boxes = boxes[idx]
    scores = scores[idx]

    # [11620, 4]
    boxes = boxes.detach().cpu().numpy()
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    boxes[:, 0] /= shrink
    boxes[:, 1] /= shrink
    boxes[:, 2] = boxes[:, 0] + w / shrink - 1
    boxes[:, 3] = boxes[:, 1] + h / shrink - 1
    # boxes = boxes / shrink
    # [11620, 2]
    # if args.test_min_scale != 0 :
    #     boxes_area = (boxes[:, 3] - boxes[:, 1] + 1) * (boxes[:, 2] - boxes[:, 0] + 1) /  (shrink * shrink)
    #     boxes = boxes[boxes_area >  args.test_min_scale**2]
    #     scores = scores[boxes_area > args.test_min_scale**2]

    scores = scores.detach().cpu().numpy()

    inds = np.where(scores[:, 0] > args.score_th)[0]
    if len(inds) == 0:
        det = np.empty([0, 5], dtype=np.float32)
        return det
    c_bboxes = boxes[inds]
    # [5,]
    c_scores = scores[inds, 0]
    # [5, 5]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
        np.float32, copy=False
    )

    keep = nms(c_dets, args.nms_th)

    c_dets = c_dets[keep, :]

    max_bbox_per_img = args.max_bbox_per_img
    if max_bbox_per_img > 0:
        image_scores = c_dets[:, -1]
        if len(image_scores) > max_bbox_per_img:
            image_thresh = np.sort(image_scores)[-max_bbox_per_img]
            keep = np.where(c_dets[:, -1] >= image_thresh)[0]
            c_dets = c_dets[keep, :]
    return c_dets


def multi_scale_test(image: np.ndarray, max_im_shrink: float) -> np.ndarray:
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s, detect_face(image, 0.75)))
    # index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    if args.scale_weight == -1:
        index = np.where(
            np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1)
            > 30
        )[0]
    else:
        index = np.where(
            ((det_s[:, 2] - det_s[:, 0]) * (det_s[:, 3] - det_s[:, 1])) > 2000
        )[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)
    if args.scale_weight == -1:
        index = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
            < 100
        )[0]
    else:
        index = np.where(
            ((det_b[:, 2] - det_b[:, 0]) * (det_b[:, 3] - det_b[:, 1]))
            < args.scale_weight * 600
        )[0]
    det_b = det_b[index, :]

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_tmp = detect_face(image, 1.5)
        if args.scale_weight == -1:
            index = np.where(
                np.minimum(
                    det_tmp[:, 2] - det_tmp[:, 0] + 1, det_tmp[:, 3] - det_tmp[:, 1] + 1
                )
                < 100
            )[0]
        else:
            index = np.where(
                ((det_tmp[:, 2] - det_tmp[:, 0]) * (det_tmp[:, 3] - det_tmp[:, 1]))
                < args.scale_weight * 800
            )[0]
        det_tmp = det_tmp[index, :]
        det_b = np.row_stack((det_b, det_tmp))

    if max_im_shrink > 2:
        det_tmp = detect_face(image, max_im_shrink)
        if args.scale_weight == -1:
            index = np.where(
                np.minimum(
                    det_tmp[:, 2] - det_tmp[:, 0] + 1, det_tmp[:, 3] - det_tmp[:, 1] + 1
                )
                < 100
            )[0]
        else:
            index = np.where(
                ((det_tmp[:, 2] - det_tmp[:, 0]) * (det_tmp[:, 3] - det_tmp[:, 1]))
                < args.scale_weight * 500
            )[0]
        det_tmp = det_tmp[index, :]
        det_b = np.row_stack((det_b, det_tmp))

    return det_s, det_b


def multi_scale_test_pyramid(image: np.ndarray, max_shrink: float) -> np.ndarray:
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    if args.scale_weight == -1:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
            > 30
        )[0]
    else:
        index = np.where(
            ((det_b[:, 2] - det_b[:, 0]) * (det_b[:, 3] - det_b[:, 1])) > 2000
        )[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if st[i] <= max_shrink:
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if i == 0:
                if args.scale_weight == -1:
                    index = np.where(
                        np.maximum(
                            det_temp[:, 2] - det_temp[:, 0] + 1,
                            det_temp[:, 3] - det_temp[:, 1] + 1,
                        )
                        > 30
                    )[0]
                else:
                    index = np.where(
                        (
                            (det_temp[:, 2] - det_temp[:, 0])
                            * (det_temp[:, 3] - det_temp[:, 1])
                        )
                        < args.scale_weight * 2000
                    )[0]
                det_temp = det_temp[index, :]
            if i == 1:
                if args.scale_weight == -1:
                    index = np.where(
                        np.minimum(
                            det_temp[:, 2] - det_temp[:, 0] + 1,
                            det_temp[:, 3] - det_temp[:, 1] + 1,
                        )
                        < 100
                    )[0]
                else:
                    index = np.where(
                        (
                            (det_temp[:, 2] - det_temp[:, 0])
                            * (det_temp[:, 3] - det_temp[:, 1])
                        )
                        < args.scale_weight * 1000
                    )[0]
                det_temp = det_temp[index, :]
            if i == 2:
                if args.scale_weight == -1:
                    index = np.where(
                        np.minimum(
                            det_temp[:, 2] - det_temp[:, 0] + 1,
                            det_temp[:, 3] - det_temp[:, 1] + 1,
                        )
                        < 100
                    )[0]
                else:
                    index = np.where(
                        (
                            (det_temp[:, 2] - det_temp[:, 0])
                            * (det_temp[:, 3] - det_temp[:, 1])
                        )
                        < args.scale_weight * 600
                    )[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))

    return det_b


def flip_test(image: np.ndarray, shrink: float) -> np.ndarray:
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2] - 1
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0] - 1
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det: np.ndarray) -> np.ndarray:
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    det[:, :4] = np.round(det[:, :4])
    while det.shape[0] > 0:
        # IOU
        box_w = np.maximum(det[:, 2] - det[:, 0], 0)
        box_h = np.maximum(det[:, 3] - det[:, 1], 0)
        area = box_w * box_h
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area[0] + area[:] - inter
        union[union <= 0] = 1
        o = inter / union
        o[0] = 1

        # get needed merge det and delete these det
        merge_index = np.where(o >= args.vote_th)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
            det_accu[:, -1:]
        )
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    if dets.shape[0] > 750:
        dets = dets[0:750, :]
    return dets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Details")
    parser.add_argument("weights", type=str, help="Pretrained model checkpoint")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="yaml file that contains all of the configurations for the given model.",
    )
    parser.add_argument(
        "--image_file",
        required=True,
        type=Path,
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
    parser.add_argument("--nms_th", default=0.6, type=float, help="nms threshold.")
    parser.add_argument(
        "--pre_nms_top_k", default=5000, type=int, help="number of max score image."
    )
    parser.add_argument("--score_th", default=0.01, type=float, help="score threshold.")
    parser.add_argument(
        "--max_bbox_per_img", default=750, type=int, help="max number of det bbox."
    )

    # Comment in all of the commented out lines for multi-scale testing
    # parser.add_argument("--scale_weight", default=15, type=float,
    #                     help="to differentiate the gap between large and small scale.")
    # parser.add_argument("--max_img_shrink", default=2.6, type=float,
    #                     help="constrain the max shrink of img.")
    # parser.add_argument("--vote_th", default=0.6, type=float,
    #                     help="bbox vote threshold")
    # parser.add_argument("--test_min_scale", default=0, type=int,
    #                     help="the min scale of det bbox")
    # parser.add_argument("--flip_ratio", default=None, type=float,
    #                     help="Whether to flip the image and get detections.")
    # parser.add_argument("--test_hard", action="store_true",
    #                     help="Whether to lower thresholds for harder images.")

    args = parser.parse_args()
    # if args.test_hard:
    #     args.max_img_shrink = 2.3
    #     args.vote_th = 0.5
    #     args.nms_th = 0.4
    #     args.scale_weight = 10
    #     args.flip_ratio = 1.4

    # generate det_info and det_result
    cfg = load_config(args.config)
    cfg["phase"] = "test"

    # create net and val_set
    net = create(cfg.architecture)
    model_path = args.weights
    print(f"Loading model from {model_path}")
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    print("Finished loading model.")

    cfg["BasePreprocess"]["data_aug_settings"] = DataAugSettings()

    transform = BasePreprocess(**cfg["BasePreprocess"])
    dataset = FaceDetBiasDataset(args.image_file, cfg["phase"], transform)
    dataloader = DataLoader(dataset, collate_fn=collate_fn)

    generate_anchors_fn = GeneartePriorBoxes(**cfg["GeneartePriorBoxes"])

    with torch.no_grad():
        # generate predict bbox
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

            max_img_shrink = (
                0x7FFFFFFF / 200.0 / (image.shape[0] * image.shape[1])
            ) ** 0.5  # the max size of input image for caffe
            max_img_shrink = min(
                max_img_shrink, 2.2
            )  # args.max_img_shrink if max_img_shrink > 2.2 else max_img_shrink
            shrink = min(max_img_shrink, 1)
            det0 = detect_face(image, shrink)  # origin test
            # det1 = flip_test(image, shrink)    # flip test
            # det2, det3 = multi_scale_test(image, max_img_shrink)
            # det4 = multi_scale_test_pyramid(image, max_img_shrink)
            # if args.flip_ratio is not None:
            #     det5 = flip_test(image, args.flip_ratio)

            # if args.flip_ratio is not None:
            #     det = np.row_stack((det0, det1, det2, det3, det4, det5))
            # else:
            #     det = np.row_stack((det0, det1, det2, det3, det4))
            dets = det0  # bbox_vote(det)

            height, width = image.shape[:2]

            dets[:, :2] = dets[:, :2].clip(min=0)
            dets[:, 2] = dets[:, 2].clip(max=width - 1)
            dets[:, 3] = dets[:, 3].clip(max=height - 1)
            dets[:, 4] = dets[:, 4].clip(max=1)

            reformatted_result = [
                {
                    "xmin": bbox[0],
                    "ymin": bbox[1],
                    "xmax": bbox[2],
                    "ymax": bbox[3],
                    "confidence": bbox[4],
                }
                for bbox in dets.tolist()
            ]

            with open(output_path, "w") as f:
                json.dump(reformatted_result, f)

        print(f"Done processing everything in {args.dataset_path}")
