import argparse
import json
import os

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from flexinfer.misc import Config, set_device
from flexinfer.preprocess import build_preprocess
from flexinfer.model import build_model
from flexinfer.postprocess import build_postprocess

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm


class FaceDetBiasDataset(Dataset):
    """A PyTorch Dataset that loads the images in the given image list."""

    def __init__(self, image_file: str, transform: transforms.Compose) -> None:
        super().__init__()
        self.image_list = pd.read_csv(image_file, header=None)[0]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        image_path = self.image_list[idx]
        image = cv2.imread(image_path)
        image = [dict(img=image)]
        image_data = self.transform(image)

        return image_data, image_path


def collate_fn(data: List[Tuple[np.ndarray, str]]) -> Tuple[np.ndarray, str]:
    return data[0]


def output_file_namer(output_dir: Path, dataset_path: str, image_name: str) -> Path:
    subdirectories = os.path.dirname(image_name).replace(dataset_path, "").lstrip("/")
    image_name = os.path.splitext(os.path.basename(image_name))[0]
    return output_dir / subdirectories / f"{image_name}.json"


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)

    # 1. set gpu id
    set_device(cfg.gpu_id)

    # 2. build preprocess
    transform = build_preprocess(cfg.preprocess)

    # 3. build model
    model = build_model(cfg.model)

    # 4. build postprocess
    postprocess = build_postprocess(cfg.postprocess)

    dataset = FaceDetBiasDataset(image_file=args.image_file, transform=transform)
    dataloader = DataLoader(dataset, collate_fn=collate_fn)

    for image_data, image_path in tqdm(dataloader):
        output_path = output_file_namer(args.output_dir, args.dataset_path, image_path)

        if output_path.is_file():
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            image_data["out"] = model(image_data.pop("img"))
        except AssertionError:
            print(f"{image_path} is a problematic image")
            continue

        results = np.vstack(postprocess(image_data)[0])
        reformatted_result = [
            {
                "xmin": bbox[0],
                "ymin": bbox[1],
                "xmax": bbox[2],
                "ymax": bbox[3],
                "confidence": bbox[4],
            }
            for bbox in results.tolist()
        ]

        with open(output_path, "w") as f:
            json.dump(reformatted_result, f)

    print(f"Done processing everything in {args.dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Facial detection inference code for TinaFace"
    )
    parser.add_argument("config", type=str, help="config file")
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

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
