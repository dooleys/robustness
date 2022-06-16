# Steps

These are the steps to process the corrupted images and with [MogFace](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_MogFace_Towards_a_Deeper_Appreciation_on_Face_Detection_CVPR_2022_paper.pdf), [TinaFace](https://arxiv.org/pdf/2011.13183.pdf), and [YOLO5Face](https://arxiv.org/pdf/2105.12931?ref=https://githubhelp.com).

## 1) Download models and setup virtual environments

In order to download all of the pretrained models and create separate virtual environments for each one, make sure to have [Miniconda](https://docs.conda.io/en/latest/miniconda.html). For each model below, we list the additional dependencies tested.

- TinaFace
  - `CUDA` 11.1
  - `gcc` 7.5
- YOLO5Face
  - `CUDA` 11.1
- MogFace
  - `CUDA` 10.2
  - `CUDnn` 8.2

_From our recollection, the `CUDA` version needed for TinaFace is quite finnicky due to the dependence on [TensorRT](https://developer.nvidia.com/tensorrt). Check Nvidia's documentation for which `CUDA` versions and `python` versions that are compatible. The `CUDA` version is pretty flexible of a dependency in general, we just happened to get everything working with these dependencies._

In order to download each model and setup their corresponding virtual environment, run:

```bash
./setup.sh
```

## 2) Perform inference

Once all of the models are setup, it's now time to perform inference. To do so, make sure that you have the datasets in the below directory structure:

```
/path/to/datasets
├─ adience
│  └─ ...
├─ ccd
│  └─ ...
├─ miap
│  └─ ...
└─utkface
   └─ ...
```

Specifically in [`infer.sh`](infer.sh), change `DATASETS` to point to the correct datasets root path.

Finally, you will need a text file that contains the absolute paths to all of the images for each dataset. To do this, run:

```bash
find </path/to/datasets/dataset/> -type f > <dataset>.txt
```

for each dataset.

To get the predictions made by the specified model on the specified dataset, run:

```bash
MODEL=<model> DATASET=<dataset> IMAGE_FILE=<dataset>.txt ./infer.sh
```

If you use SLURM to schedule your job, edit both the top of [`infer.sh`](infer.sh) and [`submit.py`](submit.py) to make sure everything submits correctly with your SLURM configuration. To submit with SLURM, run:

```bash
python3 submit.py \
  --model <model> \
  --image_file <dataset>.txt \
  --dataset <dataset> \
  --time <time> \
  --partition <partition> \
  --gpu <gpu>
```

The inference of any given model on any given dataset will produce a JSON file for each image (`path/to/datasets/dataset/image.json`) which is structured like this example:

```json
[
    {
        "xmin": 0,
        "ymin": 120,
        "xmax": 100,
        "ymax": 200,
        "confidence": 0.95,
    },
    ...
]
```

## 3) Aggregate inference results

In order to create a combined JSON for each model on each dataset, run:

```bash
python3 aggregate_jsons.py \
    --results_root <path/to/results/> \
    --models <model 1> [<model 2>, ...] \
    --datasets <dataset 1> [<dataset 2>, ...] \
    [--verbose]
```

The final format from this aggregation script for a given models prediction on a given dataset is like this:

```json
[
    {
        "json": "path/to/json",
        "response": [
            {
                "xmin": 0,
                "ymin": 120,
                "xmax": 100,
                "ymax": 200,
                "confidence": 0.95,
            },
            ...
        ]
    }
]
```

# Citation

```bibtex
@article{dooley2022commercial,
  title={Are Commercial Face Detection Models as Biased as Academic Models?},
  author={Dooley, Samuel and Wei, George Z. and Goldstein, Tom, and Dickerson, John P.},
  journal={Working Paper},
  year={2022}
}
```

# Contact

If you'd like more information, get in [contact with us](mailto:gzwei@umass.edu)! Happy to share more details or data or answer questions.
