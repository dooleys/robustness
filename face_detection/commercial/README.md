# Steps

These are the steps to create the corrupted images and process them with the commercial APIs.

## 1) Download the images

The download links for the images are included above. The download and extraction of the images from Adience, MIAP, and UTKFace are straightforward.

The CCD dataset is broken into >30 zip files, each around 100GBs. Unfortunately, they are not organized neatly and we had to download all the files and unzip them. The CCD dataset contains videos and not images. In order to turn them into images, we used [ffmpeg](https://ffmpeg.org/). Once all the video files are extracted, create a file `mp4_files.txt` that contains a list of the movies you want to extract frames for. Then run

```bash
sh ./video_extract_frame.sh mp4_files.txt
```

## 2) Select which images to include in the benchmark

Once all the clean images are downloaded and read, create a file which has all the image locations for those which you would like to corrupt. We followed a procedure to corrupt just a subset of the images in the four datasets. We chose no more than 1,500 images from each intersectional identity for each dataset. You can obtain a list like this by using the dataload objects included in `datasets.py` to load in the image metadata and then running the `select_unique_id` method to get the desired sample of images. Here is an example for the Adience dataset:

```python
from datasets import *
adience = adience_dataset()
adience.load_metadata()
adience.select_unique_ids()
```

The images that we used for our benchmark can be found in the `./data/` folder.

## 3) Corrupt the images

We used the ImageNet-C procedures to corrupt each image. The ImageNet-C code is designed for images which are `244x244` pixels -- much smaller than the images in our benchmark. To account for this, we have included in the folder `./code/imagenet_c_big/` a modified version of that code which removes the 244 assumption.

First create a list of the images you'd like to corrupt in a file; we'll call it `images_to_corrupt.txt`. Then call the `distort.py` file. You have the options to download the images in your list (if they are an S3 bucket) or not if they are local images. You also have the option to upload the corrupted images to S3 if you'd like. The default command is:

```bash
python3 distort.py images_to_corrupt.txt
```

## 4) Make API calls

Now that you have all your corrupted images, it is time to send the images to AWS and Azure to get their face detections.

First get a list of all the images you'd like to send to the services; we'll call it `images_to_detect.txt`. Then call the `detect_faces.py` file. This script assumes that each image is in an S3 bucket for faster processing. If you wanted to process every image in your file with AWS Rekognition, you'd run the following command:

```bash
python3 detect_faces.py images_to_detect.txt aws
```

This will save one JSON for each image and response. The JSON is structured like this example:

```json
{
    "original_photo": "path/to/original_photo.png",
    "photo": "original_photo.png",
    "corruption": "gaussian-noise",
    "severity": 4,
    "service": "aws",
    "response":
            ...
}
```

# Citation

```bibtex
@article{dooley2021robustness,
  title={Robustness Disparities in Commercial Face Detection},
  author={Dooley, Samuel and Goldstein, Tom and Dickerson, John P.},
  journal={Working Paper},
  year={2021}
}
```

# Contact

If you'd like more information, get in [contact with us](mailto:sdooley1@cs.umd.edu)! Happy to share more details or data or answer questions.
