# Robustness Disparities in Face Detection

## Abstract

Facial analysis systems have been deployed by large companies and critiqued by scholars and activists for the past decade. Many existing algorithmic audits examine the performance of these systems on later stage elements of facial analysis systems like facial recognition and age, emotion, or gender prediction; however, a core component to these systems has been vastly understudied from a fairness perspective: face detection. Since face detection is a pre-requisite step in facial analysis systems, the bias we observe in face detection will flow downstream to the other components like facial recognition and emotion prediction. Additionally, no prior work has focused on the robustness of these systems under various perturbations and corruptions, which leaves open the question of how various people are impacted by these phenomena. We present the first of its kind detailed benchmark of face detection systems, specifically examining the robustness to noise of commercial and academic models. We use both standard and recently released academic facial datasets to quantitatively analyze trends in face detection robustness. Across all the datasets and systems, we generally find that photos of individuals who are _masculine presenting_, _older_, of _darker skin type_, or have _dim lighting_ are more susceptible to errors than their counterparts in other identities.

## About the Benchmark

This benchmark uses four datasets to evaluate the robustness of face detection systems to natural types of noise.

- **[Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)**
- **[Casual Conversations Dataset (CCD)](https://ai.facebook.com/datasets/casual-conversations-dataset/)**
- **[Open Images V6 -- Extended; More Inclusive Annotations for People (MIAP)](https://storage.googleapis.com/openimages/web/extended.html)**
- **[UTKFace](https://susanqq.github.io/UTKFace/)**

For a subset of the images in this dataset, we created 75 corrupted versions following the [ImageNet-C](https://github.com/hendrycks/robustness) pipeline.

Subsequently, each image (1 clean + 75 corrupted images) was passed through Amazon Web Services's [Rekognition](https://docs.aws.amazon.com/rekognition/latest/dg/API_DetectFaces.html) and Microsoft [Azure](https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236) face detection APIs.

We evaluated each image on each of the following six face detection models, three of which are produced my academic research groups, and three by commercial companies:

### Academic Face Detection Models

- **[MogFace](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_MogFace_Towards_a_Deeper_Appreciation_on_Face_Detection_CVPR_2022_paper.pdf)**
- **[TinaFace](https://arxiv.org/pdf/2011.13183.pdf)**
- **[YOLO5Face](https://arxiv.org/pdf/2105.12931?ref=https://githubhelp.com)**

### Commercial Face Detection Models

- **[Amazon Web Services Rekognition](https://docs.aws.amazon.com/rekognition/latest/dg/faces.html)**
- **[Azure](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-face-detection)**
- **[Google Cloud Platform](https://cloud.google.com/vision/docs/detecting-faces)**

## About this Repo

We conducted the image corruption and commercial models parts of this benchmark using AWS's S3 and EC2 infrastructure. The image datasets were downloaded to an S3 bucket, processed/corrupted using EC2 instances (primarily `c5.large`), and then passed through each API using EC2 instances (`i3.xlarge`) and storing responses in an S3 bucket. This process was specific to our choices, though any compute environment could be used to reproduce these results. To that end, we will include the essential code used to process the images and make the API calls, and do not include specific and superfluous data management scripts which would be idiosyncratic to the specific process we chose.

In the `face_detection` folder there are two sub directories: `academic` which has the code to process the academic models, and `commercial` which has the code to create the corrupted images and process each one with the academic APIs.

The `docs` folder contains the [website's](https://dooleys.github.io/robustness) code.

# Citation

```
@article{dooley2022robustness,
  title={Robustness Disparities in Face Detection},
  author={Dooley, Samuel and Wei, George Z. and Goldstein, Thomas and Dickerson, John P.},
  journal={Working Paper},
  year={2022}
}
```

# Contact

If you'd like more information, get in [contact with us](mailto:sdooley1@cs.umd.edu)! Happy to share more details or data or answer questions.
