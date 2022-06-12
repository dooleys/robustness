#!/bin/bash -i

# Set up the models directory
MODELS_DIR=$(pwd)

# Constants
CML_DATASETS=/fs/cml-datasets
OUTPUT_DIR=$MODELS_DIR/results

case $1 in

  tinaface)
  
cat << EOF
======================================================================
                        Evaluating TinaFace!!!!!
======================================================================
EOF

cd tinaface
conda activate tinaface
module load cuda/11.1.1
python3 tinaface_infer.py \
  tinaface.py \
  --image_file=$2 \
  --dataset_path=$CML_DATASETS/$3 \
  --output_dir=$OUTPUT_DIR/tinaface/$3
conda deactivate
module rm cuda/11.1.1

cat << EOF
======================================================================
                        Finished TinaFace!!!!!
======================================================================
EOF

    ;;

  yolov5)

cat << EOF
======================================================================
                        Evaluating YOLOv5!!!!!
======================================================================
EOF

cd yolov5-face
conda activate yolov5
module load cuda/11.1.1 
module load cudnn/v8.2.1
python3 yolov5_infer.py \
  weights/face.pt \
  --image_file=$2 \
  --dataset_path=$CML_DATASETS/$3 \
  --output_dir=$OUTPUT_DIR/yolov5/$3 \
  --img-size=640 \
  --conf-thres=0.02 \
  --iou-thres=0.5 \
  --device=0 \
  --augment
module rm cudnn/v8.2.1 cuda/11.1.1
conda deactivate

cat << EOF
======================================================================
                        Finished YOLOv5!!!!!
======================================================================
EOF
    ;;

  mogface)

cat << EOF
======================================================================
                      Evaluating MogFace!!!!!
======================================================================
EOF

cd MogFace
conda activate mogface
module load cuda/10.2.89 
module load cudnn/v8.2.1
python3 mogface_infer.py \
  weights/model_140000.pth \
  --config=configs/mogface/MogFace.yml \
  --image_file=$2 \
  --dataset_path=$CML_DATASETS/$3 \
  --output_dir=$OUTPUT_DIR/mogface/$3 \
  --nms_th=0.6 \
  --pre_nms_top_k=5000 \
  --score_th=0.01 \
  --max_bbox_per_img=750 #\
  # --scale_weight=15 \
  # --max_img_shrink=2.6 \
  # --vote_th=0.6 \
  # --test_min_scale=0 \
  # --test_hard
module rm cudnn/v8.2.1 cuda/10.2.89
conda deactivate

cat << EOF
======================================================================
                      Finished MogFace!!!!!
======================================================================
EOF

    ;;

esac
