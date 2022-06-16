#!/bin/bash -i

#SBATCH --job-name={model}-{dataset}
#SBATCH --output=output/{model}/{dataset}.%j.log
#SBATCH --error=output/{model}/{dataset}.%j.err
#SBATCH --qos={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time={time}
#SBATCH --gres={gpu}
#SBATCH --mem=16gb

# Set up the models directory
MODELS_DIR=$(pwd)

# Constants
DATASETS=</path/to/datasets/>
OUTPUT_DIR=$MODELS_DIR/results

case $MODEL in

  tinaface)
  
cat << EOF
======================================================================
                        Evaluating TinaFace!!!!!
======================================================================
EOF

cd tinaface
conda activate tinaface
python3 tinaface_infer.py \
  tinaface.py \
  --image_file=$IMAGE_FILE \
  --dataset_path=$DATASETS/$DATASET \
  --output_dir=$OUTPUT_DIR/tinaface/$DATASET
conda deactivate

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
python3 yolov5_infer.py \
  weights/face.pt \
  --image_file=$IMAGE_FILE \
  --dataset_path=$DATASETS/$DATASET \
  --output_dir=$OUTPUT_DIR/yolov5/$DATASET \
  --img-size=640 \
  --conf-thres=0.02 \
  --iou-thres=0.5 \
  --device=0 \
  --augment
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
  --image_file=$IMAGE_FILE \
  --dataset_path=$DATASETS/$DATASET \
  --output_dir=$OUTPUT_DIR/mogface/$DATASET \
  --nms_th=0.6 \
  --pre_nms_top_k=5000 \
  --score_th=0.01 \
  --max_bbox_per_img=750 #\
  # --scale_weight=15 \
  # --max_img_shrink=2.6 \
  # --vote_th=0.6 \
  # --test_min_scale=0 \
  # --test_hard
conda deactivate

cat << EOF
======================================================================
                      Finished MogFace!!!!!
======================================================================
EOF

    ;;

esac
