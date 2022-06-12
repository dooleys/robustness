#!/bin/bash -i

#SBATCH --job-name=face-det-setup
#SBATCH --output=output/face-det-setup.%j.log
#SBATCH --error=output/face-det-setup.%j.err
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16gb

# Set up the models directory
MODELS_DIR=$(pwd)

# =============================================================================================== #
#                                       TinaFace Setup                                            #
# =============================================================================================== #

cat << EOF
======================================================================
                        SETTING UP TinaFace!!!!!
======================================================================
EOF

# Load supported python3 version
module load cuda/11.1.1 gcc/7.5.0

# Move into the tinaface subdirectory
cd tinaface

# Remove corresponding conda environment if it exists
conda env remove -n tinaface
# Create its environment
conda create python=3.6 -n tinaface -y
conda activate tinaface
# Install TensorRT
pip3 install --upgrade setuptools wheel
pip3 install nvidia-pyindex
pip3 install nvidia-tensorrt==7.2.2.1

# Clone volksdep since it is a dependency
git clone https://github.com/Media-Smart/volksdep.git
# Change the dependency from tensorrt to nvidia-tensorrt since that is wrong anyways
sed -i 's/tensorrt/nvidia-tensorrt/g' volksdep/setup.py
# Install volksdep
pip3 install volksdep/

# Download PyTorch and torchvision
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install tqdm for progress bar
pip3 install tqdm

# Install gdown to download from Google Drive
pip3 install gdown

# Install pandas
pip3 install pandas

# Install flexinfer for TinaFace
pip3 install "git+https://github.com/Media-Smart/flexinfer.git"

# Download the public TinaFace checkpoint
gdown https://drive.google.com/uc?id=1VkMKWPJM0oaS8eyIVZ5flcJH70Pbi-_g

# Reset for next model
conda deactivate
module rm cuda/11.1.1 gcc/7.5.0
cd $MODELS_DIR

cat << EOF
======================================================================
                        FINISHED TinaFace!!!!!
======================================================================
EOF

# =============================================================================================== #
#                                       YOLOv5 Setup                                              #
# =============================================================================================== #

cat << EOF
======================================================================
                        SETTING UP YOLOv5!!!!!
======================================================================
EOF

# Clone its contained repo
git clone https://github.com/deepcam-cn/yolov5-face.git
cd yolov5-face

# Remove corresponding conda environment if it exists
conda env remove -n yolov5
# Create its environment
conda create python=3.6 -n yolov5 -y
conda activate yolov5

# Download PyTorch and torchvision
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip3 install matplotlib opencv-python pillow pyyaml requests seaborn thop tqdm

# Install gdown to download from Google Drive
pip3 install gdown

# Install pandas
pip3 install pandas

# Download the YOLOv5-l checkpoint
gdown https://drive.google.com/uc?id=16F-3AjdQBn9p3nMhStUxfDNAE_1bOF_r -O weights/

# Copy inferrence wrapper
cp $MODELS_DIR/yolov5_infer.py .

# Reset for next model
conda deactivate
cd $MODELS_DIR

cat << EOF
======================================================================
                        FINISHED YOLOv5!!!!!
======================================================================
EOF

# =============================================================================================== #
#                                       MogFace Setup                                             #
# =============================================================================================== #

cat << EOF
======================================================================
                        SETTING UP MogFace!!!!!
======================================================================
EOF

# Load CUDA 10.2
module load cuda/10.2.89
# Load corresponding cuDNN version
module load cudnn/v8.2.1

# Clone its containing repo
git clone https://github.com/idstcv/MogFace.git
cd MogFace

# Remove corresponding conda environment if it exists
conda env remove -n mogface
# Create its environment
conda create python=3.6 -n mogface -y
conda activate mogface

# Install dependencies
pip3 install -r requirements.txt

# Build NMS extensions
cd utils/nms && python setup.py build_ext --inplace && cd ../..
# Build bounding box extensions
cd utils/bbox && python setup.py build_ext --inplace && cd ../..

# Install gdown to download from Google Drive
pip3 install gdown

# Install pandas
pip3 install pandas

# Download the MogFace checkpoint
gdown https://drive.google.com/uc?id=1s8LFXQ5-zsSRJKVHLFqmhow8cBn4JDCC -O weights/

# Copy inferrence wrapper
cp $MODELS_DIR/mogface_infer.py .

# Reset environment
module rm cudnn/v8.2.1 cuda/10.2.89
conda deactivate

cat << EOF
======================================================================
                        FINISHED MogFace!!!!!
======================================================================
EOF
