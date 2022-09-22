# conda create -n mmlab python=3.8 -y
# conda activate mmlab

pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113

pip install -U openmim
mim install mmcv-full

pip install mmsegmentation timm opencv-python

cd ..
git clone https://github.com/NVlabs/SegFormer.git
cd SegFormer
pip install -e . --user
