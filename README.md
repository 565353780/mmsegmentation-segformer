# mmsegmentation-segformer

## Install MMSegmentation
```bash
conda create -n seg python=3.8 -y
conda activate seg
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmsegmentation timm ipython attr
git clone https://github.com/NVlabs/SegFormer.git
cd SegFormer
pip install -e . --user
cd ..
git clone ssh://git@chLi:30001/mine/mmsegmentation-segformer.git
```

# Copy models and datasets
```bash
cd mmsegmentation-segformer
./init.sh
```

# Run detect
```bash
python MMSegmentationDetector.py
```

# Run train
```bash
python MMSegmentationTrainer.py
```
