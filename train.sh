source activate pytorch-gpu
cd ~/xc/Gits/StereoSiamNet/
nohup python train/train_ssn.py > train.log 2>&1 &