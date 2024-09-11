*Part of the code is being organized and will be uploaded within 24 hours*  
# Requirement
CUDA 11.4  
Python 3.8.8  
Pytorch 1.12.1  
Torchvision 0.13.1  
batchgenerators 0.25  
SimpleITK 2.2.1  

# Usage
## Installation
* install nnUNet as [nnUNet](https://github.com/MIC-DKFZ/nnUNet) instructed
* replace corresponding file from *package* to nnUNet

# Dataset
Download AISD from [here](https://github.com/griffinliang/aisd), and ISLES2018 from [here](https://www.isles-challenge.org/ISLES2018/), 
preprocess CT image first following the instruction in our manuscript, and then using the preprocess codes in nnUNet/nnunet/dataset_conversion.  

# Training
use package/run/run_training.py  
```python
nnUNet_train -network -network_trainer -task -fold
```
for the pre-training of teacher network, use *nnUNetTrainer_SL*, for the training of student network, *nnUNetTrainerV2_Stroke* is used.

# Acknowledgements
Thanks Fabian Isensee and their team for nnUNet, which is reused in this project.
