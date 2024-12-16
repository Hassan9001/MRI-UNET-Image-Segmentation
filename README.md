# MRI-UNET-Image-Segmentation
 Automated MRI image segmentation with U-NET model architecture to determine kidney and bladder volume for tracking tumor growth.

## Data: 
The original data unfortunately cannot be made public, however a comprable dataset can be used from Timothy Kline's (TLKline) AutoTKV_MouseMRI pachage https://github.com/TLKline/AutoTKV_MouseMRI. Just make sure to organise the data as I did.\
The structure of the dataset is as follows:\
data/volumes/\
 ├──test_volumes/\
 │  ├── img/\
 │  │   ├── MRIvolumes.nii.gz\
 │  │   └── ...\
 │  └── mask/\
 │      ├── MRIvolumes.nii.gz\
 │      └── ...\
 └──training_volumes/\
    ├── img/\
    │   ├── MRIvolumes.nii.gz\
    │   └── ...\
    └── mask/\
        ├── MRIvolumes.nii.gz\
        └── ...\
## UNET Model:
![My Model Structure](https://github.com/Hassan9001/MRI-UNET-Image-Segmentation/blob/main/files/my_model_structure.png)
