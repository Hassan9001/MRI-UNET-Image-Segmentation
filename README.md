# MRI-UNET-Image-Segmentation
 Automated MRI image segmentation with U-NET model architecture to determine kidney and bladder volume for tracking tumor growth.

## Navigating Repository:
	1. files/dicom2nifti.ipynb (Optional): convert DICOM to NIFTI (not needed if using TLKline's dataset)
	2. Interactive_Data_Viewer.ipynb: Interactive Image & Mask volume Viewer with widget slider.
	3. dataprep.ipynb: (1) Saving 2D image slices from 3D volumes. (2) Augmenting and saving Training Data.
	4. data_generator.py: Dataset generator class for feeeding images and masks into model (also includes some Preproc & Augmentation)
	5. Main.ipynb: Train and test the model. Save predictions in results folder.
	6. model/model_code.py: Contains the code for building the UNet architecture.
	7. model/mertrics.py: Contains the code for dice coefficient metric and dice coefficient loss.

## Data: 
The original data unfortunately cannot be made public, however a comprable dataset can be used from Timothy Kline's AutoTKV_MouseMRI pachage https://github.com/TLKline/AutoTKV_MouseMRI. Just make sure to organise the data as I did.
The structure of the dataset is as follows:
```
data/volumes/
 ├── test_volumes/
 │      ├── img/
 │      │      ├── sample.nii.gz
 │      │      └── ...
 │      └── mask/
 │             ├── sample.nii.gz
 │             └── ...
 └── training_volumes/
    	├── img/
    	│      ├── sample.nii.gz
    	│      └── ...
	└── mask/
		├── sample.nii.gz
		└── ...
```
## UNET Model:
![My Model Structure](https://github.com/Hassan9001/MRI-UNET-Image-Segmentation/blob/main/files/my_model_structure.png)

