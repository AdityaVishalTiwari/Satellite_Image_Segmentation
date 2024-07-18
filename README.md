# Satellite Image Segmentation
Hugging Face App link: https://huggingface.co/spaces/AdityaOnFire/Satellite_Image_Segmentation
The primary goal is to accurately classify different regions in satellite images into various classes such as buildings, roads, vegetation, water bodies, etc.

## Table of Contents

* Overview
* Satellite Image Segmentation
* Dataset
* Preprocessing
* Model
* Training
* Results
* Dependencies
* Acknowledgements

## Overview
The project focuses on the semantic segmentation of satellite images using a deep learning model. The U-Net model architecture is employed to perform the segmentation task. The model is trained and evaluated on a dataset of satellite images, with each image divided into patches to facilitate efficient training.

## Satellite Image Segmentation
Satellite image segmentation is a process of partitioning a satellite image into multiple segments or regions to simplify its analysis. This technique is crucial in various applications, including urban planning, agriculture monitoring, environmental management, and disaster response. By accurately classifying different regions in satellite images, we can extract valuable information about land use, vegetation cover, water bodies, and built-up areas.

## Dataset
The dataset consists of satellite images and their corresponding masks. The images are categorized into various classes such as buildings, roads, vegetation, water bodies, and unlabeled regions. Each image is split into patches for better processing and model training.
Dataset link: https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery

## Preprocessing
The preprocessing steps involve:

1. Loading and normalizing the images.
2. Converting masks to RGB format.
3. Splitting the images and masks into patches of size 256x256.
4. Scaling the images using MinMaxScaler.
5. Converting RGB masks to categorical labels.

## Model
The U-Net model is used for segmentation, which consists of an encoder (downsampling path) and a decoder (upsampling path). The model is designed to capture both local and global features in the image.

### U-Net Architecture
The U-Net architecture is a popular model for image segmentation tasks. It consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. Here is a visual representation of the U-Net architecture:
![GeeksforGeeks Image](https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg)
Key features of the model:

* Multiple convolutional layers with ReLU activation.
* Dropout layers to prevent overfitting.
* MaxPooling layers for downsampling.
* Conv2DTranspose layers for upsampling.
* Softmax activation in the output layer for multi-class segmentation.

## Training
The model is trained using a combination of Dice Loss and Categorical Focal Loss. Key training parameters include:

* Batch size: 16
* Epochs: 100
* Optimizer: Adam
* Metrics: Accuracy and Jaccard Coefficient
* Early stopping and custom callback functions are used to monitor the training process and visualize the loss and metric graphs.

### Metrics
* Accuracy: Measures the percentage of correctly classified pixels.
* Jaccard Coefficient: Also known as Intersection over Union (IoU), this metric evaluates the overlap between the predicted and ground truth segments.

## Results
The training and validation loss, as well as the Jaccard Coefficient, are plotted to evaluate the model's performance. Random test images and their corresponding predicted masks are displayed to showcase the model's segmentation capabilities.

## Acknowledgements
This project is based on the U-Net architecture and uses the segmentation_models library for loss functions. Thanks to the authors and contributors of these libraries.
