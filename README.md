# Cell-detector
Detection and segmentation of cells on brightfield and phase contrast microscpy images.

## Data generation
The cells are detected using the yolov8 model. Microscopy images can contains up to several hundreds cells which makes the process of labeling a dataset tedious. To bypass this limitation, a dataset of fake images (see bellow) can be generated. 

![fake_data](https://github.com/chagab/Cell-detector/assets/28218716/7fa7e8d5-3e6d-4f1c-a573-8121efaf3577)
*Left: real brightfield image. Right: fake generated brightfield image used for the model training*

## Model training

Training the yolov8-nano model on a small dataset (350 images) for 100 epochs gives satisfying results. On a GPU, training takes around two hours.

## Analysis

The trained segmentation model can be used to analyse migration experiments

![1mT_L_E](https://github.com/chagab/Cell-detector/assets/28218716/c39b4ef1-fda6-46a6-8213-06ebf920dd91)

And viability assays

![6](https://github.com/chagab/Cell-detector/assets/28218716/c233c18e-14ad-43c3-b720-840dd674003d)
