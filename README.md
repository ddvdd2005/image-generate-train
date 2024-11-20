# image-generate-train

This project was built to revamp entirely the machine learning infrastructure at SONIA (https://sonia.etsmtl.ca/). This project was started from scratch in order to use more modern libraries like PyTorch 2 instead of the outdated Tensorflow 1.7 that was used on the Xavier.
The main difficulties with this project is the Xavier being a bottleneck for infering images in real times. Thus, the model needs to be as small as possible

## Files in this folder:

createDistortedImage.py: generates images that are distorted in order to be used for a first more general level of training. Then when we are able to collect real images at the competition, we can run a fine-tuning

infer.py: look at the results of the inference of a model on one specific image

infer_best.py: compare the results of a multitude of images on a multitude of models in order to find the best model. Prints out a table allowing us to check this information quickly

infer_tf.py: Due to some complications with updating the Xavier to be able to use Pytorch 2 in 2023 (would have required us to update Ros), this tests that the model trained also works on tensorflow

onnx_to_tf.py: converts an onnx to tf for tensorflow, specifically version 1.7 of tensorflow

pt_to_onnx.py: converts the trained model in an onnx

train.py: the training loop

## How to achieve the best results

Before the competition, use createDistortedImage.py to create a first fine-tuned-ish version of yolov8n
During the competition, manually label around 200-300 varied images in order to be able to fine tune the model on competition data. Use label-studio which allows us to export in coco format for training easily

## Results in 2023

By being the only person doing the manual labelling since so few images were required due to it being a finetuning, we were able to have a more accurate model than other teams, with a very stable bonding box, allowing us to use said bonding box to directly calculate the position and distance of our targets.
Furthermore, we were the only team to train in such a way, other teams were all training using thousands of images while our final trainings were done using 200 images (v1) and then 260 images (v2, addressed specific issues with v1)
