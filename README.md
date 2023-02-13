
# Pedestrian bounding box prediction library: <br/>
This library contains two research projects for bounding box predictions in 2D and 3D
---

These two codes will be merged soon to make a unified library:

> __Pedestrian 3D Bounding Box Prediction, hEART 2022__<br /> 
>  S. Saadatnejad, Y. Ju, A. Alahi <br /> 
>  __[Paper](https://arxiv.org/abs/2206.14195)__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; __[Citation](https://github.com/vita-epfl/bounding-box-prediction#for-citation)__   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  __[Code](https://github.com/vita-epfl/bounding-box-prediction/tree/master/3D)__
     


> __Pedestrian Intention Prediction: A Multi-task Perspective, hEART 2020__<br /> 
> S. Bouhsain, S. Saadatnejad, A. Alahi <br /> 
> __[Paper](https://arxiv.org/abs/2010.10270)__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; __[Citation](https://github.com/vita-epfl/bounding-box-prediction#for-citation)__  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  __[Code](https://github.com/vita-epfl/bounding-box-prediction/tree/master/2D)__
     
# Pedestrian Intention Prediction: A Multi-task Perspective

## _Absract_:
In order to be globally deployed, autonomous cars must guarantee the safety of pedestrians. This is the reason why forecasting pedestrians' intentions sufficiently in advance is one of the most critical and challenging tasks for autonomous vehicles.
This work tries to solve this problem by jointly predicting the intention and visual states of pedestrians.
In terms of visual states, whereas previous work focused on x-y coordinates, we will also predict the size and indeed the whole bounding box of the pedestrian.
The method is a recurrent neural network in a multi-task learning approach. It has one head that predicts the intention of the pedestrian for each one of its future position and another one predicting the visual states of the pedestrian.
Experiments on the JAAD dataset show the superiority of the performance of our method compared to previous works for intention prediction.
Also, although its simple architecture (more than 2 times faster), the performance of the bounding box prediction is comparable to the ones yielded by much more complex architectures.

## Introduction:
This is the official code for the paper ["Pedestrian Intention Prediction: A Multi-task Perspective"](https://arxiv.org/abs/2010.10270), accepted and published in [hEART 2021](http://www.heart-web.org/) (the 9th Symposium of the European Association for Research in Transportation).

## Contents
------------
  * [Repository Structure](#repository-structure)
  * [Proposed Method](#proposed-method)
  * [Results](#results)
  * [Installation](#installation)
  * [Dataset](#dataset)
  * [Training/Testing](#training-testing)
  * [Tested Environments](#tested-environments)
  
## Repository structure:
------------
    ├── 2D                              : Project repository
            ├── prepare_data.py         : Script for processing raw JAAD data.
            ├── train.py                : Script for training PV-LSTM.  
            ├── test.py                 : Script for testing PV-LSTM.  
            ├── DataLoader.py           : Script for data pre-processing and loader. 
            ├── networks.py             : Script containing the implementation of the network.
            ├── utils.py                : Script containing necessary math and transformation functions.
            
## Proposed method
-------------
![Our proposed multitask Position-Speed-LSTM (PV-LSTM) architecture](Images/network.PNG)


## Results
--------------
![Example of outputs](Images/visualizations.png)
  
## Installation:
------------
Start by cloning this repositiory:
```
git clone https://github.com/vita-epfl/bounding-box-prediction.git
cd bounding-box-prediction
```
Create a new conda environment (Python 3.7):
```
conda create -n pv-lstm python=3.7
conda activate pv-lstm
```
And install the dependencies:
```
pip install -r requirements.txt
```

## Dataset:
  
  * Clone the dataset's [repository](https://github.com/ykotseruba/JAAD).
  ```
  git clone https://github.com/ykotseruba/JAAD
  ```
  * Run the `prepare_data.py` script, make sure you provide the path to the JAAD repository and the train/val/test ratios (ratios must be in [0,1] and their sum should equal 1.
  ```
  python3 prepare_data.py |path/to/JAAD/repo| |train_ratio| |val_ratio| |test_ratio|
  ```
  * Download the [JAAD clips](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) (UNRESIZED) and unzip them in the `videos` folder.
  * Run the script `split_clips_to_frames.sh` to convert the JAAD videos into frames. Each frame will be placed in a folder under the `scene` folder. Note that this takes 169G of space.
  
  
## Training/Testing:
Open `train.py` and `test.py` and change the parameters in the args class depending on the paths of your files.
Start training the network by running the command:
```
python3 train.py
```
Test the trained network by running the command:
```
python3 test.py
```

## Tested Environments:
------------
  * Ubuntu 18.04, CUDA 10.1
  * Windows 10, CUDA 10.1



### Citation

```
@inproceedings{bouhsain2020pedestrian,
title={Pedestrian Intention Prediction: A Multi-task Perspective},
 author={Bouhsain, Smail and Saadatnejad, Saeed and Alahi, Alexandre},
  booktitle = {European Association for Research in Transportation  (hEART)},
  year={2020},
}
```



# Pedestrian 3d Bounding Box Prediction


## Abstract
Safety is still the main issue of autonomous driving, and in order to be globally deployed, they
need to predict pedestrians’ motions sufficiently in advance. While there is a lot of research on
coarse-grained (human center prediction) and fine-grained predictions (human body keypoints
prediction), we focus on 3D bounding boxes, which are reasonable estimates of humans without
modeling complex motion details for autonomous vehicles. This gives the flexibility to predict in
longer horizons in real-world settings. We suggest this new problem and present a simple yet effective
model for pedestrians’ 3D bounding box prediction. This method follows an encoder-decoder
architecture based on recurrent neural networks, and our experiments show its effectiveness in
both the synthetic (JTA) and real-world (NuScenes) datasets. The learned representation has useful
information to enhance the performance of other tasks, such as action anticipation.

This project uses the [Joint Track Auto (JTA)](https://github.com/fabbrimatteo/JTA-Dataset) and the [NuScenes](https://www.nuscenes.org/) datasets.


## Introduction:
This is the official code for the paper ["Pedestrian 3D Bounding Box Prediction"](https://arxiv.org/abs/2010.10270), accepted and published in [hEART 2022](http://www.heart-web.org/) (the 10th Symposium of the European Association for Research in Transportation).


## Repository structure
```
|─── 3D                                        : Project repository
      |─── exploration                         : Jupyter notebooks for data exploration and visualization
            |─── JTA_exploration.ipynb   
            |─── NuScenes_exploration.ipynb
      |─── preprocess                          : Scripts for preprocessing
            |─── jta_preprocessor.py
            |─── nu_preprocessor.py
            |─── split.py
      |─── utils                               : Scripts containing necessary calculations
            |─── utils.py  
            |─── nuscenes.py
      |─── visualization                       : Scripts for visualizing the results and making GIFs
            |─── visualize.py
      |─── Dataloader.py                       : Script for loading preprocessed data
      |─── network.py                          : Script containing network 
      |─── network_pos_decoder.py              : Script containing network variation that has a position decoder (not used)
      |─── test.py                             : Script for testing
      |─── train.py                            : Script for training 
```

## Proposed network
![](./images/network_diag.png)

## Results
![JTA](./images/test_seq_478_frame177_idx17500.gif)
![NuScenes](./images/test_scene-0593_frame0_idx35.gif)

## Setup
Please install the required dependencies from the <requirements.txt> file.
For Nuscenes, clone the ```nuscenes-devkit``` repository from [here](https://github.com/nutonomy/nuscenes-devkit). The scripts in the folder <nuscenes-devkit/python-sdk/nuscenes> are required, so it is recommended to copy this folder to the ```3d-bounding-box-prediction```, otherwise path dependencies may need to be updated.

## Preprocessing
The input, output, stride, and skip parameters of the loaded dataset can be set the in the '''args''' class.
To load the datasets, first run the preprocessing scripts, then ```Dataloader.py```.

**Note** Due to the large number of samples in the JTA dataset, the preprocessing script first saves files containing all available samples to a file titled "Preprocesed annotations". This data can then be read by the ```Dataloader.py``` file to get sequences of bounding boxes that are passed to the network. For Nuscenes the sequences are generated directly during preprocessing.

## Jupyter notebooks
The Jupyter notebooks provided demonstrate how all the code in this repository can be used, including data loading, training, testing, and visualization.


## Tested Environments:
------------
  * Ubuntu 18.04, CUDA 10.1
  * Windows 10, CUDA 10.1

### Citation
```
@inproceedings{saadatnejad2022pedestrian,
title={Pedestrian 3D Bounding Box Prediction},
 author={Saadatnejad, Saeed and Ju, Yi Zhou and Alahi, Alexandre},
  booktitle = {European Association for Research in Transportation  (hEART)},
  year={2022},
}
```