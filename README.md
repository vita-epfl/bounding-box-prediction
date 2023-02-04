
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