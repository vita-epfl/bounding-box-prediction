# Pedestrian Intention Forecasting: A Future Bounding Box Prediction Approach

## _Absract_:
In order to be globally deployed, self driving cars must guarantee safety. This is the
reason why forecasting pedestrians intentions is one of the most important and challenging
tasks for autonomous vehicles. This work proposes an approach based on predicting future
bounding boxes of pedestrians.  The method is a multitask learning recurrent neural
network, with one head that predicts the future velocities of the pedestrian, and another
predicting the action of the latter for each one of its future positions. Experiments on the
datasets JAAD and Citywalks show that the performance of our method is better than
previous works for bounding box prediction. Also, although its simple architecture, the
performance of the intention prediction is comparable to the ones yielded by much more
complex architectures

## Contents
------------
  * [Requirements](#requirements)
  * [Repository Structure](#repository-structure)
  * [Proposed Method](#proposed-method)
  * [Results](#results)
  * [Setup](#setup)
  
## Requirements:
------------

  * Python 3
  * Pytorch 1.2.1
  * OpenCV
  * Scikit-Learn
  * Pillow
  
## Repository structure:
------------

    ├── bounding-box-prediction         : Project repository    
            ├── Multi-Task-PV-LSTM.py   : Script for training and testing PV-LSTM.              
            ├── DataLoader.py           : Script for data pre-processing and loader. 
            ├── networks.py             : Script containing the implementation of the network.
            ├── utils.py                : Script containing necessary math and transformation functions.
           
 ## Proposed method
 -------------
 
![Our proposed multitask Position-Speed-LSTM (PS-LSTM) architecture](Images/network.PNG)


## Results
--------------

![Example of output 1](Images/vis1.png)
![Example of output 1](Images/vis2.png)
![Example of output 1](Images/vis3.png)

## Setup
-------------
* Create anaconda environment and install the required libraries
```
conda create -n env python=3.6 anaconda
conda activate env
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install opencv-python
pip install scikit-learn
pip install pillow
```
Download the github repository
```
git clone https://github.com/vita-epfl/bounding-box-prediction.git
cd bounding-box-prediction
```
* Download the JAAD clips (UNRESIZED) and unzip them in the `videos` folder.
* Run the script `split_clips_to_frames.sh` to convert the JAAD videos into frames. Each frame will be placed in a folder under the `scene` folder. Note that this takes 169G of space.
* Download and unzip the JAAD annotation files into the `annotations` folder.
* Since there isn't yet a command line interface parser, you need to open the file you want to run and modify the args class in order to specify the relevant paths and hyperparameters.
