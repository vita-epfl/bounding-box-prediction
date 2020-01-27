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
  * [Project Structure](#requirements)
  * [Proposed method](#brief-project-structure)
  * [Results](#results)
  * [Train](#train)
  * [Test](#test)
  
## Requirements:
------------

  * Python 3
  * Pytorch 1.2.1
  * OpenCV
  * Scikit-Learn
  * Pillow
  
## Project structure:
------------

    ├── bounding-box-prediction         : Project repository
        ├── openpifpaf      
            ├── models.py               : Script of all the relevant models designed during the project.              
            ├── datasets.py             : Script containing the dataset implementation of JAAD as well as the data loader. 
            ├── trainer.py              : Script containing the training and testing steps of the model.
            ├── train.py                : Script to train the model on the JAAD dataset.
            ├── predict.py              : Script to perform predictions on new data.
            ├── show.py                 : Script containing visualization functions for the predictions
            ├── utils.py                : Script containing necessary math and transformation functions.
            ├── Jupyter_notebooks       : Folder containing a group of notebooks corresponding to the performed experiments.
           
 ## Proposed method
 
![Our proposed multitask Position-Speed-LSTM (PS-LSTM) architecture](Images/network.PNG)
