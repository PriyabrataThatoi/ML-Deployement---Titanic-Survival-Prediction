# ML Deployment Configuration

This repository explains the steps how a jupyter notebook is converted into a format that is used during machine learning deployment. The notebook is converted into 3 files.
For this project, I have used TITANIC case study to convert the solution notebook into the below three files.

![ml_lifecycle](https://user-images.githubusercontent.com/54467567/87469164-79800280-c5e0-11ea-8995-b5819280c6d6.png)

### 1. CONFIG FILE: 
       It has information such as link to dataset, variable groups 
       to be used in the preprocessing and modeling step
       
### 2. PREPROCESSING FILE:
       It has information of all the definition required to preprocess the files. 
       It has a pipeline class that contains 3 important functions : FIT, TRANSFORM & PREDICT
       
### 3.  PIPELINE FILE:
       In this file, we will call python utilities such as config.py and preocessing.py 
       to use pipeline class and methods insided it that is FIT, TRANSFORM & PREDICT
