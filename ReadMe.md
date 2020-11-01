# Clothing Category Prediction 

## Description
This project uses the concept of transfer learning to build a model on top of ResNet-50 to predict the clothing category from an image.

## Dataset
I have used the DeepFashion database which is a large-scale fashion database. The dataset can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). Note that I have used the "Category and Attribute Prediction Benchmark" from the DeepFashion dataset.

## Code Modules and How to run
First the folders inside the "Category and Attribute Prediction Benchmark" need to be downloaded inside the *dataset* folder.

Next run the **create-dataset.py** script to split the data into train-val-test data and to prepare the data into an usable format for creating a model. Once the script completes its execution, the data files can be seen inside the *split-data* folder.

Run the **model.ipynb** notebook sequentially to train and test the model.

The *models* subfolder will contain the trained model.

## Author
**Shrey Shah**
[GitHub](https://github.com/imshreyshah)
[LinkedIn](https://www.linkedin.com/in/imshreyshah/)

