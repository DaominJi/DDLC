# Overview
This is the repository of source code for paper `Dataset Discovery via Line Charts`, whose structure is shown as follows:

sample_data/: sample data

data.py: data loader for model training

model.py, train.py, eval.py: model definition, training, and, evaluation

helper.py: helpful functions

requirements: python dependencies

# DDLC Bechmark
The new benchmark DDLC created for dataset discovery via line charts is released, which can be accessed through [https://zenodo.org/record/10547361](https://zenodo.org/record/10547361) 

DDLC consists of the following directories or files:

training: consists of two directories, `table` and `vis`. The former contains the all the tables for training and the latter contains all the charts for training. Each pair of table and chart have the same id.For example, '1.csv' and '1.png' is a pair of positive training instance.

query: consists of 400 line chart queries.

repository: consisits of around 19,000 candidate tables.

ground_truth: consists of the ground-truth table ids for each query id. The table id is also the file name of the table in the repository.

# Training and Evaluation
Please run the commands `python train.py` and `python eval.py` to train and evaluation the model by default parameter settings.



