# Overview
This is the repository of source code for paper `Dataset Discovery via Line Charts`, whose structure is shown as follows:

Sample_Data/: sample data used in quick-start examples

data.py: dataset definitions and padding utilities

model.py: chart/table encoders and the dual-encoder retrieval model

train.py: contrastive training loop

eval.py: retrieval evaluation against a ground-truth file

helper.py: metric helpers and optimisation utilities

requirements.txt: Python dependencies

# DDLC Bechmark
The new benchmark DDLC created for dataset discovery via line charts is released, which can be accessed on zenodo platform [https://zenodo.org/records/10577906](https://zenodo.org/records/10577906).

DDLC consists of the following directories or files:

training: consists of two directories, `table` and `vis`. The former contains the all the tables for training and the latter contains all the charts for training. Each pair of table and chart have the same id.For example, '1.csv' and '1.png' is a pair of positive training instance.

query: consists of 400 line chart queries.

repository: consisits of around 19,000 candidate tables.

ground_truth: consists of the ground-truth table ids for each query id. The table id is also the file name of the table in the repository.

# Training and Evaluation

Install the dependencies with `pip install -r requirements.txt` and then train the model:

```
python train.py --chart_dir Sample_Data --table_dir Sample_Data --epochs 20
```

This trains a contrastive model that aligns chart and table embeddings. Checkpoints are written to `checkpoints/ddlc_model.pt` by default.

To evaluate retrieval quality, prepare a JSON file that maps query ids to a list of relevant table ids (e.g. `{ "1": ["1", "42"] }`). Run:

```
python eval.py \
  --checkpoint checkpoints/ddlc_model.pt \
  --query_dir path/to/query/charts \
  --repository_dir path/to/repository/tables \
  --ground_truth path/to/ground_truth.json \
  --topk 10
```

The script reports Precision@k and NDCG@k across all queries.



