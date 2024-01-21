import torch
import numpy as np
from model import FCM
from helper import ndcg_at_k, precision_at_k
import argparse
import pandas as pd
import time 
from PIL import Image
import os

parser = argparse.ArgumentParser(description='Process input parameters.')

parser.add_argument('--P1', type=int, default=80, help='Parameter P1')
parser.add_argument('--P2', type=int, default=160, help='Parameter P2')
parser.add_argument('--H', type=int,  default=600, help='Height')
parser.add_argument('--W', type=int,  default=800, help='Width')
parser.add_argument('--beta', type=float,  default=5, help='Parameter Beta')
parser.add_argument('--input_dim', type=int, default=768, help='Input Dimension')
parser.add_argument('--output_dim', type=int, default=64, help='Output Dimension')
parser.add_argument('--num_hidden_units', type=int, default=512, help='Number of Hidden Units')
parser.add_argument('--num1_layers', type=int, default=2, help='Number of Layers in MoE')
parser.add_argument('--num2_layers', type=int,  default=12, help='Number of Layers in Network 2')
parser.add_argument('--num_heads', type=int,  default=8, help='Number of Heads')
parser.add_argument('--num_experts', type=int,  default=5, help='Number of Experts in MoE')
args = parser.parse_args()

model = FCM(args.P1, args.P2, args.H, args.W, args.beta, args.input_dim, args.output_dim, args.num_hidden_units, args.num1_layers, args.num2_layers, args.num_heads, args.num_experts)
saved_path = './checkpoints/model.pth'
checkpoint = torch.load(saved_path)
model.load_state_dict(checkpoint['model_state_dict'])

query_path = './query/'
repo_path = './repo/'
query_files = os.listdir(query_path)
repo_files = os.listdir(repo_path)
gt_path = './ground_truth.csv'
ground_truth = pd.read_csv(gt_path)

prec = np.zeros(len(query_files))
ndcg = np.zeros(len(query_files))

for i, query in enumerate(query_files):
    chart = Image.open(query)
    list = []
    for j, table in enumerate(repo_files):
        t = pd.read_csv(table)
        list.append(model(chart, t))
    topk_list, idx = sorted(list, reverse=True)[:50]
    gt = ground_truth[i].tables_id
    prec[i] = precision_at_k(gt, topk_list, 50)
    ndcg[i] = ndcg_at_k(gt, topk_list, 50)

print (f'prec:{prec.mean()}')
print (f'ndcg:{ndcg.mean()}')

