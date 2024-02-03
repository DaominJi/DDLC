import torch
import numpy as np
from model import FCM
from helper import ndcg_at_k, precision_at_k
import argparse
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import torch.nn.functional as F
import math

parser = argparse.ArgumentParser(description='Process input parameters.')

parser.add_argument('--P1', type=int, default=80, help='Parameter P1')
parser.add_argument('--P2', type=int, default=160, help='Parameter P2')
parser.add_argument('--H', type=int,  default=600, help='Height of a Chart Image')
parser.add_argument('--W', type=int,  default=800, help='Width of a Chart Image')
parser.add_argument('--beta', type=float,  default=4, help='Parameter Beta')
parser.add_argument('--emb_dim', type=int, default=768, help='Embedding Size')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden Units Size')
parser.add_argument('--num_heads', type=int,  default=8, help='Number of Heads')
parser.add_argument('--num_experts', type=int,  default=5, help='Number of Experts in MoE')
parser.add_argument('--cuda',type=int, default=0, help='Whether using Cuda for Acceleration')
parser.add_argument('--max_length',type=int,default=4096,help='max number of smallest segments')
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda else "cpu")

model = FCM(args.P1, args.P2, args.H, args.W, args.beta, args.num_heads, args.num_layers, args.num_experts, args.emb_dim, args.hidden_dim, args.max_length).to(device)
saved_path = './checkpoints/ckp.pth'
checkpoint = torch.load(saved_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

query_path = './query/'
repo_path = './repo/'
query_files = os.listdir(query_path)
repo_files = os.listdir(repo_path)
gt_path = './ground_truth.csv'
ground_truth = pd.read_csv(gt_path)['TID']

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
    ])

prec = np.zeros(len(query_files))
ndcg = np.zeros(len(query_files))

for i, query in enumerate(query_files):
    chart = transforms(Image.open(query)).to(device)
    lists = []
    for j, table in enumerate(repo_files):
        t = pd.read_csv(table).select_dtypes(include=['number'])
        t = torch.Tensor(t.to_numpy())
        pad_length = math.ceil(t.shape[0] / args.P2) * args.P2
        t = F.pad(t, pad=(0,pad_length-t.shape[0],0,0),mode='constant',value=0)
        t = t.view(1, t.shape[1], -1, int(args.P2/2**args.beta)).to(device)
        
        lists.append(model(chart, t))
        
    topk_list, idx = sorted(list, reverse=True)[:50]
    gt = ground_truth[i]
    prec[i] = precision_at_k(gt, topk_list, 50)
    ndcg[i] = ndcg_at_k(gt, topk_list, 50)

print (f'prec:{prec.mean()}')
print (f'ndcg:{ndcg.mean()}')

