import torch
import torch
import torch.nn as nn
import argparse
import glob
from data import DDLC, collate_batch
from helper import Semihard
import pandas as pd
import random
import numpy as np
from model import FCM
import torch.optim as optim
from torch.utils.data import DataLoader
from helper import semihard

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
parser.add_argument('--load', type=int, default=0, help='Whether Load a Saved model for Training')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch_size')
parser.add_argument('--num_epochs',type=int, default=60, help='Training Epochs')
parser.add_argument('--cuda',type=int, default=0, help='Whether using Cuda for Acceleration')
parser.add_argument('--lr',type=float,default=1e-5,help='Learning Rate')
parser.add_argument('--max_length',type=int,default=4096,help='max number of smallest segments')
parser.add_argument('--neg',type=int,default=3,help='negative sampling size')
parser.add_argument('--selection',type=str,default='rand',help='negative sampling strategy')
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda else "cpu")

model = FCM(args.P1, args.P2, args.H, args.W, args.beta, args.num_heads, args.num_layers, args.num_experts, args.emb_dim, args.hidden_dim, args.max_length).to_device()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
checkpoint_path = '../checkpoints/ckp.pth'
table_dir = '../data/table/'
chart_dir= '../data/vis/'

if args.load:
    print ('The FCM model is recovered from a checkpoint...')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
dataset =  DDLC(table_dir, chart_dir)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: collate_batch(x, args.neg, 4, 64, args.selection))

model.train()
for epoch in range(args.num_epochs):
    total_loss = 0.0

    for batch_data in dataloader:
        charts, tables, labels = batch_data
        charts, tables, labels = charts.to(device), tables.to(device), labels.to(device)
        outputs = model(charts, tables)       
        optimizer.zero_grad()

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print (f'Epoch {epoch}: total_loss')
    if epoch%10 == 0:    
        torch.save(model.state_dict(), 'ckp.pth')

    
