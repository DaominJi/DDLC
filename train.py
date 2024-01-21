import torch
import torch
import torch.nn as nn
import argparse
import glob
from data import DDLCDataset
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
parser.add_argument('--loading', type=int, default=0, help='Whether Load a Saved model for Training')
parser.add_argument('--B', type=int, default=5, help='Training Batch')
parser.add_argument('--num_epochs',type=int, default=60, help='Training Epochs')
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda else "cpu")

model = FCM(args.P1, args.P2, args.H, args.W, args.beta, args.input_dim, args.output_dim, args.num_hidden_units, args.num1_layers, args.num2_layers, args.num_heads, args.num_experts).to_device()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
checkpoint_path = '../checkpoints/ckp.pth'
table_path = '../data/table/'
chart_path= '../data/vis/'

if args.load:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
dataset =   DDLCDataset(table_path, chart_path)
dataloader = DataLoader(dataset, batch_size=args.B, shuffle=True)
    
for epoch in range(args.num_epochs):
    model.train()
    total_loss = 0.0

    for batch_data in dataloader:
        tables, charts = batch_data
        labels = torch.ones((1,len(tables)))
        inputs, labels = semihard(tables, charts)        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print (f'Epoch {epoch}: total_loss')
    if epoch%10 == 0:    
        torch.save(model.state_dict(), 'model.pth')

    
