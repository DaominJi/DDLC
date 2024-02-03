from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import torch
import math
import torch.nn.functional as F
from helper import rand_select, hard_select, semihard_select, easy_select

class DDLC(Dataset):
    def __init__(self, chart_dir, table_dir):
        self.chart_dir = chart_dir
        self.table_dir = table_dir
        self.charts = self.load_charts(chart_dir)
        self.tables = self.load_tables(table_dir)
        assert (len(self.charts)==len(self.tables)), "numbers of training instances of tables and charts should be equal"

    def load_charts(self, chart_dir):
        charts = []
        num = len(os.listdir(chart_dir))
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5])
            ])
        for i in range(num):
            lines_dir = chart_dir + f'{i}/'
            lines = []
            for line_path in os.listdir(lines_dir):
                line = transform(Image.open(lines_dir+line_path))
                lines.append(line)
            lines = torch.stack(lines,dim=0)
            charts.append(lines)
        return charts
    
    def load_tables(self, table_dir):
        tables = []
        num = len(os.listdir(table_dir))
        for i in range(num):
            table_path = table_dir + f'{i}.csv'
            table = pd.read_csv(table_path).select_dtypes(include=['number'])
            tables.append(torch.Tensor(table.to_numpy()))
        return tables          

    def __len__(self):
        return len(self.tables)
    
    def __getitem__(self, idx):
        chart = self.charts[idx]
        table = self.tables[idx]
        return chart, table
    
def collate_batch(batch, neg, beta, P2, selection='semihard'):
    input_charts = []
    input_tables = []
    
    max_line_num = 0
    max_col_num = 0
    max_row_num = 0
    
    for chart, table in batch:
        #print (chart.shape, table.shape)
        max_line_num = max(max_line_num, len(chart))
        input_charts.append(chart)
        max_col_num = max(max_col_num, table.shape[1])
        max_row_num = max(max_row_num, table.shape[0])
        input_tables.append(table)
    max_row_num = math.ceil(max_row_num / P2) * P2
    
    batch_size = len(input_charts)
    
    if selection == 'rand':
        #print ('rand')
        input_charts, input_tables = rand_select(input_charts, input_tables, neg)
    elif selection == 'hard':
        #print ('hard')
        input_charts, input_tables = hard_select(input_charts, input_tables, neg)
    elif selection == 'semihard':
        #print ('semihard')
        input_charts, input_tables = semihard_select(input_charts, input_tables, neg)
    else:
        #print ('easy')
        input_charts, input_tables = easy_select(input_charts, input_tables, neg)
    
    print (max_line_num)
    for i in range(len(input_charts)):
        input_charts[i] = F.pad(input=input_charts[i], pad=(0,0,0,0,0,0,0,max_line_num-len(input_charts[i])), mode='constant',value=0)
    for i in range(len(input_charts)):
        input_tables[i] = F.pad(input=input_tables[i], pad=(0,max_row_num-input_tables[i].shape[1],0,max_col_num-input_tables[i].shape[0]), \
            mode='constant', value=0)
    input_charts = torch.stack(input_charts,dim=0)
    input_tables = torch.stack(input_tables,dim=0)
    
    input_charts = torch.squeeze(input_charts, dim=2)
    input_tables = input_tables.view(batch_size*(1+neg), max_col_num, -1, int(P2/2**beta))
    
    labels = torch.ones(batch_size*(1+neg))
    labels[batch_size:] = 1
    
    return input_charts, input_tables, labels
                
        
        
        
    
