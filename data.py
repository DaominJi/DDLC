from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torch

class DDLCDataset(Dataset):
    def __init__(self, chart_dir, table_dir, transform=None):
        self.chart_dir = chart_dir
        self.table_dir = table_dir
        self.transform = transform
        self.charts = self.load_charts(chart_dir)
        self.tables = self.load_tables(table_dir)

    def load_chart(self, chart_dir):
        charts = []
        for chart in os.listdir(chart_dir):
            chart_path = os.path.join(chart_dir)
            if os.path.isdir(chart_path):
                charts.append(chart_path)
    
    def load_table(self, table_dir):
        tables = []
        for table in os.listdir(table_dir):
            table_path = os.path.join(table_dir)
            if os.path.isdir(table_path):
                tables.append(table_path)
                

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        chart_path = self.charts[idx]
        chart = Image.open(chart_path[idx]).convert('grey')
        if self.transform:
            chart = self.transform(chart)
        
        table_path = self.tables[idx]
        table = pd.read_csv(table_path[idx])
        return (chart, torch.tensor(table.values))
    
