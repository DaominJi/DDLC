import torch
import torch.nn as nn
import torch.nn.functional as F

class ChartEncoder(nn.Module):
    def __init__(self, H = 600, W = 800, P1 = 100, dim = 768, num_heads = 8, num_layers = 12):
        super(ChartEncoder, self).__init__()
        self.H = H
        self.W = W
        self.P1 = P1
        
        num_segments = int(W / P1)
        segment_dim = H * P1
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_segments = num_segments
        self.image_to_segment = nn.Conv2d(1, dim, kernel_size = (H, P1), stride = P1)
        
        self.position_embeddings = nn.Parameter(torch.randn(1, num_segments, dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = dim, nhead = num_heads),
            num_layers = num_layers
            )
  
    def forward(self, x): 
        #print (x.shape)
        B, M, H, W = x.shape
        x = x.view(-1, 1, H, W)
        x = self.image_to_segment(x)
        #print (x.shape)
        x = x.view(B * M, self.dim, -1).permute(0, 2, 1)
        #print (x.shape)
        pos = self.position_embeddings.repeat(B*M, 1, 1)
        #print (pos.shape)
        x += pos
        #print (x.shape)
        x = x.view(B * M, self.num_segments, self.dim)
        x = self.transformer_encoder(x)
        #x = x.mean(dim = 1)
        x = x.reshape(B, M, self.num_segments, -1)
        return x
    
    
class HMSRL(nn.Module):  
    def __init__(self, beta=4, P2=64, emb_dim=768, hidden_dim=512):
        super(HMSRL, self).__init__()  # Updated class name
        self.beta = beta
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        #self.P2 = P2
        self.trans = nn.Linear(int(P2/2**beta),emb_dim)
        self.MLPs = nn.ModuleList([nn.Linear(2*emb_dim, emb_dim) for _ in range(self.beta)])

    def forward(self, x):
        B, C, S, D = x.shape
        x = x.view(-1, D)
        x = self.trans(x)
        x = x.view(-1, self.emb_dim)
        for mlp in self.MLPs:
            N, emb_dim = x.shape
            x = x.view(-1, 2*self.emb_dim)
            x = torch.relu(mlp(x))
        
        x = x.view(B, C, -1, self.emb_dim)
        return x

class MoE(nn.Module):
    def __init__(self, beta=4, P2=64, emb_dim = 768, num_experts = 5, hidden_dim=512):
        super(MoE, self).__init__()
        self.P2 = P2
        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([HMSRL(beta, P2, emb_dim, hidden_dim) for i in range(num_experts)])
        self.gates = nn.Sequential(
            nn.Linear(P2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.ReLU()
            )    
    
    def forward(self, x):
        B, C, S, D = x.shape
        x = x.view(-1, self.P2)
        gating_weights = F.softmax(self.gates(x),dim=1)
        x = x.view(B, C, S, D)
        expert_outputs = [expert(x) for expert in self.experts]
        output = torch.stack(expert_outputs, dim=3).view(-1, self.num_experts, self.emb_dim)
        #print (output.shape)
        #print (gating_weights.shape)
        output = torch.einsum('ijk,ij->ik',[output, gating_weights])
        output = output.view(B, C, -1, self.emb_dim)
        return output        
    
class DatasetEncoder(nn.Module):
    def __init__(self, dim=768, num_heads=8, num_layers=12, max_length=4096):
        super(DatasetEncoder,self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.position_embeddings = nn.Parameter(torch.randn(1, max_length, self.dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = self.dim, nhead = num_heads),
            num_layers = num_layers
            )
    
    def forward(self, x):
        B, C, S, _ = x.shape
        x = x.view(B*C, S, self.dim)
        pos = self.position_embeddings.repeat(B*C, 1, 1)[:, 0:S, :]
        x += pos
        x = self.transformer_encoder(x)
        x = x.view(B, C, S, -1)
        return x
    
class Matcher(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=512):
        super(Matcher, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.SLSAN = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = self.emb_dim, nhead = 1),
            num_layers = 1
            )
        self.LCSAN = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = self.emb_dim, nhead = 1),
            num_layers = 1
            )
        self.linear1 = nn.Linear(2*emb_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,1)
        
    def forward(self, x1, x2):
        B, M, S1, D = x1.shape
        B, C, S2, D = x2.shape
        x1 = x1.view(B, M, -1).repeat(1, C, 1)
        x2 = x2.view(B, C, -1).repeat(1, 1, M).view(B, C * M, -1)
        x = torch.concatenate((x1,x2), dim = 2)
        x = x.view(B*C*M, -1, self.emb_dim)
        x = self.SLSAN(x)
        x1 = x[:,0:S1,:]
        x2 = x[:,S1:S1+S2,:]
        x1 = x1.reshape(B, M, -1, self.emb_dim)
        x2 = x2.reshape(B, C, -1, self.emb_dim)
        x1 = x1.mean(dim=2)
        x2 = x2.mean(dim=2)
        #print (x1.shape, x2.shape)
        x = torch.cat((x1, x2),dim=1)
        #x = torch.cat((x1,x2),dim=1)
        x = x.view(B, -1, self.emb_dim)
        x = self.LCSAN(x)
        #print (x.shape)
        x1 = x[:,:M,:]
        x2 = x[:,M:M+C,:]
        #print (x1.shape, x2.shape, M, C)
        x1 = x1.mean(dim=1)
        x2 = x2.mean(dim=1)
        #print(x1.shape,x2.shape)
        x = torch.concat((x1,x2),dim=1)
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    
class FCM(nn.Module):
    def __init__(self, P1=100, P2=64, H=600, W=800, beta=4, num_heads=8, num_layers=12, num_experts=5, emb_dim=768, hidden_dim=512, max_length=4096):
        super(FCM, self).__init__()
        self.moe = MoE(beta, P2, emb_dim, num_experts, hidden_dim)
        self.chartencoder = ChartEncoder(H, W, P1, emb_dim, num_heads, num_layers)
        self.datasetencoder = DatasetEncoder(emb_dim, num_heads, num_layers, max_length)
        self.matcher = Matcher(emb_dim, hidden_dim)
        
    def forward(self, x1, x2):
        x1 = self.chartencoder(x1)
        x2 = self.moe(x2)
        x2 = self.datasetencoder(x2)
        y = self.matcher(x1,x2)

        return y        
        
        
    
    

    
