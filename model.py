import torch
import torch.nn as nn
import torch.nn.functional as F

class ChartEncoder(nn.Module):
    def __init__(self, H, W, P1, dim, num_heads, num_layers):
        super(ChartEncoder, self).__init__()
        self.H = H
        self.W = W
        self.P1 = P1
        
        num_segments = W / P1
        segment_dim = H * P1
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.image_to_segment = nn.Conv2d(1, segment_dim, kernel_size = (H, P1), stride = P1)
        
        self.position_embeddings = nn.Parameter(torch.randnn(1, num_segments+1, dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = dim, nhead = num_heads),
            num_layers = num_layers
            )
  
    def forword(self, x):
        B, C, H, W = x.shape
        x = self.image_to_patch()
        x = self.image_to_segment()
        B,C,H,W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x += self.position_embeddings
        x = self.transformer_encoder(x)
        x = x.mean(dim = 1)
        return x
    
    
class HMSRL(nn.Module):  
    def __init__(self, beta, input_dim, output_dim, num_layers, num_hidden_units):
        super(HMSRL, self).__init__()  # Updated class name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.beta = beta

        self.tree_structure = self.construct_binary_tree(input_dim, num_layers)

        self.mlp = nn.ModuleList([nn.Linear(2 * input_dim, num_hidden_units), nn.ReLU()])
        for _ in range(num_layers - 1):
            self.mlp.extend([nn.Linear(num_hidden_units, num_hidden_units), nn.ReLU()])
        self.mlp.append(nn.Linear(num_hidden_units, output_dim))

    def construct_binary_tree(self, input_dim, num_layers):
        tree_structure = []

        def build_tree(level, node_idx, left, right):
            if level == num_layers:
                return
            mid = (left + right) // 2
            tree_structure.append((level, node_idx, left, right))
            build_tree(level + 1, 2 * node_idx, left, mid)
            build_tree(level + 1, 2 * node_idx + 1, mid, right)

        build_tree(0, 0, 0, input_dim)
        return tree_structure

    def forward(self, x):
        e = []

        for level, node_idx, left, right in self.tree_structure:
            left_child = e[left]
            right_child = e[right] if right < len(e) else torch.zeros_like(left_child)
            concatenated = torch.cat([left_child, right_child], dim=-1)
            node_representation = self.mlp[level](concatenated)
            e.append(node_representation)

        return e[-1]  

class MoE(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts):
        super(MoE, self).__init__()
        self.expert_dim = expert_dim
        self.num_experts = num_experts

        self.expert_layers = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])

        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)  # Apply softmax to get gating coefficients
        )

    def forward(self, input_segments):
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.expert_layers[i](input_segments[i]))

        expert_outputs = torch.stack(expert_outputs, dim=1)  # Stack outputs along the experts dimension
        gating_weights = self.gating_network(input_segments[0])  # Use the first input segment for gating

        mixed_output = torch.sum(expert_outputs * gating_weights.unsqueeze(2), dim=1)

        return mixed_output
    
class DatasetEncoder(nn.Module):
    def __init__(self, P2, beta, input_dim, output_dim, num_hidden_units, num1_layers, num2_layers, num_heads, num_experts):
        super(DatasetEncoder,self).__init__()
        self.transformation = nn.Linear(1,input_dim)
        self.HMSRL = nn.Linear(beta, input_dim, output_dim, num1_layers, num_hidden_units)
        self.moe = MoE(input_dim, output_dim, num_experts)
        self.dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num2_layers
        
        self.position_embeddings = nn.Parameter(torch.randnn(1, 512, self.dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = self.dim, nhead = num_heads),
            num_layers = num2_layers
            )
    
    def forward(self, x):
        B, C, S = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.transformation(x)
        x = x.view(-1, C, S)
        x = self.HMSRL(x)
        x = x.view(B,-1,C)
        x = self.moe(x)
        x += self.position_embeddings
        x = self.transformer_encoder(x)
        x = x.mean(dim = 1)
        return x

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        attention_scores = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, value)

        return output
    
class Matcher(nn.Module):
    def __init__(self, hidden_dim):
        super(Matcher, self).__init__()
        self.hidden_dim = hidden_dim
        self.SLSAN = SelfAttention(self.hidden_dim)
        self.LCSAN = SelfAttention(self.hidden_dim)
        self.mlp = nn.functional.ReLU(nn.Linear(2*hidden_dim, 1))
        
    def forward(self,x1, x2):
        B1, L, S1 = x1.shape
        B2, C, S2 = x2.shape
        x1 = x1.permute(1,2,0)
        x1.repeat(B2, C, 1)
        x = torch.concatenate((x1,x2), dim = 0)
        x = self.SLSAN(x)
        x = x.mean(dim = 1)
        x = self.LCSAN(x)
        x = x.mean(dim=0)
        x = self.MLP(x)
        
        return x
    
class FCM(nn.Module):
    def __init__(self, P1, P2, H, W, beta, input_dim, output_dim, num_hidden_units, num1_layers, num2_layers, num_heads, num_experts):
        self.chartencoder = ChartEncoder(H, W, P1, input_dim, num_heads, num1_layers)
        self.datasetencoder = DatasetEncoder(P2, beta, input_dim, output_dim, num_hidden_units, num1_layers, num2_layers, num_heads, num_experts)
        self.matcher = Matcher(output_dim)
        
    def forward(self, x1, x2):
        B, L, S1 = x1.shape
        B, C, S2 = x2.shape
        x1.view(B*L, -1)
        x2.view(B*C, -1)
        x1 = self.chartencoder(x1)
        x2 = self.datasetencoder(x2)
        x = self.matcher(x1, x2)

        return x        
        
        
    
    

    
