import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class HybridRecModel(nn.Module):
    def __init__(self, num_users, num_items, item_feature_dim, embedding_dim=256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.item_content_proj = nn.Linear(item_feature_dim, embedding_dim)
        self.gat1 = GATConv(embedding_dim * 2, embedding_dim, heads=4)
        self.gat2 = GATConv(embedding_dim * 4, embedding_dim, heads=1)
        self.fc = nn.Linear(embedding_dim * 3, 1)

    def forward(self, user_ids, item_ids, item_features, edge_index):
        user_emb = self.user_emb(user_ids)
        item_emb_cf = self.item_emb(item_ids)
        item_emb_content = self.item_content_proj(item_features)
        x = torch.cat([user_emb, item_emb_cf], dim=1)
        x = F.leaky_relu(self.gat1(x, edge_index))
        x = F.leaky_relu(self.gat2(x, edge_index))
        combined = torch.cat([user_emb, item_emb_content, x], dim=1)
        return torch.sigmoid(self.fc(combined))