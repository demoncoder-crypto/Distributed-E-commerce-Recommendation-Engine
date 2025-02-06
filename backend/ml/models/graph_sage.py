import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import dgl

class GraphSAGE(nn.Module):
    """
    GraphSAGE model for generating user and item embeddings.
    """
    def __init__(self, in_feats: int, h_feats: int, num_layers: int = 2):
        """
        Args:
            in_feats (int): Input feature dimension.
            h_feats (int): Hidden feature dimension.
            num_layers (int): Number of GraphSAGE layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, h_feats, aggregator_type="mean"))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(h_feats, h_feats, aggregator_type="mean"))

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GraphSAGE.

        Args:
            g (dgl.DGLGraph): Input graph.
            features (torch.Tensor): Node features.

        Returns:
            torch.Tensor: Node embeddings.
        """
        h = features
        for layer in self.layers:
            h = layer(g, h)
            h = F.relu(h)
        return h

def build_user_item_graph(user_ids: list, item_ids: list, interactions: list) -> dgl.DGLGraph:
    """
    Builds a bipartite user-item graph from interaction data.

    Args:
        user_ids (list): List of user IDs.
        item_ids (list): List of item IDs.
        interactions (list): List of tuples (user_id, item_id, interaction_type).

    Returns:
        dgl.DGLGraph: Bipartite graph.
    """
    # Create graph
    g = dgl.heterograph({
        ("user", "interacts", "item"): (user_ids, item_ids)
    })
    
    # Add node features (optional)
    g.nodes["user"].data["feat"] = torch.randn(len(user_ids), 128)  # Random features for demo
    g.nodes["item"].data["feat"] = torch.randn(len(item_ids), 128)
    
    return g

def train_graph_sage(g: dgl.DGLGraph, epochs: int = 10