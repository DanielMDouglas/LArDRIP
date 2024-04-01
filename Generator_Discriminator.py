import torch
import torch.nn.functional as F
from itertools import combinations
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# not used
class EdgePredictionModuleDistNN(nn.Module):
    def __init__(self, node_feat_dim, threshold=0.5):
        super(EdgePredictionModuleDistNN, self).__init__()
        self.threshold = threshold
        self.fc1 = nn.Linear(2 * node_feat_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, node_features):
        # Compute pairwise distances between nodes
        dists = torch.cdist(node_features, node_features, p=2)

        # Use a threshold to determine potential edges
        potential_edges = (dists < self.threshold).nonzero(as_tuple=False)

        # Create features for each potential edge by concatenating the features of the nodes
        edge_features = torch.cat((node_features[potential_edges[:, 0]], node_features[potential_edges[:, 1]]), dim=1)

        # Predict the existence of an edge
        edge_predictions = torch.sigmoid(self.fc2(F.relu(self.fc1(edge_features))).squeeze())

        # Filter edges based threshold
        predicted_edges = potential_edges[edge_predictions > 0.5]

        return predicted_edges.t().contiguous()

class ClosestEdgeGenerator(nn.Module):
    def __init__(self, k=2):
        super(ClosestEdgeGenerator, self).__init__()
        self.k = k

    def forward(self, node_features, batch_size, num_nodes):
        # Forward call assumes node_features has shape [batch_size * num_nodes, node_feat_dim]
        # and already contains the features for all nodes in the batch
        # nodes are XYZ coordinates (these are features) with additional feature E (energy)

        global_edge_indices = []
        node_offset = 0

        for b in range(batch_size):
            # Grab node features for the current graph in the batch
            start_idx = b * num_nodes
            end_idx = start_idx + num_nodes
            graph_node_features = node_features[start_idx:end_idx]

            # Calculate pairwise distances and find 'k' closest nodes for each node (energy is included for now)
            dist_matrix = torch.cdist(graph_node_features, graph_node_features, p=2)
            dist_matrix.fill_diagonal_(float('inf'))  # Ignore self-loops by setting the diagonal elements, which represent the distances of nodes to themselves, to zero
            _, closest_nodes = torch.topk(dist_matrix, k=self.k, largest=False) # Find the indices of the k smallest values in the dist_matrix for each node, which correspond to the k closest nodes.

            # Generate edge indices for the current graph, then adjust node_offset to point to the correct graphs
            source_nodes = torch.arange(num_nodes).unsqueeze(-1).repeat(1, self.k).flatten().to(device)
            target_nodes = closest_nodes.flatten().to(device)
            graph_edge_indices = torch.stack((source_nodes, target_nodes), dim=0) + node_offset

            global_edge_indices.append(graph_edge_indices)
            node_offset += num_nodes  # Update offset for the next graph

        # Concatenate edge indices for all graphs in the batch
        global_edge_indices = torch.cat(global_edge_indices, dim=1)  # Should have shape: [2, num_edges_total]

        return global_edge_indices

class GNNGeneratorWithEdgePrediction(nn.Module):
    def __init__(self, latent_dim, num_points, node_feat_dim=4, k=2):
        super(GNNGeneratorWithEdgePrediction, self).__init__()
        self.num_points = num_points  # 726 for this run
        self.node_feat_dim = node_feat_dim  # node_feat_dim = 4
        self.fc1 = nn.Linear(latent_dim, 128)
        # Intermediate layer to expand 128 features to 2904 features
        self.fc2 = nn.Linear(128, self.num_points * self.node_feat_dim)  # 726 * 4 = 2904
        self.gcn1 = GCNConv(128, 256) # GCN's arent used yet 
        self.gcn2 = GCNConv(256, node_feat_dim)
        self.k = k
        self.edge_generator = ClosestEdgeGenerator(k=k)

    def forward(self, z):
        # Use x and y here. They are the same thing until the .view calls 
        batch_size = z.size(0)  
        x = F.relu(self.fc1(z))
        y = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))  # Now x has enough features
        y = F.relu(self.fc2(y))
        x = x.view(batch_size, self.num_points, self.node_feat_dim)  # Reshape to (batch_size, 726, 4)
        x = x.view(batch_size*self.num_points, self.node_feat_dim)
        y = y.view(-1, self.node_feat_dim) 
        # Use the ClosestEdgeGenerator to generate edges
        global_edge_index = self.edge_generator(y, batch_size, self.num_points)
        # Generate a batch index vector
        batch_index = torch.arange(batch_size, device=z.device).repeat_interleave(self.num_points)
        
        return x, global_edge_index, batch_index

  class GraphDiscriminator(nn.Module):
    def __init__(self, node_feat_dim, output_dim=1):
        super(GraphDiscriminator, self).__init__()
        self.gcn1 = GCNConv(node_feat_dim, 128)
        self.gcn2 = GCNConv(128, 256)
        # Use a linear layer for binary classification
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x, edge_index, batch_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        # Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, batch_index)
        # Final classification
        x = torch.sigmoid(self.fc(x))
        return x
