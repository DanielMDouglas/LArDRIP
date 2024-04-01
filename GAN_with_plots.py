from Generator_Discriminator.py import *
from torch_geometric.data import Data, Dataset
from torch_cluster import knn_graph
from torch_geometric.data import Batch
%matplotlib inline
import matplotlib
#import matplotlib as mpl
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PyQt5
import random
from torch_geometric.loader import DataLoader 
import torch.nn.functional as F

loaded_normalized_tracks_np = np.load(save_path)

loaded_normalized_tracks_tensor = torch.from_numpy(loaded_normalized_tracks_np).float().to(device)  # or 'cuda' if you're using GPU

normalized_tracks = loaded_normalized_tracks_tensor

coords = normalized_tracks[:, :, :3]
energies = normalized_tracks[:, :, 3:].unsqueeze(-1)
energies = energies.squeeze(-1)

class ParticleTrackDataset(Dataset):
    def __init__(self, coords, energies):
        super(ParticleTrackDataset, self).__init__()
        self.graphs = self._prepare_data(coords, energies)
    
    def _prepare_data(self, coords, energies):
        graphs = []
        for i in range(coords.shape[0]):
            points_tensor = coords[i]  # Tensor of shape (num_points, 3)
            energy_tensor = energies[i]  # Tensor of shape (num_points, 1)
            node_features = torch.cat([points_tensor, energy_tensor], dim=-1)
            
            # Create edges based on spatial proximity
            edge_index = knn_graph(points_tensor, k=2, loop=False)
            
            graph = Data(x=node_features, edge_index=edge_index)
            graphs.append(graph)
        return graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

dataset = ParticleTrackDataset(coords, energies)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

latent_dim = 1000  # Dimensionality of the input noise vector
n_points = 726  
output_dim_discriminator = 1  # Output dimension for the generator (X, Y, Z, Energy for each point), this should always be 1, I think
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
generator = GNNGeneratorWithEdgePrediction(latent_dim=latent_dim, num_points=n_points, node_feat_dim=4).to(device)
discriminator = GraphDiscriminator(node_feat_dim=4, output_dim=output_dim_discriminator).to(device)

# Initialize the optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

num_epochs = 100  # Number of epochs to train for

nodes = []
edge_indices = []
batches = []

for epoch in range(num_epochs):
    for i, real_data in enumerate(train_dataloader):
        real_data = real_data.to(device)
        #batch_size = 1
        batch_size = real_data.num_graphs
        #print(batch_size)
        # --------------
        # Train Discriminator
        # --------------
        d_optimizer.zero_grad()

        # Real data
        #print('REAL DATA IS COMING')
        real_pred = discriminator(real_data.x, real_data.edge_index, real_data.batch)
        real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))

        # Generate fake data
        noise = torch.randn(batch_size, latent_dim, device=device)  # Noise dimension is 100, maybe this should match the latent_dim? 
        #print('FAKE DATA IS COMING')
        fake_nodes, fake_edge_index, fake_batch = generator(noise)
        fake_nodes, fake_edge_index, fake_batch = fake_nodes.to(device), fake_edge_index.to(device), fake_batch.to(device)
        # Fake predictions
        fake_pred = discriminator(fake_nodes, fake_edge_index, fake_batch)
        fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))

        # Total discriminator loss
        d_loss = real_loss + fake_loss #they do this in literature, im not sure I understand it
        d_loss.backward()
        d_optimizer.step()

        # --------------
        # Train Generator
        # --------------
        g_optimizer.zero_grad()

        # Generate fake data
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_nodes, fake_edge_index, fake_batch = generator(noise)
        # Save nodes at 90%+ training
        if epoch >= (90/100)*num_epochs: 
            fake_nodes_np = fake_nodes.detach().cpu().numpy()
            fake_edge_index_np = fake_edge_index.detach().cpu().numpy()
            fake_batch_np = fake_batch.detach().cpu().numpy()
            for j in range(batch_size):  # Assuming batch_size is the number of tracks per batch
                # Extract nodes for the i-th track
                track_nodes = fake_nodes_np[j*n_points:(j+1)*n_points]
                nodes.append(track_nodes)
        # Try to fool the discriminator into thinking the data is real
        trick_pred = discriminator(fake_nodes, fake_edge_index, fake_batch)
        g_loss = F.binary_cross_entropy(trick_pred, torch.ones_like(trick_pred))

        g_loss.backward()
        g_optimizer.step()
        #print(g_loss, d_loss)
    print(f"Epoch {epoch}: D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Correctly select a track randomly using a random index
import random

#random_index = random.randint(0, len(nodes) - 1)
#track = nodes[random_index]
track = nodes[0]
X, Y, Z, E = track[:,0], track[:,1], track[:,2], track[:,3]

# Plotting using plotly 

selected_og_track = normalized_tracks[8].cpu().numpy()  
import plotly.graph_objects as go

# Creating a 3D scatter plot with Plotly
pred_track = go.Scatter3d(
    x=X,
    y=Y,
    z=Z,
    mode='markers',
    marker=dict(
        size=5,
        color=E,  # Use the fourth column (energy) for color
        colorscale='Viridis',  # Choose a color scale
        colorbar=dict(title='Energy'),
        opacity=0.8
    )
)

og_track = go.Scatter3d(
    x=selected_og_track[:, 0],
    y=selected_og_track[:, 1],
    z=selected_og_track[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=selected_og_track[:, 3],  # Energy values for color
        colorscale='Inferno',
        colorbar=dict(title='Energy'),
        opacity=0.8
    )
)

fig = go.Figure(data=[pred_track, og_track])

fig.update_layout(title=f'3D Track Visualization (Index {random_index})',
                  scene=dict(
                      xaxis_title='X',
                      yaxis_title='Y',
                      zaxis_title='Z'
                  ),
                  margin=dict(l=0, r=0, b=0, t=0))

fig.show()
