import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialEncoding(nn.Module):
    def __init__(self, input_dim=3, encoding_dim=64):
        super(SpatialEncoding, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim//2),
            nn.ReLU(),
            nn.Linear(encoding_dim//2, encoding_dim)
        )
    
    def forward(self, xyz):
        # xyz: [batch_size, num_points, 3]
        return self.encoder(xyz)    


class AttentionPool(nn.Module):
    def __init__(self, feature_dim, out_features, num_heads):
        super(AttentionPool, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.out_features = out_features // num_heads  # Assume out_features is divisible by num_heads

        # Linear transformations for queries, keys, and values
        self.query = nn.Linear(feature_dim, out_features)
        self.key = nn.Linear(feature_dim, out_features)
        self.value = nn.Linear(feature_dim, out_features)

        # Final linear layer to combine head outputs
        self.final_linear = nn.Linear(out_features, out_features)

    def forward(self, x):
        batch_size, num_points, _ = x.shape

        # Apply linear transformations and split into heads
        query = self.query(x).view(batch_size, num_points, self.num_heads, self.out_features)
        key = self.key(x).view(batch_size, num_points, self.num_heads, self.out_features)
        value = self.value(x).view(batch_size, num_points, self.num_heads, self.out_features)

        # Transpose to get dimensions [batch_size, num_heads, num_points, out_features]
        query, key, value = [tensor.transpose(1, 2) for tensor in (query, key, value)]

        # Compute attention scores and weights
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.out_features ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate heads and apply final linear layer
        concatenated = attention_output.transpose(1, 2).contiguous().view(batch_size, num_points, -1)
        output = self.final_linear(concatenated)

        # Could use max pooling instead?
        pooled_output = output.mean(dim=1)

        return pooled_output


'''
class AttentionPool(nn.Module):
    def __init__(self, feature_dim, out_features):
        super(AttentionPool, self).__init__()
        self.query = nn.Linear(feature_dim, out_features)
        self.key = nn.Linear(feature_dim, out_features)
        self.value = nn.Linear(feature_dim, out_features)
        
    def forward(self, x):
        query, key, value = self.query(x), self.key(x), self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        pooled_features = torch.matmul(attention_weights, value)
        #return pooled_features.mean(dim=1)
        return pooled_features.max(dim=1).values
'''

class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(PointTransformerLayer, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        attn_output, _ = self.self_attn(x, x, x)
        output = self.linear2(F.relu(attn_output))
        return output
    
# Decoder
class SophisticatedPointGenerator(nn.Module):
    def __init__(self, feature_dim, num_heads, max_points, out_features=512):
        super(SophisticatedPointGenerator, self).__init__()
        #self.attention_pool = AttentionPool(feature_dim, out_features) #non-multihead
        self.attention_pool = AttentionPool(feature_dim, out_features, num_heads)
        self.max_points = max_points
        
        mlp_dims = [512, 256, 128, 64]  # Example MLP dimensions
        mlp_layers = []
        last_dim = out_features  # The output dimension of AttentionPool
        for dim in mlp_dims:
            mlp_layers.extend([nn.Linear(last_dim, dim), nn.ReLU()])
            last_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        self.point_distribution_decoder = nn.Sequential(nn.Linear(last_dim, 3 * max_points))
        self.cardinality_decoder = nn.Sequential(nn.Linear(last_dim, max_points))
        
    def forward(self, x):
        # Apply attention-based pooling directly
        x = self.attention_pool(x)
        
        # Pass the pooled features through the MLP
        mlp_output = self.mlp(x)
        
        # Generate point distribution and cardinality probabilities
        point_distribution = self.point_distribution_decoder(mlp_output).view(-1, self.max_points, 3)
        cardinality_logits = self.cardinality_decoder(mlp_output)
        cardinality_probs = F.softmax(cardinality_logits, dim=-1)
        
        return point_distribution, cardinality_probs

'''
# Integrated Model using AttentionPool for feature pooling
class IntegratedPointCloudCompletionModel(nn.Module):
    def __init__(self, feature_dim, num_heads, max_points):
        super(IntegratedPointCloudCompletionModel, self).__init__()
        self.context_encoder = PointTransformerLayer(3, feature_dim, num_heads)
        self.point_generator = SophisticatedPointGenerator(feature_dim, num_heads, max_points)
        
    def forward(self, known_regions, mask):
        mask_expanded = mask.unsqueeze(-1).float()  # Ensure mask is float for multiplication
        known_regions_masked = known_regions * mask_expanded
        
        # Encode context using the PointTransformerLayer
        context_features = self.context_encoder(known_regions_masked)
        # Here, we skip the mean pooling to maintain feature dimensions and rely on AttentionPool
        # context_features = context_features.mean(dim=1) # Optionally, if needed
        
        # Generate point distribution and cardinality probabilities using SophisticatedPointGenerator
        point_distribution, cardinality_probs = self.point_generator(context_features)
        
        return point_distribution, cardinality_probs
'''

class IntegratedPointCloudCompletionModel(nn.Module):
    def __init__(self, feature_dim, num_heads, max_points, encoding_dim=512):
        super(IntegratedPointCloudCompletionModel, self).__init__()
        self.spatial_encoding = SpatialEncoding(input_dim=3, encoding_dim=encoding_dim)
        self.context_encoder = PointTransformerLayer(encoding_dim, feature_dim, num_heads)
        self.point_generator = SophisticatedPointGenerator(feature_dim, num_heads, max_points, out_features=feature_dim)
        
    def forward(self, known_regions, mask):
        mask_expanded = mask.unsqueeze(-1).float()  # Unsqueeze and ensure mask is float for multiplication
        known_regions_masked = known_regions * mask_expanded
        
        # Apply spatial encoding to the known regions first
        encoded_spatial_features = self.spatial_encoding(known_regions_masked)
        
        # Encode context using the PointTransformerLayer with spatially encoded features
        context_features = self.context_encoder(encoded_spatial_features)
        
        # Generate point distribution and cardinality probabilities using SophisticatedPointGenerator
        point_distribution, cardinality_probs = self.point_generator(context_features)
        
        return point_distribution, cardinality_probs
