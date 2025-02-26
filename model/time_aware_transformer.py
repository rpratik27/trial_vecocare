# import torch
# import torch.nn as nn
# import pickle

# class TimeAwareTransformer(nn.Module):
#     def __init__(self, d, num_layers=2, num_heads=4, dim_feedforward=256):
#         super(TimeAwareTransformer, self).__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
#     def forward(self, visit_vectors, time_differences):
#         # Concatenate visit vectors with time differences
#         if time_differences.shape[-1] == 1:
#             pad = torch.zeros(time_differences.shape[0], time_differences.shape[1], 1, device=time_differences.device)
#             time_differences = torch.cat([time_differences, pad], dim=-1)
    
#         combined_input = torch.cat((visit_vectors, time_differences), dim=-1)
#         return self.transformer_encoder(combined_input)

# # Load encoded visit vectors and time differences
# with open("/Users/pratikranjan/Desktop/vecocare_v2.0/base_encoded_visits.pkl", "rb") as f:
#     encoded_patient_visits = pickle.load(f)

# with open("time_diffs.pkl", "rb") as f:
#     time_diffs = pickle.load(f)

# # Transformer model
# d = 128  # Original embedding dimension from base encoding
# transformer_model = TimeAwareTransformer(d * 2)  # Update to d * 2 to accommodate concatenated input

# # Process each patient separately
# encoded_transformed_visits = {}
# for pid, visit_vector in encoded_patient_visits.items():
#     visit_tensor = torch.tensor(visit_vector, dtype=torch.float32)  # Convert to tensor
#     time_tensor = torch.tensor(time_diffs.get(pid, []), dtype=torch.float32).unsqueeze(1)  # Convert time differences to tensor

#     # Add a 0 at the start of the time tensor to match the dimensions
#     time_tensor = torch.cat((torch.zeros(1, 1), time_tensor), dim=0)  # Shape: (max_visits + 1, 1)

#     # Ensure the time tensor matches the size of visit_tensor
#     time_tensor = time_tensor.expand(-1, visit_tensor.size(1))  # Match the size of visit_tensor

#     # Process with Time-Aware Transformer
#     transformed_visits = transformer_model(visit_tensor.unsqueeze(0), time_tensor.unsqueeze(0)).squeeze(0)
#     encoded_transformed_visits[pid] = transformed_visits.detach().numpy()

# # Save the transformed visits
# with open("transformed_patient_visits.pkl", "wb") as f:
#     pickle.dump(encoded_transformed_visits, f)

# print("Transformed patient visits saved successfully.")





#old transformer above
# _________________________________________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(self.max_seq_len)
        ])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
        # Register as a buffer so it’s not updated during training
        self.register_buffer('pe', position_encoding.unsqueeze(0))  # Shape: (1, max_seq_len, d_model)

    def forward(self, seq_len):
        # Return positional encoding for a given sequence length
        return self.pe[:, :seq_len, :]

class TimeAwareTransformer(nn.Module):
    def __init__(self, d, num_layers=2, num_heads=4, dim_feedforward=256, max_seq_len=100):
        super(TimeAwareTransformer, self).__init__()
        self.d = d
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Layers for processing time differences
        self.selection_layer = nn.Linear(1, d)
        self.time_layer = nn.Linear(d, d)
        # Learnable bias added to visit embeddings
        self.bias_embedding = nn.Parameter(torch.Tensor(d))
        bound = 1 / math.sqrt(d)
        nn.init.uniform_(self.bias_embedding, -bound, bound)
        # Positional encoding for the sequence
        self.positional_encoding = PositionalEncoding(d, max_seq_len)

    def forward(self, visit_vectors, time_differences):
        """
        visit_vectors: shape (batch, seq_len, d)
        time_differences: shape (batch, seq_len, 1)
        """
        # Process time differences:
        # Apply a linear transformation followed by a non-linear function:
        # time_feature = 1 - tanh( (selection_layer(time_differences))^2 )
        time_feature = 1 - torch.tanh(self.selection_layer(time_differences)**2)
        time_feature = self.time_layer(time_feature)
        # Add a bias term to visit embeddings
        output = visit_vectors + self.bias_embedding
        # Inject the time information
        output = output + time_feature
        # Add positional encoding
        batch_size, seq_len, _ = output.size()
        pos_enc = self.positional_encoding(seq_len)  # (1, seq_len, d)
        output = output + pos_enc.expand(batch_size, -1, -1)
        # Process through the transformer encoder
        output = self.transformer_encoder(output)
        return output

# Load encoded visit vectors and time differences from pickle files
with open("/Users/pratikranjan/Desktop/vecocare_v2.0/base_encoded_visits.pkl", "rb") as f:
    encoded_patient_visits = pickle.load(f)

with open("time_diffs.pkl", "rb") as f:
    time_diffs = pickle.load(f)

# Define the embedding dimension (original visit vectors have dimension d)
d = 128  
# Initialize the transformer model.
# Note: We use the same dimension d (not d*2) because we incorporate time via addition, not concatenation.
transformer_model = TimeAwareTransformer(d, num_layers=2, num_heads=4, dim_feedforward=256, max_seq_len=100)

# Process each patient separately
encoded_transformed_visits = {}
for pid, visit_vector in encoded_patient_visits.items():
    # Convert the visit vectors to a tensor: shape (seq_len, d)
    visit_tensor = torch.tensor(visit_vector, dtype=torch.float32)
    
    # Get the corresponding time differences and ensure shape (seq_len, 1)
    # Here we assume time_diffs[pid] is a list or array of scalar time differences.
    time_diff = time_diffs.get(pid, [])
    if len(time_diff) == 0:
        # If no time differences exist, use zeros.
        time_tensor = torch.zeros(visit_tensor.size(0), 1, dtype=torch.float32)
    else:
        time_tensor = torch.tensor(time_diff, dtype=torch.float32).unsqueeze(1)
    
    # If needed, pad the time tensor to match the number of visits
    seq_len = visit_tensor.size(0)
    if time_tensor.size(0) != seq_len:
        diff = seq_len - time_tensor.size(0)
        pad = torch.zeros(diff, 1, dtype=torch.float32)
        time_tensor = torch.cat([time_tensor, pad], dim=0)
    
    # Add batch dimension: shape becomes (1, seq_len, d) and (1, seq_len, 1)
    visit_tensor = visit_tensor.unsqueeze(0)
    time_tensor = time_tensor.unsqueeze(0)
    
    # Process with the time-aware transformer
    transformed_visits = transformer_model(visit_tensor, time_tensor)
    # Remove the batch dimension and store the result as a NumPy array
    encoded_transformed_visits[pid] = transformed_visits.squeeze(0).detach().numpy()

# Save the transformed visits to a pickle file
with open("transformed_patient_visits.pkl", "wb") as f:
    pickle.dump(encoded_transformed_visits, f)

print("Transformed patient visits saved successfully.")
