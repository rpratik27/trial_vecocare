import torch
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Import your submodules (adjust import paths as needed)
from base_encoding import BaseEncoder, load_pkl, encode_patient_data
from text_transformer import TransformerEncoder
from time_aware_transformer import TimeAwareTransformer
from non_linear_layer import JointEmbeddingMapper
from gwd_loss import compute_gwd_infonce_loss_torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load raw data
    patient_visits = load_pkl("patient_visits_multihot.pkl")
    patient_notes  = load_pkl("patient_notes_multihot.pkl")
    code_map       = load_pkl("code_map.pkl")
    vocab          = load_pkl("vocabulary.pkl")
    time_diffs     = load_pkl("time_diffs.pkl")
    
    icd_dim = len(code_map)
    note_dim = len(vocab)
    d = 128  # Base embedding dimension

    # Initialize submodules and move to device
    base_encoder       = BaseEncoder(icd_dim, note_dim, d).to(device)
    transformer_notes  = TransformerEncoder(d).to(device)
    transformer_visits = TimeAwareTransformer(2*d).to(device)
    joint_mapper       = JointEmbeddingMapper(d).to(device)
    
    optimizer = optim.Adam(
        list(base_encoder.parameters()) +
        list(transformer_notes.parameters()) +
        list(transformer_visits.parameters()) +
        list(joint_mapper.parameters()),
        lr=1e-3
    )
    
    # Build the pipeline and keep outputs in the computational graph:
    joint_embeddings = {}
    for pid in patient_visits:
        # Base Encoding
        icd_tensor  = torch.tensor(patient_visits[pid], dtype=torch.float32).to(device)
        note_tensor = torch.tensor(patient_notes.get(pid, np.zeros((1, note_dim))), dtype=torch.float32).to(device)
        base_v, base_n = base_encoder(icd_tensor, note_tensor)  # Each: (T, d)
        
        # Transformer for Notes
        tn = transformer_notes(base_n.unsqueeze(0)).squeeze(0)  # (T, d)
        
        # Process time differences for Visits:
        t_info = time_diffs.get(pid, None)
        if t_info is None:
            t_info = np.zeros((base_v.shape[0], 1))
        else:
            t_info = np.array(t_info)
            if t_info.ndim == 1:
                t_info = t_info[:, None]
        t_tensor = torch.tensor(t_info, dtype=torch.float32).to(device)
        if t_tensor.shape[0] < base_v.shape[0]:
            pad = torch.zeros(base_v.shape[0] - t_tensor.shape[0], t_tensor.shape[1], device=device)
            t_tensor = torch.cat((t_tensor, pad), dim=0)
        if t_tensor.shape[1] < d:
            pad_feat = torch.zeros(t_tensor.shape[0], d - t_tensor.shape[1], device=device)
            t_tensor = torch.cat((t_tensor, pad_feat), dim=-1)
        
        # TimeAwareTransformer will concatenate internally.
        tv = transformer_visits(base_v.unsqueeze(0), t_tensor.unsqueeze(0)).squeeze(0)  # (T, 2*d)
        
        # Joint Embedding Mapping (do not detach; keep on device)
        jn, jv = joint_mapper(tn, tv)  # jn: (T, d), jv: (T, d)
        joint_embeddings[pid] = {"note_embedding": jn, "visit_embedding": jv}
    
    # Batch the joint embeddings (pad sequences as needed)
    note_list  = [joint_embeddings[pid]["note_embedding"] for pid in joint_embeddings]
    visit_list = [joint_embeddings[pid]["visit_embedding"] for pid in joint_embeddings]
    H_batch = pad_sequence(note_list, batch_first=True)  # (B, T_max, d)
    U_batch = pad_sequence(visit_list, batch_first=True)  # (B, T_max, d)
    
    # Compute loss using differentiable GWD InfoNCE loss
    tau_g = 0.1
    tau_val = 0.05
    loss = compute_gwd_infonce_loss_torch(H_batch.to(device), U_batch.to(device), tau_g, tau_val)
    print("Loss:", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Backpropagation complete.")

if __name__ == "__main__":
    main()
