import torch
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Import your submodules (adjust paths as needed)
from base_encoding import BaseEncoder, load_pkl, encode_patient_data
from text_transformer import TransformerEncoder
from time_aware_transformer import TimeAwareTransformer
from non_linear_layer import JointEmbeddingMapper
from gwd_loss import compute_gwd_infonce_loss_torch
from amlm_trial import mlm_loss_patient, load_data_from_pkl

def train_gwd_phase(patient_visits, patient_notes, time_diffs, code_map, vocab, d, device, num_epochs=10, batch_size=16):
    icd_dim = len(code_map)
    note_dim = len(vocab)
    base_encoder = BaseEncoder(icd_dim, note_dim, d).to(device)
    transformer_notes = TransformerEncoder(d).to(device)
    transformer_visits = TimeAwareTransformer(2 * d).to(device)
    joint_mapper = JointEmbeddingMapper(d).to(device)
    
    # Set models to training mode
    base_encoder.train()
    transformer_notes.train()
    transformer_visits.train()
    joint_mapper.train()
    
    optimizer = optim.Adam(
        list(base_encoder.parameters()) +
        list(transformer_notes.parameters()) +
        list(transformer_visits.parameters()) +
        list(joint_mapper.parameters()),
        lr=1e-3
    )
    
    patient_ids = list(patient_visits.keys())
    num_patients = len(patient_ids)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Optionally, shuffle patient_ids for each epoch
        # random.shuffle(patient_ids)
        
        for i in range(0, num_patients, batch_size):
            batch_ids = patient_ids[i:i + batch_size]
            note_embeddings_batch = []
            visit_embeddings_batch = []
            
            optimizer.zero_grad()  # Zero gradients for each batch
            
            for pid in batch_ids:
                # Convert patient visits and notes to tensors.
                icd_tensor = torch.tensor(patient_visits[pid], dtype=torch.float32, device=device)
                # If no note data exists for a patient, use a zero tensor.
                note_data = patient_notes.get(pid)
                if note_data is None:
                    note_tensor = torch.zeros((1, note_dim), dtype=torch.float32, device=device)
                else:
                    note_tensor = torch.tensor(note_data, dtype=torch.float32, device=device)
                
                # Base encoding (each output has shape (T, d))
                base_v, base_n = base_encoder(icd_tensor, note_tensor)
                
                # Process note features via the text transformer.
                # Add a batch dimension and then remove it.
                tn = transformer_notes(base_n.unsqueeze(0)).squeeze(0)  # (T, d)
                
                # Process time differences for visits.
                t_info = time_diffs.get(pid, None)
                if t_info is None:
                    t_info = np.zeros((base_v.shape[0], 1))
                else:
                    t_info = np.array(t_info)
                    if t_info.ndim == 1:
                        t_info = t_info[:, None]
                t_tensor = torch.tensor(t_info, dtype=torch.float32, device=device)
                # If the time tensor has fewer rows than base_v, pad it.
                if t_tensor.shape[0] < base_v.shape[0]:
                    pad = torch.zeros(base_v.shape[0] - t_tensor.shape[0], t_tensor.shape[1], device=device)
                    t_tensor = torch.cat([t_tensor, pad], dim=0)
                # If the feature dimension is less than d, pad with zeros.
                if t_tensor.shape[1] < d:
                    pad_feat = torch.zeros(t_tensor.shape[0], d - t_tensor.shape[1], device=device)
                    t_tensor = torch.cat([t_tensor, pad_feat], dim=-1)
                
                # Process visit features through the time-aware transformer.
                # tv will have shape (T, 2*d)
                tv = transformer_visits(base_v.unsqueeze(0), t_tensor.unsqueeze(0)).squeeze(0)
                
                # Map embeddings jointly.
                # Both outputs have shape (T, d)
                jn, jv = joint_mapper(tn, tv)
                
                # Collect the embeddings for later batch processing.
                note_embeddings_batch.append(jn)
                visit_embeddings_batch.append(jv)
            
            # Pad sequences in the batch to the same length.
            H_batch = pad_sequence(note_embeddings_batch, batch_first=True)  # (B, T_max, d)
            U_batch = pad_sequence(visit_embeddings_batch, batch_first=True)  # (B, T_max, d)
            
            # Compute the GWD loss using the batched embeddings.
            tau_g = 0.1
            tau_val = 0.05
            loss_gwd = compute_gwd_infonce_loss_torch(H_batch, U_batch, tau_g, tau_val)
            loss_gwd.backward()
            optimizer.step()
            
            epoch_loss += loss_gwd.item()
            print(f"Epoch {epoch+1}, Batch {i//batch_size+1}/{(num_patients+batch_size-1)//batch_size}, Loss: {loss_gwd.item():.4f}")
        
        avg_epoch_loss = epoch_loss / ((num_patients + batch_size - 1) // batch_size)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
    
    # Optionally save the models
    torch.save(base_encoder.state_dict(), "trained_base_encoder_gwd.pth")
    torch.save(transformer_notes.state_dict(), "trained_transformer_notes_gwd.pth")
    torch.save(transformer_visits.state_dict(), "trained_transformer_visits_gwd.pth")
    torch.save(joint_mapper.state_dict(), "trained_joint_mapper_gwd.pth")
    
    return base_encoder

# Return base_encoder (or all modules) for next phase

# from base_encoding import BaseEncoder, load_pkl, encode_patient_data
# from text_transformer import TransformerEncoder
# from time_aware_transformer import TimeAwareTransformer
# from non_linear_layer import JointEmbeddingMapper
# from gwd_loss import compute_gwd_infonce_loss_torch
# from amlm_trial import mlm_loss_patient, load_data_from_pkl

def train_amlm_phase(patient_visits, patient_notes, time_diffs , code_map, vocab, base_encoder, d, device, num_epochs=10):
    # Set base_encoder to training mode
    base_encoder.train()
    
    # Initialize transformer modules and set to training mode
    transformer_notes = TransformerEncoder(d).to(device)
    transformer_visits = TimeAwareTransformer(2 * d).to(device)
    transformer_notes.train()
    transformer_visits.train()
    
    # Create an optimizer that updates all modules
    optimizer = optim.Adam(
        list(base_encoder.parameters()) +
        list(transformer_notes.parameters()) +
        list(transformer_visits.parameters()),
        lr=1e-3
    )
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Recompute base-encoded vectors on every epoch using the updated base_encoder
        encoded_visits, encoded_notes = encode_patient_data(patient_visits, patient_notes, base_encoder, device, vocab)
        patient_ids = list(encoded_visits.keys())
        patient_losses = []
        
        for pid in patient_ids:
            # Convert the freshly computed base encodings to tensors
            H_patient = torch.tensor(encoded_visits[pid], dtype=torch.float32, device=device)  # shape: (T, d)
            U_patient = torch.tensor(encoded_notes[pid], dtype=torch.float32, device=device)  # shape: (T, d)

            # Process the note encoding through the text transformer
            tn = transformer_notes(U_patient.unsqueeze(0)).squeeze(0)  # (T, d)

            # Process time differences for visits using the actual time diff array.
            t_info = time_diffs.get(pid, None)
            if t_info is None:
                # If no time differences are available, create a zero array with one feature per time step.
                t_info = np.zeros((H_patient.shape[0], 1))
            else:
                t_info = np.array(t_info)
                # If the time diff is a 1D array, convert it to a 2D column vector.
                if t_info.ndim == 1:
                    t_info = t_info[:, None]

            # Convert the time differences into a tensor on the proper device.
            t_tensor = torch.tensor(t_info, dtype=torch.float32, device=device)

            # If the time tensor has fewer rows than H_patient, pad it with zeros.
            if t_tensor.shape[0] < H_patient.shape[0]:
                pad = torch.zeros(H_patient.shape[0] - t_tensor.shape[0], t_tensor.shape[1], device=device)
                t_tensor = torch.cat([t_tensor, pad], dim=0)

            # If the feature dimension of t_tensor is less than d, pad with zeros to reach dimension d.
            if t_tensor.shape[1] < d:
                pad_feat = torch.zeros(t_tensor.shape[0], d - t_tensor.shape[1], device=device)
                t_tensor = torch.cat([t_tensor, pad_feat], dim=-1)

            # Process visit features through the time-aware transformer.
            # tv will have shape (T, 2*d); you can later select the needed dimensions.
            tv = transformer_visits(H_patient.unsqueeze(0), t_tensor.unsqueeze(0)).squeeze(0)
    
            # Select the first d dimensions of the time-aware output as the visit representation.
            H_trans = tv[:, :d]  # (T, d)
            U_trans = tn         # (T, d)
            
            # Compute the AMLM loss using the transformed embeddings
            Lc, Ln = mlm_loss_patient(H_trans, U_trans, mask_ratio=0.15, num_heads=8, L=4)
            loss_patient = (Lc + Ln) / 2
            patient_losses.append(loss_patient)
            # print(f"AMLM Patient {pid} Epoch {epoch+1}: Loss = {loss_patient.item():.4f}")
        
        total_loss_amlm = torch.stack(patient_losses).mean()
        print(f"AMLM Epoch {epoch+1}/{num_epochs} Average Loss: {total_loss_amlm.item():.4f}")
        total_loss_amlm.backward()
        optimizer.step()
    
    # Optionally save the models after AMLM training
    torch.save(base_encoder.state_dict(), "trained_base_encoder_amlm.pth")
    torch.save(transformer_notes.state_dict(), "trained_transformer_notes_amlm.pth")
    torch.save(transformer_visits.state_dict(), "trained_transformer_visits_amlm.pth")
    return



def compute_loss(H, U, Y, memory_bank, device="cpu"):
    """
    Compute classification loss with a dual-channel retrieval mechanism.
    
    Args:
        H (torch.Tensor): Patient contextual representations (B, T, D)
        U (torch.Tensor): Patient clinical note representations (B, T, D)
        Y (torch.Tensor): Ground truth labels (B, |Y|)
        memory_bank (dict): Dictionary with keys 'Gk' and 'Gv' for memory retrieval
        device (str): Device for computation ('cpu' or 'cuda')
    
    Returns:
        tuple: Computed loss value and CLS tokens (H_cls, U_cls)
    """
    B, T, D = H.shape
    
    # Compute CLS token using attention mechanism
    query = torch.mean(H, dim=1, keepdim=True)  # (B, 1, D)
    attention_weights = F.softmax(torch.matmul(query, H.transpose(-1, -2)), dim=-1)  # (B, 1, T)
    H_cls = torch.matmul(attention_weights, H).squeeze(1)  # (B, D)
    
    query_U = torch.mean(U, dim=1, keepdim=True)  # (B, 1, D)
    attention_weights_U = F.softmax(torch.matmul(query_U, U.transpose(-1, -2)), dim=-1)  # (B, 1, T)
    U_cls = torch.matmul(attention_weights_U, U).squeeze(1)  # (B, D)
    
    if not memory_bank:
        # Handle empty memory bank: Directly use H_cls and U_cls for prediction
        r = torch.cat([H_cls, U_cls, torch.zeros_like(H_cls), torch.zeros_like(H_cls)], dim=-1)  # (B, 4D)
    else:
        Gk = memory_bank["Gk"].to(device)  # (Tr, D)
        Gv = memory_bank["Gv"].to(device)  # (Tr, D)
        
        # Compute positive and negative attention weights
        alpha_pos = F.softmax(torch.matmul(H_cls, Gk.T), dim=-1)  # (B, Tr)
        alpha_neg = -F.softmax(-torch.matmul(H_cls, Gk.T), dim=-1)  # (B, Tr)
        
        # Retrieve similar patient embeddings
        U_p = torch.matmul(alpha_pos, Gv)  # (B, D)
        U_n = torch.matmul(alpha_neg, Gv)  # (B, D)
        
        # Concatenate vectors
        r = torch.cat([H_cls, U_cls, U_p, U_n], dim=-1)  # (B, 4D)
    
    # Compute prediction probability
    linear_layer = torch.nn.Linear(4 * D, Y.shape[1]).to(device)
    logits = linear_layer(r)  # (B, |Y|)
    Y_pred = torch.sigmoid(logits)
    
    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy(Y_pred, Y)
    
    return loss, H_cls, U_cls

def train_model(patient_visits, patient_notes, time_diffs, code_map, vocab, d, device, patient_labels, num_epochs=10, batch_size=4):
    icd_dim = len(code_map)
    note_dim = len(vocab)
    base_encoder = BaseEncoder(icd_dim, note_dim, d).to(device)
    transformer_notes = TransformerEncoder(d).to(device)
    transformer_visits = TimeAwareTransformer(2 * d).to(device)
    
    base_encoder.train()
    transformer_notes.train()
    transformer_visits.train()
    
    optimizer = optim.Adam(
        list(base_encoder.parameters()) +
        list(transformer_notes.parameters()) +
        list(transformer_visits.parameters()),
        lr=1e-3
    )
    
    memory_bank = {}
    patient_ids = list(patient_visits.keys())
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        for i in range(0, len(patient_ids), batch_size):
            batch_ids = patient_ids[i:i + batch_size]
            joint_embeddings = {}
            
            for pid in batch_ids:
                icd_tensor  = torch.tensor(patient_visits[pid], dtype=torch.float32, device=device)
                note_tensor = torch.tensor(patient_notes.get(pid, torch.zeros((1, note_dim))), dtype=torch.float32, device=device)
                base_v, base_n = base_encoder(icd_tensor, note_tensor)
                
                # Process note features through the text transformer
                tn = transformer_notes(base_n.unsqueeze(0)).squeeze(0)  # (T, d)
                
                # Process visit features with time-aware transformer
                t_info = time_diffs.get(pid, None)
                if t_info is None:
                    t_info = torch.zeros((base_v.shape[0], 1), dtype=torch.float32).to(device)
                else:
                    t_info = torch.tensor(t_info, dtype=torch.float32).to(device)
                    t_info = t_info.view(-1, 1) if t_info.ndim == 1 else t_info
                
                if t_info.shape[0] < base_v.shape[0]:
                    pad = torch.zeros(base_v.shape[0] - t_info.shape[0], t_info.shape[1], device=device)
                    t_info = torch.cat((t_info, pad), dim=0)
                if t_info.shape[1] < d:
                    pad_feat = torch.zeros(t_info.shape[0], d - t_info.shape[1], device=device)
                    t_info = torch.cat((t_info, pad_feat), dim=-1)
                
                tv = transformer_visits(base_v.unsqueeze(0), t_info.unsqueeze(0)).squeeze(0)  # (T, 2*d)
                # Select the first d dimensions to match tn's dimension
                tv = tv[:, :d]  # (T, d)
                
                # Directly use tn and tv as note and visit embeddings.
                joint_embeddings[pid] = {"note_embedding": tn, "visit_embedding": tv}
            
            note_list  = [joint_embeddings[pid]["note_embedding"] for pid in joint_embeddings]
            visit_list = [joint_embeddings[pid]["visit_embedding"] for pid in joint_embeddings]
            H_batch = pad_sequence(note_list, batch_first=True)  # (B, T_max, d)
            U_batch = pad_sequence(visit_list, batch_first=True)  # (B, T_max, d)
            
            # Use the patient_labels dictionary to create Y_batch.
            Y_list = []
            for pid in batch_ids:
                # Convert the one-hot label to a tensor.
                label = torch.tensor(patient_labels[pid], dtype=torch.float32, device=device)
                Y_list.append(label)
            Y_batch = torch.stack(Y_list)  # Shape: (B, label_dim)
            
            loss, H_cls, U_cls = compute_loss(H_batch, U_batch, Y_batch, memory_bank, device)
            loss.backward()
            optimizer.step()
            
            if not memory_bank:
                memory_bank["Gk"] = H_cls.detach()
                memory_bank["Gv"] = U_cls.detach()
            else:
                memory_bank["Gk"] = torch.cat([memory_bank["Gk"], H_cls.detach()])
                memory_bank["Gv"] = torch.cat([memory_bank["Gv"], U_cls.detach()])
            
            print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss.item()}")
def train_model(patient_visits, patient_notes, time_diffs, code_map, vocab, d, device, patient_labels, num_epochs=10, batch_size=4):
    icd_dim = len(code_map)
    note_dim = len(vocab)
    base_encoder = BaseEncoder(icd_dim, note_dim, d).to(device)
    transformer_notes = TransformerEncoder(d).to(device)
    transformer_visits = TimeAwareTransformer(2 * d).to(device)
    
    base_encoder.train()
    transformer_notes.train()
    transformer_visits.train()
    
    optimizer = optim.Adam(
        list(base_encoder.parameters()) +
        list(transformer_notes.parameters()) +
        list(transformer_visits.parameters()),
        lr=1e-3
    )
    
    memory_bank = {}
    patient_ids = list(patient_visits.keys())
    # Filter out any patient IDs that are not present in patient_labels.
    patient_ids = [pid for pid in patient_ids if pid in patient_labels]
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        for i in range(0, len(patient_ids), batch_size):
            batch_ids = patient_ids[i:i + batch_size]
            joint_embeddings = {}
            
            for pid in batch_ids:
                icd_tensor  = torch.tensor(patient_visits[pid], dtype=torch.float32, device=device)
                note_tensor = torch.tensor(patient_notes.get(pid, torch.zeros((1, note_dim))), dtype=torch.float32, device=device)
                base_v, base_n = base_encoder(icd_tensor, note_tensor)
                
                # Process note features through the text transformer
                tn = transformer_notes(base_n.unsqueeze(0)).squeeze(0)  # (T, d)
                
                # Process visit features with time-aware transformer
                t_info = time_diffs.get(pid, None)
                if t_info is None:
                    t_info = torch.zeros((base_v.shape[0], 1), dtype=torch.float32).to(device)
                else:
                    t_info = torch.tensor(t_info, dtype=torch.float32).to(device)
                    t_info = t_info.view(-1, 1) if t_info.ndim == 1 else t_info
                
                if t_info.shape[0] < base_v.shape[0]:
                    pad = torch.zeros(base_v.shape[0] - t_info.shape[0], t_info.shape[1], device=device)
                    t_info = torch.cat((t_info, pad), dim=0)
                if t_info.shape[1] < d:
                    pad_feat = torch.zeros(t_info.shape[0], d - t_info.shape[1], device=device)
                    t_info = torch.cat((t_info, pad_feat), dim=-1)
                
                tv = transformer_visits(base_v.unsqueeze(0), t_info.unsqueeze(0)).squeeze(0)  # (T, 2*d)
                # Select the first d dimensions to match tn's dimension
                tv = tv[:, :d]  # (T, d)
                
                # Directly use tn and tv as note and visit embeddings.
                joint_embeddings[pid] = {"note_embedding": tn, "visit_embedding": tv}
            
            note_list  = [joint_embeddings[pid]["note_embedding"] for pid in joint_embeddings]
            visit_list = [joint_embeddings[pid]["visit_embedding"] for pid in joint_embeddings]
            H_batch = pad_sequence(note_list, batch_first=True)  # (B, T_max, d)
            U_batch = pad_sequence(visit_list, batch_first=True)  # (B, T_max, d)
            
            # Use the patient_labels dictionary to create Y_batch.
            Y_list = []
            for pid in batch_ids:
                # Convert the one-hot label to a tensor.
                label = torch.tensor(patient_labels[pid], dtype=torch.float32, device=device)
                Y_list.append(label)
            Y_batch = torch.stack(Y_list)  # Shape: (B, label_dim)
            
            loss, H_cls, U_cls = compute_loss(H_batch, U_batch, Y_batch, memory_bank, device)
            loss.backward()
            optimizer.step()
            
            if not memory_bank:
                memory_bank["Gk"] = H_cls.detach()
                memory_bank["Gv"] = U_cls.detach()
            else:
                memory_bank["Gk"] = torch.cat([memory_bank["Gk"], H_cls.detach()])
                memory_bank["Gv"] = torch.cat([memory_bank["Gv"], U_cls.detach()])
            
            print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss.item()}")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load raw data
    patient_visits = load_pkl("patient_visits_multihot.pkl")
    patient_notes  = load_pkl("patient_notes_multihot.pkl")
    code_map       = load_pkl("code_map.pkl")
    vocab          = load_pkl("vocabulary.pkl")
    time_diffs     = load_pkl("time_diffs.pkl")
    patient_labels  = load_pkl("/Users/pratikranjan/Desktop/vecocare_v2.0/patient_labels_multihot.pkl")
    
    d = 128  # Base embedding dimension
    
    print("Starting GWD-based training phase...")
    base_encoder = train_gwd_phase(patient_visits, patient_notes, time_diffs, code_map, vocab, d, device, num_epochs=10)
    print("GWD-based training phase complete.")
    
    print("Starting AMLM-based training phase...")
    train_amlm_phase(patient_visits, patient_notes,time_diffs, code_map, vocab, base_encoder, d, device, num_epochs=10)
    print("AMLM-based training phase complete.")


    # # Print all patient IDs in patient_visits and patient_labels
    # print("Patient IDs in patient_visits:")
    # print(list(patient_visits.keys()))
    # print("\nPatient IDs in patient_labels:")
    # print(list(patient_labels.keys()))

    # # Find any patient IDs that are in patient_visits but missing in patient_labels
    # missing_ids = set(patient_visits.keys()) - set(patient_labels.keys())
    # if missing_ids:
    #     print("\nPatient IDs missing in patient_labels:")
    #     print(missing_ids)
    # else:
    #     print("\nAll patient IDs in patient_visits are present in patient_labels.")






    print("starting train_model")
    train_model(patient_visits, patient_notes, time_diffs, code_map, vocab, d, device, patient_labels, num_epochs=10, batch_size=4)

if __name__ == "__main__":
    main()
