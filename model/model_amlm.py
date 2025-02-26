import torch
import torch.optim as optim
import pickle
from torch.nn.utils.rnn import pad_sequence
from base_encoding import BaseEncoder

# Import your AMLM loss function and loader from your AMLM module.
from amlm_trial import mlm_loss_patient, load_data_from_pkl

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the base-encoded vectors from pickle files.
    # Each file is a dictionary mapping patient ID to a 2D numpy array.
    H_dict = load_data_from_pkl("base_encoded_visits.pkl")  # Visits base-encoded (shape: [T, d])
    U_dict = load_data_from_pkl("base_encoded_notes.pkl")   # Notes base-encoded (shape: [T, d])
    
    patient_ids = list(H_dict.keys())
    losses = []
    
    # Initialize an optimizer for the BaseEncoder (or any persistent module you're training).
    # Here, we assume you're training a BaseEncoder; replace this with the appropriate module.
    # For demonstration, let's assume you have a module 'base_encoder' already defined and loaded.
    # If not, you'll need to set up a model that produces these base-encoded outputs.
    # For now, we create a dummy optimizer with the parameters from the base encoder.
    # Replace this with your actual model's parameters.
    # from base_encoder import BaseEncoder, load_pkl  # Assuming BaseEncoder is defined in base_encoder.py
    # For this example, we need icd_dim and note_dim; we assume these are available via code_map and vocabulary.
    code_map = load_data_from_pkl("code_map.pkl")
    vocab = load_data_from_pkl("vocabulary.pkl")
    icd_dim = len(code_map)
    note_dim = len(vocab)
    d = 128
    base_encoder = BaseEncoder(icd_dim, note_dim, d).to(device)
    
    optimizer = optim.Adam(base_encoder.parameters(), lr=1e-3)
    
    # Loop over patients, computing the AMLM loss per patient.
    for pid in patient_ids:
        H_patient = torch.tensor(H_dict[pid], dtype=torch.float32, device=device)
        U_patient = torch.tensor(U_dict[pid], dtype=torch.float32, device=device)
        
        # Compute AMLM loss for this patient.
        # mlm_loss_patient returns a tuple (Lc, Ln), so we average them.
        Lc, Ln = mlm_loss_patient(H_patient, U_patient, mask_ratio=0.15, num_heads=8, L=4)
        loss_patient = (Lc + Ln) / 2
        losses.append(loss_patient)
        print(f"Patient {pid}: Aggregated Loss = {loss_patient.item():.4f}")
    
    # Aggregate loss over patients (e.g., average)
    total_loss = torch.stack(losses).mean()
    print(f"Average AMLM Loss: {total_loss.item():.4f}")
    
    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("Backpropagation complete.")

if __name__ == "__main__":
    main()
