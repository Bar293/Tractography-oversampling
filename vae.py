import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# -------------------- CONFIG --------------------
vae_config = {
    "BATCH_SIZE": 64,
    "INPUT_DIM": 15 * 3,       # 45 features
    "LATENT_DIM": 64,          # latent space dimension
    "HIDDEN1": 256,
    "HIDDEN2": 128,
    "NUM_EPOCHS": 200,
    "LEARNING_RATE": 1e-4,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- VAE DEFINITION --------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden1, hidden2):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden2)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
            nn.Tanh()  # output normalized to [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder_input(z)
        recon = self.decoder(recon)
        return recon, mu, logvar

    def sample(self, num_samples):
        z = torch.randn(num_samples, vae_config["LATENT_DIM"]).to(device)
        recon = self.decoder_input(z)
        recon = self.decoder(recon)
        return recon

# -------------------- LOSS FUNCTION --------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# -------------------- TRAINING --------------------
def train_vae(data_array, model_path="vae_model.pth"):
    # Flatten (N, 15, 3) → (N, 45)
    data_array = data_array.reshape(data_array.shape[0], -1)
    dataset = TensorDataset(torch.tensor(data_array, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=vae_config["BATCH_SIZE"], shuffle=True)

    model = VAE(
        vae_config["INPUT_DIM"],
        vae_config["LATENT_DIM"],
        vae_config["HIDDEN1"],
        vae_config["HIDDEN2"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=vae_config["LEARNING_RATE"])

    for epoch in range(vae_config["NUM_EPOCHS"]):
        model.train()
        train_loss = 0
        recon_mse_per_sample, recon_mse_per_dim, recon_rmse_per_dim, kl_per_sample = [], [], [], []

        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Metrics
            batch_size = batch.size(0)
            mse_sum_per_sample = recon_loss.item() / batch_size
            mse_per_dim = mse_sum_per_sample / vae_config["INPUT_DIM"]
            rmse_per_dim = mse_per_dim ** 0.5
            kl_per_samp = kl_loss.item() / batch_size

            recon_mse_per_sample.append(mse_sum_per_sample)
            recon_mse_per_dim.append(mse_per_dim)
            recon_rmse_per_dim.append(rmse_per_dim)
            kl_per_sample.append(kl_per_samp)

        avg_loss = train_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{vae_config['NUM_EPOCHS']} "
              f"- Loss: {avg_loss:.4f} "
              f"| MSE_sum/sample: {np.mean(recon_mse_per_sample):.4f} "
              f"| MSE/dim: {np.mean(recon_mse_per_dim):.4f} "
              f"| RMSE/dim: {np.mean(recon_rmse_per_dim):.4f} "
              f"| KL/sample: {np.mean(kl_per_sample):.2f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    return model

# -------------------- GENERATION + CONCAT --------------------
def generate_and_concat(model_path, dataset_path, labels_path, target, class_index, output_data, output_labels):
    # Load dataset and labels
    X = np.load(dataset_path)  # (N, 15, 3)
    y = np.load(labels_path)   # (N, num_classes)

    y_classes = np.argmax(y, axis=1)
    current_count = np.sum(y_classes == class_index)
    print(f"Class {class_index}: {current_count} samples available.")

    to_generate = target - current_count
    if to_generate <= 0:
        print(f"{current_count} samples already exist, nothing to generate.")
        np.save(output_data, X)
        np.save(output_labels, y)
        return

    # Load model
    model = VAE(
        input_dim=vae_config["INPUT_DIM"],
        latent_dim=vae_config["LATENT_DIM"],
        hidden1=vae_config["HIDDEN1"],
        hidden2=vae_config["HIDDEN2"]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Generating {to_generate} synthetic samples...")
    generated = []
    with torch.no_grad():
        total_generated = 0
        while total_generated < to_generate:
            batch = min(512, to_generate - total_generated)
            samples = model.sample(batch).cpu().numpy()
            samples = samples.reshape(-1, 15, 3)
            generated.append(samples)
            total_generated += samples.shape[0]

    generated = np.concatenate(generated, axis=0)

    # Concatenate dataset and labels
    X_new = np.concatenate([X, generated], axis=0)
    num_classes = y.shape[1]
    synthetic_labels = np.zeros((to_generate, num_classes))
    synthetic_labels[:, class_index] = 1
    y_new = np.concatenate([y, synthetic_labels], axis=0)

    np.save(output_data, X_new)
    np.save(output_labels, y_new)

    print(f"Final dataset: {X_new.shape}, Final labels: {y_new.shape}")
    print(f"Saved at: {output_data}, {output_labels}")

# -------------------- EXECUTION --------------------
if __name__ == "__main__":
    # Train with FX_L data
    train_data = np.load("data/fx_l_database.npy")  # (N, 15, 3)
    print("Loaded dataset:", train_data.shape)
    vae_model = train_vae(train_data, "vae_fxl.pth")

    # Generate for FX_L
    generate_and_concat(
        model_path="vae_fxl.pth",
        dataset_path="dataset/base_train_database.npy",
        labels_path="dataset/base_train_labels.npy",
        target=20000,
        class_index=13,  # FX_L
        output_data="dataset/vae_train_database.npy",
        output_labels="dataset/vae_train_labels.npy"
    )

    # Train with FX_R data
    train_data = np.load("data/fx_r_database.npy")  # (N, 15, 3)
    print("Loaded dataset:", train_data.shape)
    vae_model = train_vae(train_data, "vae_fxr.pth")

    # Generate for FX_R
    generate_and_concat(
        model_path="vae_fxr.pth",
        dataset_path="dataset/vae_train_database.npy",
        labels_path="dataset/vae_train_labels.npy",
        target=20000,
        class_index=14,  # FX_R
        output_data="dataset/vae_train_database.npy",
        output_labels="dataset/vae_train_labels.npy"
    )
