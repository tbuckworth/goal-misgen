import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Define the forward diffusion process
def forward_diffusion(x_0, t, noise_schedule):
    """ Adds noise to the image x_0 at step t."""
    sqrt_alpha = torch.sqrt(1 - noise_schedule[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = torch.sqrt(noise_schedule[t]).view(-1, 1, 1, 1)
    noise = torch.randn_like(x_0)
    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise


# Text-conditioned U-Net model for noise prediction
class TextConditionedUNet(nn.Module):
    def __init__(self, text_embedding_dim=256, vocab_size=10000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, padding=1)
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, text_embedding_dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(self, x, t, text):
        # Encode time t
        t = t.view(-1, 1).float() / 100  # Normalize t
        t_embed = self.time_mlp(t)
        t_embed = t_embed.view(-1, 128, 1, 1)  # Reshape for broadcasting

        # Encode text prompt
        text_embed = self.text_embedding(text)
        text_embed = self.text_mlp(text_embed)
        text_embed = text_embed.view(-1, 128, 1, 1)  # Reshape for broadcasting

        x = self.encoder(x)
        x = x + t_embed + text_embed  # Inject time and text information
        x = self.middle(x)
        x = self.decoder(x)
        return x


# Custom dataset with text conditioning
class MNISTWithText(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root, train=train, transform=transform, download=download)
        self.image_to_text = {i: f"Digit {label}" for i, label in enumerate(self.targets)}

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        text = self.image_to_text[index]
        return image, label, text


# Training loop
def train_diffusion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextConditionedUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNISTWithText(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define noise schedule
    T = 100
    noise_schedule = torch.linspace(0.0001, 0.02, T).to(device)

    # Training
    for epoch in range(5):
        for x_0, label, text in dataloader:
            x_0 = x_0.to(device)
            t = torch.randint(0, T, (x_0.shape[0],), device=device)
            text = torch.tensor(label, device=device)  # Using labels as text prompts
            x_t, noise = forward_diffusion(x_0, t, noise_schedule)

            noise_pred = model(x_t, t, text)
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# Sampling from the trained model
def sample_images(model, noise_schedule, device, T=100, text_prompt=3):
    model.eval()
    x_t = torch.randn((1, 1, 28, 28), device=device)
    text = torch.tensor([text_prompt], device=device)
    with torch.no_grad():
        for t in reversed(range(T)):
            noise_pred = model(x_t, torch.tensor([t], device=device), text)
            x_t = (x_t - noise_pred) / torch.sqrt(1 - noise_schedule[t])
    return x_t.cpu().squeeze().numpy()

def main():
    # Train the model
    train_diffusion_model()

    # Generate a sample image conditioned on a text prompt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextConditionedUNet().to(device)
    noise_schedule = torch.linspace(0.0001, 0.02, 100).to(device)
    plt.imshow(sample_images(model, noise_schedule, device, text_prompt=3), cmap="gray")
    plt.savefig("data/diffusion_example.png")
    plt.show()

if __name__ == "__main__":
    main()
