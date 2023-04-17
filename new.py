import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torch.nn.functional as F
import os

# Define the generator
class Generator(nn.Module):
    def __init__(self, input_size=100, num_features=64):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128*num_features*7*7)
        self.conv1 = nn.ConvTranspose2d(128*num_features, 64*num_features, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64*num_features, num_features, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(num_features, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x, num_features=64):
        x = self.fc1(x)
        x = x.view(-1, 128*num_features, 7, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return x

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, num_features=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_features, 2 * num_features, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2 * num_features, 4 * num_features, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(4 * num_features * 4 * 4, 1)

    def forward(self, x, num_features=64):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(-1, 4 * num_features * 4 * 4)
        x = torch.sigmoid(self.fc1(x))
        return x


# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define the loss function and optimizer
loss_fn = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the data loader
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
latent_size = 100
dataset = ImageFolder("Humans", transform=transform)
dataloader = DataLoader(dataset, batch_size, shuffle=True)


# Train the model number of epochs = number of iterations
num_epochs = 5
# Use CPU for MacOS else use "cuda"
device = torch.device("cpu")
generator.to(device)
discriminator.to(device)

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Train the discriminator
        real_images = real_images.to(device)
        labels_real = torch.ones(real_images.size(0), 1).to(device)
        labels_fake = torch.zeros(real_images.size(0), 1).to(device)
        # Generate noise for the generator
        noise = torch.randn(batch_size, latent_size, device=device)

        # Generate a batch of fake images
        fake_images = generator(noise)

        # Train the discriminator
        discriminator_real = discriminator(real_images)
        discriminator_fake = discriminator(fake_images)

        # Create labels for the loss functions
        labels_real = torch.ones(batch_size, 1, device=device)
        labels_fake = torch.zeros(batch_size, 1, device=device)

        # Assuming labels_real has shape (batch_size, num_classes)
        batch_size, num_classes = labels_real.shape

        # Add spatial dimensions to the tensor
        labels_real = labels_real.view(batch_size, num_classes, 1, 1)

        # Resize the tensor to the desired size
        labels_real_resized = F.interpolate(labels_real, size=(36, 1), mode='nearest')
        labels_fake_resized = F.interpolate(labels_fake.view(labels_fake.size(0), labels_fake.size(1), 1, 1),
                                            size=(196, 1), mode='nearest')

        # Calculate the loss for the discriminator
        loss_d_real = F.binary_cross_entropy(discriminator_real.squeeze(),
                                      torch.ones(discriminator_real.squeeze().size()).to(device))
        loss_d_fake = loss_fn(discriminator_fake, labels_fake_resized)
        loss_d = loss_d_real + loss_d_fake

        # Backpropagate the gradients and update the discriminator weights
        discriminator.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # Train the generator
        noise = torch.randn(real_images.size(0), 100).to(device)
        fake_images = generator(noise)
        discriminator_fake = discriminator(fake_images)
        loss_g = loss_fn(discriminator_fake, labels_real)
        generator.zero_grad()
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # Print the loss and save the model
        print(f"Epoch {epoch + 1}/{num_epochs}, Generator Loss: {loss_g:.4f}, Discriminator Loss: {loss_d:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"Downloads/AI-Photo-Generator/generator{epoch + 1}.pth")

# Generate a single photo and save it
generator.eval()
with torch.no_grad():
    noise = torch.randn(1, 100).to(device)
    output = generator(noise)
    save_image(output, "Downloads/AI-Photo-Generator/Humans/output.jpg", normalize=True)

# Confirm photo has been saved
print("Output photo saved successfully!")