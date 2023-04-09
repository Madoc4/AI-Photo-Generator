import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Define the generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 7*7*256)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.view(-1, 256, 7, 7)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        return x

# Define the discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        x = x.view(batch_size, 1, 28, 28)
        return x




# Initialize the generator and discriminator networks
generator = Generator()
discriminator = Discriminator()

# Define the loss function and optimizer for the discriminator
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Define the loss function and optimizer for the generator
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = dset.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Device set up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Train the model
num_epochs = 50

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        real_images, _ = data
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        # Train the discriminator
        discriminator.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator with real images
        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, labels_real)

        # Train the discriminator with fake images
        noise = torch.randn(batch_size, 100).to(device)
        fake_images = generator(noise)
        outputs_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs_fake, labels_fake)

        # Compute the total discriminator loss and update the discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train the generator
        generator.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)

        # Generate fake images and calculate the generator loss
        noise = torch.randn(batch_size, 100).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, labels_real.expand_as(outputs).to(device))

        # Backpropagate the generator loss and update the generator
        g_loss.backward()
        g_optimizer.step()

        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch + 1, num_epochs, i, len(train_loader),
                     d_loss.item(), g_loss.item()))

    # Save a generated image for each epoch
    with torch.no_grad():
        fake = generator(torch.randn(1, 100).to(device)).detach().cpu()
        vutils.save_image(fake, 'generated_images/fake_image_epoch_%03d.png' % (epoch + 1), normalize=True)


