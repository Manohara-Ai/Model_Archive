import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        # Initialize the Discriminator with the specified number of channels, feature dimensions,
        # number of classes, and image size.
        self.img_size = img_size  # Store the image size for embedding
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),  # Initial convolution
            nn.LeakyReLU(0.2),  # Activation function
            self._block(features_d, features_d * 2, 4, 2, 1),  # Downsampling block
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # Downsampling block
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # Downsampling block
            nn.Conv2d(features_d * 8, 1, 4, 2, 0),  # Final convolution to output a single channel
            nn.Sigmoid(),  # Sigmoid activation to get probabilities
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)  # Class embedding

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # Create a convolutional block with batch normalization and LeakyReLU activation.
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, X, labels):
        # Forward pass through the Discriminator
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)  # Reshape embedding
        X = torch.cat([X, embedding], dim=1)  # Concatenate input image and class embedding
        return self.disc(X)  # Pass through the discriminator network
    

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
        # Initialize the Generator with the specified latent dimension, image channels, features,
        # number of classes, image size, and embedding size.
        super(Generator, self).__init__()
        self.img_size = img_size  # Store the image size
        self.gen = nn.Sequential(
            self._block(z_dim + embed_size, features_g * 16, 4, 1, 0),  # Initial block from latent space and embeddings
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # Upsampling block
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # Upsampling block
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # Upsampling block
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1,
            ),  # Final transpose convolution to output the image
            nn.Tanh(),  # Tanh activation to get pixel values in range [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_size)  # Class embedding

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # Create an upsampling block with batch normalization and ReLU activation.
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, X, labels):
        # Forward pass through the Generator
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)  # Reshape embedding for concatenation
        X = torch.cat([X, embedding], dim=1)  # Concatenate latent vector and class embedding
        return self.gen(X)  # Pass through the generator network

def initialize_weights(model):
    # Initialize the weights of the model
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # Normal initialization with mean 0 and std 0.02
