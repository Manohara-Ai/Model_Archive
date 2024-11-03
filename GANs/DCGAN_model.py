import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        # Initialize the Discriminator model
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # Initial convolution layer
            nn.LeakyReLU(0.2),  # Activation function
            self._block(features_d, features_d * 2, 4, 2, 1),  # Downsampling block
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # Downsampling block
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # Downsampling block
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # Final convolution layer
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # Create a convolutional block with BatchNorm and LeakyReLU
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),  # LeakyReLU activation for non-linearity
        )

    def forward(self, x):
        # Forward pass through the Discriminator
        return self.disc(x)  # Output logits for real or fake classification


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        # Initialize the Generator model
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4 from latent space
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # Output image size: 64x64
            nn.Tanh(),  # Tanh activation to output pixel values in range [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # Create an upsampling block with LeakyReLU activation
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),  # LeakyReLU activation for non-linearity
        )

    def forward(self, x):
        # Forward pass through the Generator
        return self.net(x)  # Generate an image from noise


def initialize_weights(model):
    # Initialize weights of the model using normal distribution
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # Normal initialization
