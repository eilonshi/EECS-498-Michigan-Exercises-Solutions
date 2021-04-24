from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
NOISE_DIM = 96

def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
  """
  Generate a PyTorch Tensor of uniform random noise.

  Input:
  - batch_size: Integer giving the batch size of noise to generate.
  - noise_dim: Integer giving the dimension of noise to generate.
  
  Output:
  - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
    random noise in the range (-1, 1).
  """
  noise = None
  ##############################################################################
  # TODO: Implement sample_noise.                                              #
  ##############################################################################
  # Replace "pass" statement with your code

  noise = 2 * (torch.rand((batch_size, noise_dim), dtype=dtype, device=device) - 0.5)

  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################

  return noise



def discriminator():
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement discriminator.                                           #
  ############################################################################
  # Replace "pass" statement with your code

  input_dim = 784
  hiddem_dim = 256
  output_dim = 1
  alpha = 0.01

  model = nn.Sequential(
    nn.Linear(input_dim, hiddem_dim),
    nn.LeakyReLU(alpha),
    nn.Linear(hiddem_dim, hiddem_dim),
    nn.LeakyReLU(alpha),
    nn.Linear(hiddem_dim, output_dim)
  )

  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  
  return model


def generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement generator.                                               #
  ############################################################################
  # Replace "pass" statement with your code

  hiddem_dim = 1024
  output_dim = 784
  alpha = 0.01

  model = nn.Sequential(
    nn.Linear(noise_dim, hiddem_dim),
    nn.ReLU(),
    nn.Linear(hiddem_dim, hiddem_dim),
    nn.ReLU(),
    nn.Linear(hiddem_dim, output_dim),
    nn.Tanh()
  )

  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return model  

def discriminator_loss(logits_real, logits_fake):
  """
  Computes the discriminator loss described above.
  
  Inputs:
  - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement discriminator_loss.                                        #
  ##############################################################################
  # Replace "pass" statement with your code

  target_real = torch.ones_like(logits_real, device='cuda')
  target_fake = torch.zeros_like(logits_fake, device='cuda')
  
  real_data_loss = nn.functional.binary_cross_entropy_with_logits(logits_real, target_real)
  fake_data_loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, target_fake)

  loss = real_data_loss + fake_data_loss

  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss

def generator_loss(logits_fake):
  """
  Computes the generator loss described above.

  Inputs:
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing the (scalar) loss for the generator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement generator_loss.                                            #
  ##############################################################################
  # Replace "pass" statement with your code

  target_fake = torch.ones_like(logits_fake, device='cuda')
  
  loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, target_fake)

  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss

def get_optimizer(model):
  """
  Construct and return an Adam optimizer for the model with learning rate 1e-3,
  beta1=0.5, and beta2=0.999.
  
  Input:
  - model: A PyTorch model that we want to optimize.
  
  Returns:
  - An Adam optimizer for the model with the desired hyperparameters.
  """
  optimizer = None
  ##############################################################################
  # TODO: Implement optimizer.                                                 #
  ##############################################################################
  # Replace "pass" statement with your code

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
  """
  Compute the Least-Squares GAN loss for the discriminator.
  
  Inputs:
  - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_discriminator_loss.                                     #
  ##############################################################################
  # Replace "pass" statement with your code
  
  real_loss = 0.5 * ((scores_real - 1) ** 2).mean()
  fake_loss = 0.5 * (scores_fake ** 2).mean()

  loss = real_loss + fake_loss

  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss

def ls_generator_loss(scores_fake):
  """
  Computes the Least-Squares GAN loss for the generator.
  
  Inputs:
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_generator_loss.                                         #
  ##############################################################################
  # Replace "pass" statement with your code

  loss = 0.5 * ((scores_fake - 1) ** 2).mean()
  
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss


def build_dc_classifier():
  """
  Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
  the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_classifier.                                     #
  ############################################################################
  # Replace "pass" statement with your code

  image_width_height = 28
  image_size = image_width_height ** 2
  hidden_size = 4 * 4 * 64
  out_size = 1
  alpha = 0.01
  kernel_size = 5
  padding_size = 2
  stride_conv = 1
  pool_size = 2
  num_filters1 = 32
  num_filters2 = 64
  input_channels = 1

  model = nn.Sequential(
    nn.Unflatten(1, (input_channels, image_width_height, image_width_height)),
    nn.Conv2d(in_channels=input_channels, out_channels=num_filters1, kernel_size=kernel_size, stride=stride_conv),
    nn.LeakyReLU(alpha),
    nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
    nn.Conv2d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size, stride=stride_conv),
    nn.LeakyReLU(alpha),
    nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
    nn.Flatten(),
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.LeakyReLU(alpha),
    nn.Linear(in_features=hidden_size, out_features=out_size)
  )

  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return model

def build_dc_generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
  the architecture described in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_generator.                                      #
  ############################################################################
  # Replace "pass" statement with your code

  image_size = 96
  hidden_width_height = 7

  hidden_channels = 128
  hidden_channels2 = 64
  hidden_channels3 = 1  

  hidden_size = 1024
  hidden_size2 = hidden_width_height ** 2 * hidden_channels

  kernel_size = 4
  stride = 2
  padding = 1

  model = nn.Sequential(
    nn.Linear(in_features=image_size, out_features=hidden_size),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=hidden_size),
    nn.Linear(in_features=hidden_size, out_features=hidden_size2),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=hidden_size2),
    nn.Unflatten(1, (hidden_channels, hidden_width_height, hidden_width_height)),
    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels2, kernel_size=kernel_size, stride=stride, padding=padding),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=hidden_channels2),
    nn.ConvTranspose2d(in_channels=hidden_channels2, out_channels=hidden_channels3, kernel_size=kernel_size, stride=stride, padding=padding),
    nn.Tanh(),
    nn.Flatten()
  )
  
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return model
