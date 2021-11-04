from .lenet import L0LeNet, LeNet1D, LeNet_3C
from .model_loader import model_loader
# from .vae import VAE
from .vae_v2 import VAE_1D

from .resnet import resnet18_1d, resnet34_1d, resnet50_1d
from .vae_gan import VaeGan_pert
from .vae_gan_mini import VAE_2d

from .latent_classifier import LatentClf

from .har_clf import HARClassifier