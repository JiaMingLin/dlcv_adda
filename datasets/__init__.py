from .datagen import DataGenerator
from .mnist import get_mnist
from .usps import get_usps
from .mnist_m import get_mnistm
from .svhn import get_svhn

__all__ = (get_usps, get_mnist,get_mnistm, get_svhn)
