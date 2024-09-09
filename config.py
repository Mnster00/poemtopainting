import torch

class Config:
    # Data
    data_path = './data/CPDD'
    batch_size = 32
    num_workers = 4

    # Model
    latent_dim = 512
    num_channels = 3
    image_size = (256, 512)
    max_poem_length = 80
    vocab_size = 5000

    # Training
    num_epochs = 100
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    lambda_cycle = 10.0
    lambda_adv = 1.0
    lambda_sup = 5.0

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
