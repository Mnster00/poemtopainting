import torch
from torch.optim import Adam
from models.encoder import PaintingEncoder, PoemEncoder
from models.generator import PaintingGenerator, PoemGenerator
from models.discriminator import PaintingDiscriminator, PoemDiscriminator
from utils.losses import CycleConsistencyLoss, AdversarialLoss, SupervisedLoss
from data.dataloader import get_dataloader
from config import config

def train():
    # Initialize models
    painting_encoder = PaintingEncoder(config.latent_dim).to(config.device)
    poem_encoder = PoemEncoder(config.vocab_size, config.latent_dim, config.latent_dim, config.latent_dim).to(config.device)
    painting_generator = PaintingGenerator(config.latent_dim, config.num_channels).to(config.device)
    poem_generator = PoemGenerator(config.latent_dim, config.vocab_size, config.latent_dim, config.latent_dim).to(config.device)
    painting_discriminator = PaintingDiscriminator(config.num_channels).to(config.device)
    poem_discriminator = PoemDiscriminator(config.vocab_size, config.latent_dim, config.latent_dim).to(config.device)

    # Initialize optimizers
    optimizer_G = Adam(list(painting_encoder.parameters()) + list(poem_encoder.parameters()) +
                       list(painting_generator.parameters()) + list(poem_generator.parameters()),
                       lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_D = Adam(list(painting_discriminator.parameters()) + list(poem_discriminator.parameters()),
                       lr=config.learning_rate, betas=(config.beta1, config.beta2))

    # Initialize losses
    cycle_loss = CycleConsistencyLoss()
    adv_loss = AdversarialLoss()
    sup_loss = SupervisedLoss()

    # Get data loader
    dataloader = get_dataloader('train')

    # Training loop
    for epoch in range(config.num_epochs):
        for i, (real_paintings, real_poems) in enumerate(dataloader):
            real_paintings = real_paintings.to(config.device)
            real_poems = real_poems.to(config.device)

            # Forward pass
            painting_latent = painting_encoder(real_paintings)
            poem_latent = poem_encoder(real_poems)
            fake_poems = poem_generator(painting_latent, real_poems)
            fake_paintings = painting_generator(poem_latent)
            reconstructed_paintings = painting_generator(painting_encoder(fake_paintings))
            reconstructed_poems = poem_generator(poem_encoder(fake_poems), real_poems)

            # Compute losses
            cycle_painting_loss = cycle_loss(real_paintings, reconstructed_paintings)
            cycle_poem_loss = cycle_loss(real_poems, reconstructed_poems)
            adv_painting_loss = adv_loss(painting_discriminator(real_paintings), painting_discriminator(fake_paintings))
            adv_poem_loss = adv_loss(poem_discriminator(real_poems), poem_discriminator(fake_poems))
            sup_painting_loss = sup_loss(fake_paintings, real_paintings)
            sup_poem_loss = sup_loss(fake_poems, real_poems)

            # Total loss
            g_loss = (config.lambda_cycle * (cycle_painting_loss + cycle_poem_loss) +
                      config.lambda_adv * (adv_painting_loss + adv_poem_loss) +
                      config.lambda_sup * (sup_painting_loss + sup_poem_loss))

            # Update generators
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # Update discriminators
            d_painting_loss = adv_loss(painting_discriminator(real_paintings), painting_discriminator(fake_paintings.detach()))
            d_poem_loss = adv_loss(poem_discriminator(real_poems), poem_discriminator(fake_poems.detach()))
            d_loss = d_painting_loss + d_poem_loss

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{config.num_epochs}], Step [{i}/{len(dataloader)}], "
                      f"G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")

        # Save models
        torch.save(painting_encoder.state_dict(), f'checkpoints/painting_encoder_epoch_{epoch}.pth')
        torch.save(poem_encoder.state_dict(), f'checkpoints/poem_encoder_epoch_{epoch}.pth')
        torch.save(painting_generator.state_dict(), f'checkpoints/painting_generator_epoch_{epoch}.pth')
        torch.save(poem_generator.state_dict(), f'checkpoints/poem_generator_epoch_{epoch}.pth')

if __name__ == '__main__':
    train()
