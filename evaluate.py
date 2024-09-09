import torch
from models.encoder import PaintingEncoder, PoemEncoder
from models.generator import PaintingGenerator, PoemGenerator
from data.dataloader import get_dataloader
from utils.metrics import calculate_fid, calculate_bleu, calculate_meteor, calculate_perplexity, calculate_distribution_consistency_error
from config import config

def evaluate():
    # Load models
    painting_encoder = PaintingEncoder(config.latent_dim).to(config.device)
    poem_encoder = PoemEncoder(config.vocab_size, config.latent_dim, config.latent_dim, config.latent_dim).to(config.device)
    painting_generator = PaintingGenerator(config.latent_dim, config.num_channels).to(config.device)
    poem_generator = PoemGenerator(config.latent_dim, config.vocab_size, config.latent_dim, config.latent_dim).to(config.device)

    painting_encoder.load_state_dict(torch.load('checkpoints/painting_encoder_latest.pth'))
    poem_encoder.load_state_dict(torch.load('checkpoints/poem_encoder_latest.pth'))
    painting_generator.load_state_dict(torch.load('checkpoints/painting_generator_latest.pth'))
    poem_generator.load_state_dict(torch.load('checkpoints/poem_generator_latest.pth'))

    # Set models to evaluation mode
    painting_encoder.eval()
    poem_encoder.eval()
    painting_generator.eval()
    poem_generator.eval()

    # Get data loader
    dataloader = get_dataloader('test')

    real_paintings = []
    real_poems = []
    fake_paintings = []
    fake_poems = []

    with torch.no_grad():
        for paintings, poems in dataloader:
            paintings = paintings.to(config.device)
            poems = poems.to(config.device)

            painting_latent = painting_encoder(paintings)
            poem_latent = poem_encoder(poems)

            generated_poems = poem_generator(painting_latent, poems)
            generated_paintings = painting_generator(poem_latent)

            real_paintings.append(paintings)
            real_poems.append(poems)
            fake_paintings.append(generated_paintings)
            fake_poems.append(generated_poems)

    real_paintings = torch.cat(real_paintings, dim=0)
    real_poems = torch.cat(real_poems, dim=0)
    fake_paintings = torch.cat(fake_paintings, dim=0)
    fake_poems = torch.cat(fake_poems, dim=0)

    # Calculate FID for paintings
    fid_score = calculate_fid(real_paintings.cpu().numpy(), fake_paintings.cpu().numpy())

    # Calculate BLEU and METEOR for poems
    bleu_scores = []
    meteor_scores = []
    for real, fake in zip(real_poems, fake_poems):
        bleu_scores.append(calculate_bleu(real.cpu().tolist(), fake.cpu().tolist()))
        meteor_scores.append(calculate_meteor(real.cpu().tolist(), fake.cpu().tolist()))

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    # Calculate perplexity for poems
    perplexity = calculate_perplexity(poem_generator, fake_poems)

    # Calculate Distribution Consistency Error
    dce = calculate_distribution_consistency_error(painting_encoder(real_paintings), poem_encoder(real_poems))

    print(f"FID Score: {fid_score:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average METEOR Score: {avg_meteor:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Distribution Consistency Error: {dce:.4f}")

if __name__ == '__main__':
    evaluate()
