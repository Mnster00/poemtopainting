import torch
from torchvision.transforms import ToPILImage
from models.encoder import PaintingEncoder, PoemEncoder
from models.generator import PaintingGenerator, PoemGenerator
from config import config
from PIL import Image

def load_models():
    painting_encoder = PaintingEncoder(config.latent_dim).to(config.device)
    poem_encoder = PoemEncoder(config.vocab_size, config.latent_dim, config.latent_dim, config.latent_dim).to(config.device)
    painting_generator = PaintingGenerator(config.latent_dim, config.num_channels).to(config.device)
    poem_generator = PoemGenerator(config.latent_dim, config.vocab_size, config.latent_dim, config.latent_dim).to(config.device)

    painting_encoder.load_state_dict(torch.load('checkpoints/painting_encoder_latest.pth'))
    poem_encoder.load_state_dict(torch.load('checkpoints/poem_encoder_latest.pth'))
    painting_generator.load_state_dict(torch.load('checkpoints/painting_generator_latest.pth'))
    poem_generator.load_state_dict(torch.load('checkpoints/poem_generator_latest.pth'))

    painting_encoder.eval()
    poem_encoder.eval()
    painting_generator.eval()
    poem_generator.eval()

    return painting_encoder, poem_encoder, painting_generator, poem_generator

def generate_poem_from_painting(painting_path):
    painting_encoder, _, _, poem_generator = load_models()

    # Load and preprocess the painting
    painting = Image.open(painting_path).convert('RGB')
    painting = config.transform(painting).unsqueeze(0).to(config.device)

    with torch.no_grad():
        painting_latent = painting_encoder(painting)
        generated_poem = poem_generator(painting_latent, torch.zeros(1, config.max_poem_length).long().to(config.device))

    # Convert generated poem indices to characters
    poem_chars = [chr(idx) for idx in generated_poem[0].argmax(dim=1).cpu().numpy() if idx != 0]
    generated_poem = ''.join(poem_chars)

    return generated_poem

def generate_painting_from_poem(poem_text):
    _, poem_encoder, painting_generator, _ = load_models()

    # Preprocess the poem
    poem_indices = [ord(char) for char in poem_text]
    poem_tensor = torch.tensor(poem_indices).unsqueeze(0).long().to(config.device)

    with torch.no_grad():
        poem_latent = poem_encoder(poem_tensor)
        generated_painting = painting_generator(poem_latent)

    # Convert generated painting tensor to PIL Image
    generated_painting = generated_painting.squeeze(0).cpu()
    generated_painting = ToPILImage()(generated_painting)

    return generated_painting

if __name__ == '__main__':
    # Example usage
    painting_path =