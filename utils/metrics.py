import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, resnet18
from scipy.linalg import sqrtm
import numpy as np
import nltk
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import LSTM, Linear
from sklearn.decomposition import PCA
from sklearn.covariance import ledoit_wolf

def calculate_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Calculate the inception score of the generated images."""
    N = len(imgs)
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = model(x)
        return F.softmax(x).data.cpu().numpy()

    preds = np.zeros((N, 1000))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_fid(real_features, fake_features):
    """Calculate the Frechet Inception Distance (FID) score."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score for generated text."""
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

def calculate_meteor(reference, hypothesis):
    """Calculate METEOR score for generated text."""
    return nltk.translate.meteor_score.meteor_score([reference], hypothesis)

def calculate_perplexity(model, text):
    """Calculate perplexity of generated text."""
    tokenized_text = tokenize(text)
    tensor = torch.tensor(tokenized_text, dtype=torch.long).to(device)
    with torch.no_grad():
        loss = model(tensor, labels=tensor).loss
    return math.exp(loss.item())

def calculate_distribution_consistency_error(painting_features, poem_features):
    """Calculate Distribution Consistency Error (DCE)."""
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=100)
    painting_features_pca = pca.fit_transform(painting_features.cpu().numpy())
    poem_features_pca = pca.transform(poem_features.cpu().numpy())

    # Use Ledoit-Wolf shrinkage for covariance estimation
    painting_cov = ledoit_wolf(painting_features_pca)[0]
    poem_cov = ledoit_wolf(poem_features_pca)[0]

    mu1, sigma1 = np.mean(painting_features_pca, axis=0), painting_cov
    mu2, sigma2 = np.mean(poem_features_pca, axis=0), poem_cov
    
    diff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    dce = diff + np.trace(sigma1 + sigma2 - 2*covmean)
    return dce

def calculate_mce(generated_poem, gpt2_model, gpt2_tokenizer):
    """Calculate Mean Cross-Entropy Error (MCE)."""
    inputs = gpt2_tokenizer(generated_poem, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()

def calculate_mte(paintings, poem_generator, gpt2_model, gpt2_tokenizer, k=5):
    """Calculate Mean Top-k Cross Entropy (MTE)."""
    total_ce = 0
    for painting in paintings:
        for _ in range(k):
            generated_poem = poem_generator(painting)
            total_ce += calculate_mce(generated_poem, gpt2_model, gpt2_tokenizer)
    return total_ce / (len(paintings) * k)

class PoemEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(PoemEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.bilstm = torch.nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        return self.fc(output[:, -1, :])

def load_pretrained_models():
    """Load pre-trained models for evaluation."""
    # Load pre-trained ResNet-18 for painting encoding
    painting_encoder = resnet18(pretrained=True)
    painting_encoder.fc = Linear(painting_encoder.fc.in_features, 512)
    
    # Load pre-trained GPT2-Chinese model
    gpt2_model = GPT2LMHeadModel.from_pretrained("ckiplab/gpt2-base-chinese")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("ckiplab/gpt2-base-chinese")
    
    # Initialize PoemEncoder (you'll need to train this separately)
    poem_encoder = PoemEncoder(vocab_size=len(gpt2_tokenizer), embed_size=256, hidden_size=256, output_size=512)
    
    return painting_encoder, gpt2_model, gpt2_tokenizer, poem_encoder

# Usage example:
# painting_encoder, gpt2_model, gpt2_tokenizer, poem_encoder = load_pretrained_models()
# 
# # Calculate MCE
# generated_poem = "Your generated poem here"
# mce = calculate_mce(generated_poem, gpt2_model, gpt2_tokenizer)
# 
# # Calculate MTE
# paintings = [painting1, painting2, ...]  # Your list of paintings
# mte = calculate_mte(paintings, poem_generator_function, gpt2_model, gpt2_tokenizer)
# 
# # Calculate DCE
# painting_features = painting_encoder(paintings)
# poem_features = poem_encoder(poems)
# dce = calculate_distribution_consistency_error(painting_features, poem_features)
