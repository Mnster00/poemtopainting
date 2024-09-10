import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, resnet18
from scipy.linalg import sqrtm
import numpy as np
import nltk
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA

def calculate_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Calculate the inception score of the generated images."""
    N = len(imgs)
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    dataloader = DataLoader(imgs, batch_size=batch_size)
    model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = model(x)
        return F.softmax(x).data.cpu().numpy()

    preds = np.zeros((N, 1000))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
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

def calculate_mce(generated_poem, gpt2_model):
    """Calculate Mean Cross-Entropy Error (MCE) for a generated poem."""
    tokenized_poem = gpt2_model.tokenizer.encode(generated_poem)
    input_ids = torch.tensor([tokenized_poem]).to(gpt2_model.device)
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss
    return loss.item()

def calculate_mte(paintings, poem_generator, gpt2_model, k=5, p=0.9):
    """Calculate Mean Top-k Cross Entropy (MTE) for generated poems."""
    total_ce = 0
    total_poems = 0
    
    for painting in paintings:
        for _ in range(k):
            generated_poem = poem_generator.generate(painting, do_sample=True, top_p=p)
            ce = calculate_mce(generated_poem, gpt2_model)
            total_ce += ce
            total_poems += 1
    
    return total_ce / total_poems

def calculate_distribution_consistency_error(painting_features, poem_features):
    """Calculate Distribution Consistency Error (DCE) with improvements."""
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=100)
    painting_features_pca = pca.fit_transform(painting_features.cpu().numpy())
    poem_features_pca = pca.transform(poem_features.cpu().numpy())
    
    # Calculate mean and covariance using Ledoit-Wolf shrinkage
    mu1, sigma1 = np.mean(painting_features_pca, axis=0), ledoit_wolf(painting_features_pca)[0]
    mu2, sigma2 = np.mean(poem_features_pca, axis=0), ledoit_wolf(poem_features_pca)[0]
    
    diff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    dce = diff + np.trace(sigma1 + sigma2 - 2*covmean)
    return dce

# Utility function to extract features from paintings or poems
def extract_features(model, data_loader):
    features = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch_features = model(batch)
            features.append(batch_features)
    return torch.cat(features, dim=0)

# Example usage of the new metrics
def evaluate_model(poem_painting_model, test_data, gpt2_model):
    painting_encoder = resnet18(pretrained=False)
    painting_encoder.fc = nn.Linear(512, 512)  # Modify the last layer to output 512-dim features
    painting_encoder.load_state_dict(torch.load('path_to_pretrained_painting_encoder.pth'))
    
    poem_encoder = nn.Sequential(
        nn.Embedding(vocab_size, 512),
        nn.LSTM(512, 512, bidirectional=True),
        nn.Linear(1024, 512)
    )
    poem_encoder.load_state_dict(torch.load('path_to_pretrained_poem_encoder.pth'))
    
    painting_features = extract_features(painting_encoder, test_data.painting_loader)
    poem_features = extract_features(poem_encoder, test_data.poem_loader)
    
    generated_paintings = poem_painting_model.generate_paintings(test_data.poems)
    generated_poems = poem_painting_model.generate_poems(test_data.paintings)
    
    # Calculate metrics
    fid_score = calculate_fid(painting_features, extract_features(painting_encoder, generated_paintings))
    bleu_score = calculate_bleu(test_data.poems, generated_poems)
    meteor_score = calculate_meteor(test_data.poems, generated_poems)
    perplexity = calculate_perplexity(gpt2_model, generated_poems)
    mce_score = calculate_mce(generated_poems, gpt2_model)
    mte_score = calculate_mte(test_data.paintings, poem_painting_model.poem_generator, gpt2_model)
    dce_score = calculate_distribution_consistency_error(painting_features, poem_features)
    
    return {
        'FID': fid_score,
        'BLEU': bleu_score,
        'METEOR': meteor_score,
        'Perplexity': perplexity,
        'MCE': mce_score,
        'MTE': mte_score,
        'DCE': dce_score
    }
