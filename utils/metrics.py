import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def calculate_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Calculate the inception score of the generated images."""
    N = len(imgs)
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
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

def calculate_distribution_consistency_error(painting_features, poem_features):
    """Calculate Distribution Consistency Error (DCE)."""
    mu1, sigma1 = torch.mean(painting_features, dim=0), torch.cov(painting_features.t())
    mu2, sigma2 = torch.mean(poem_features, dim=0), torch.cov(poem_features.t())
    
    diff = torch.sum((mu1 - mu2)**2)
    covmean = torch.matrix_power(sigma1.mm(sigma2), 0.5)
    
    dce = diff + torch.trace(sigma1 + sigma2 - 2*covmean)
    return dce.item()