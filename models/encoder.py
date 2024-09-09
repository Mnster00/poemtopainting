import torch
import torch.nn as nn
import torchvision.models as models

class PaintingEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PaintingEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, latent_dim)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

class PoemEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_dim):
        super(PoemEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)
