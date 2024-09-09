import torch
import torch.nn as nn

class PaintingGenerator(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super(PaintingGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x.unsqueeze(-1).unsqueeze(-1))

class PoemGenerator(nn.Module):
    def __init__(self, latent_dim, vocab_size, embed_size, hidden_size):
        super(PoemGenerator, self).__init__()
        self.lstm = nn.LSTM(latent_dim + embed_size, hidden_size, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, latent, target, teacher_forcing_ratio=0.5):
        batch_size, max_len = target.size()
        vocab_size = self.fc.out_features
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(target.device)

        input = target[:, 0]
        hidden = None

        for t in range(1, max_len):
            input_emb = self.embedding(input).unsqueeze(1)
            input_latent = latent.unsqueeze(1).repeat(1, 1, 1)
            input_combined = torch.cat((input_emb, input_latent), dim=2)

            output, hidden = self.lstm(input_combined, hidden)
            output = self.fc(output.squeeze(1))
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = target[:, t] if teacher_force else output.argmax(1)

        return outputs
