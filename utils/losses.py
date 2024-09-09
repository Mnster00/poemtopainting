import torch
import torch.nn as nn

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, x_reconstructed):
        return self.l1_loss(x_reconstructed, x)

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, real_output, fake_output):
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2

class SupervisedLoss(nn.Module):
    def __init__(self):
        super(SupervisedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, generated, target):
        return self.l1_loss(generated, target)
