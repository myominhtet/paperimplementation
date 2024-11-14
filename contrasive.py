from dataset import train_loader, test_loader
from model import model
import torch
import torch.nn as nn
import torch.optim as optimizer
from tqdm import tqdm

# Define the contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) + 
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

for epoch in range(10):
    model.train()
    for (img1, img2), label in train_loader:  # train_loader should yield paired data
        img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()

        optimizer.zero_grad()
        output1 = model(img1)
        output2 = model(img2)

        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')
