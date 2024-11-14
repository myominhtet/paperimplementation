from dataset import train_loader, test_loader
from model import model
import torch
import torch.nn as nn
import torch.optim as optimizer
from tqdm import tqdm

criterion = nn.BCELoss()
optimizer = optimizer.AdamW(model.parameters(), lr=0.001)
device = 'cuda' if torch.cuda.is_available() else  'cpu'
epoch = 10
for epoch in range(epoch):
    model.train()
    running_loss = 0.0
    for image, label in tqdm(train_loader):
        optimizer.zero_grad()
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(torch.sigmoid(output), label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss/len(train_loader)
    print(f'Epoch [{epoch+1}/{epoch}], Loss: {avg_loss:.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            output = torch.sigmoid(output)
            predicted = (output > 0.5).float()

            correct += (predicted==label).sum().item()
            total +=  label.numel()
    accuracy = (correct / total) * 100
    print(f'Test Accuracy after Epoch {epoch+1}: {accuracy:.2f}%')