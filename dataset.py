import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


data = np.load('Wafer_Map_Datasets.npz')
images = data['arr_0']
labels = data['arr_1']
flattened_images = images.reshape(images.shape[0], -1)
df_images = pd.DataFrame(flattened_images, columns=[f'pixel_{i}' for i in range(flattened_images.shape[1])])
df_labels = pd.DataFrame(labels, columns=[f'labels_{i}' for i in range(labels.shape[1])])
df_labels_list = df_labels.apply(lambda row: ' '.join(row.astype(str)), axis=1).to_frame(name='label')
df = pd.concat([df_images, df_labels_list], axis=1)
label_mapping = {
    '0 0 0 0 0 0 0 0': 'Normal',
    '1 0 0 0 0 0 0 0': 'Center(C)',
    '0 1 0 0 0 0 0 0': 'Donut(D)',
    '0 0 1 0 0 0 0 0': 'Edge_Loc(EL)',
    '0 0 0 1 0 0 0 0': 'Edge_Ring(ER)',
    '0 0 0 0 1 0 0 0': 'Loc(L)',
    '0 0 0 0 0 1 0 0': 'Near_Full(NF)',
    '0 0 0 0 0 0 1 0': 'Scratch(S)',
    '0 0 0 0 0 0 0 1': 'Random(R)',
    '1 0 1 0 0 0 0 0': 'C+EL',
    '1 0 0 1 0 0 0 0': 'C+ER',
    '1 0 0 0 1 0 0 0': 'C+L',
    '1 0 0 0 0 0 1 0': 'C+S',
    '0 1 1 0 0 0 0 0': 'D+EL',
    '0 1 0 1 0 0 0 0': 'D+ER',
    '0 1 0 0 1 0 0 0': 'D+L',
    '0 1 0 0 0 0 1 0': 'D+S',
    '0 0 1 0 1 0 0 0': 'EL+L',
    '0 0 1 0 0 0 1 0': 'EL+S',
    '0 0 0 1 1 0 0 0': 'ER+L',
    '0 0 0 1 0 0 1 0': 'ER+S',
    '0 0 0 0 1 0 1 0': 'L+S',
    '1 0 1 0 1 0 0 0': 'C+EL+L',
    '1 0 1 0 0 0 1 0': 'C+EL+S',
    '1 0 0 1 1 0 0 0': 'C+ER+L',
    '1 0 0 1 0 0 1 0': 'C+ER+S',
    '1 0 0 0 1 0 1 0': 'C+L+S',
    '0 1 1 0 1 0 0 0': 'D+EL+L',
    '0 1 1 0 0 0 1 0': 'D+EL+S',
    '0 1 0 1 1 0 0 0': 'D+ER+L',
    '0 1 0 1 0 0 1 0': 'D+ER+S',
    '0 1 0 0 1 0 1 0': 'D+L+S',
    '0 0 1 0 1 0 1 0': 'EL+L+S',
    '0 0 0 1 1 0 1 0': 'ER+L+S',
    '1 0 1 0 1 0 1 0': 'C+L+EL+S',
    '1 0 0 1 1 0 1 0': 'C+L+ER+S',
    '0 1 1 0 1 0 1 0': 'D+L+EL+S',
    '0 1 0 1 1 0 1 0': 'D+L+ER+S'
}
df['labels'] = df.label.map(label_mapping)
df = df.drop(columns=['label'])

encoder = OneHotEncoder(sparse_output=False)
encoded_labels = encoder.fit_transform(df[['labels']])
one_hot_labels = pd.DataFrame(encoded_labels, columns=encoder.categories_[0])
df_train = df.loc[: , 'pixel_0':'pixel_2703']
x_train, x_test, y_train, y_test = train_test_split(df_train, one_hot_labels,test_size=0.2 ,random_state=42, stratify=one_hot_labels)


# Define your Dataset class (as you have already defined)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images.iloc[idx].values.astype(np.float32)).reshape(1, 52, 52)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
        return image, label
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.RandomRotation(degrees=15),
    # transforms.Resize((224, 224))
])
# Assuming you have your train_dataset already created
train_dataset = Dataset(x_train, y_train, transform=transform)
test_dataset = Dataset(x_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
