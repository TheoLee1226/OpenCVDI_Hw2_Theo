import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class My_CatDog_Dataset(Dataset):
    def __init__(self, train=True, transform=None):
        if train:
            self.image_folder = 'Q2_Dataset/training_dataset'
        else:
            self.image_folder = 'Q2_Dataset/validation_dataset'

        self.image_paths = []
        self.labels = []
        
        for class_name, label in [('Cat', 'cat'), ('Dog', 'dog')]:
            class_folder = os.path.join(self.image_folder, class_name)
            for img in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, img))
                self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

if __name__ == '__main__':

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = My_CatDog_Dataset(train=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for images, labels in dataloader:
        print(images.shape, labels)
    
    