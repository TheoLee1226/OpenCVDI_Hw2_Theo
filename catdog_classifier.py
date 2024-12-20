import torch
from torch.utils.data import DataLoader

import torchsummary
import torchvision

from torch import nn
from torch import optim

from PIL import Image
from PIL import ImageFile

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime

import my_catdog_dataset

from torchvision.transforms import ToPILImage


class CatDog_Classifier:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(self.device)
        else:
            device_name = "CPU"

        print(f"Device: {self.device} ({device_name})")
    
        self.num_classes = 2
        self.learning_rate = 0.0001
        self.epochs = 20

        self.model = self.build_model().to(self.device)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        warnings.filterwarnings("ignore", message=".*Truncated File Read.*")

    def build_model(self):
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, self.num_classes, torch.nn.Softmax()))
        return model
    
    def show_model(self):
        return torchsummary.summary(self.model, (3, 224, 224))
    
    def train_model(self, erasing = False):

        if erasing:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.RandomErasing(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),          
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),  
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])       
            ])

        train_batch_size = 100
        test_batch_size = 100

        train_ds = my_catdog_dataset.My_CatDog_Dataset(train=True, transform=transforms)
        test_ds = my_catdog_dataset.My_CatDog_Dataset(train=False, transform=transforms)

        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

        best_accuracy = 0.0
        best_model_wts = None
        best_epoch = 0

        train_losses = []
        test_losses = []

        train_accuracies = []
        test_accuracies = []

        self.model.train()

        print("Start Training")

        for epoch in range(self.epochs):

            train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

            loss_val = 0.0
            correct = 0
            total = 0 

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{self.epochs}")

            for i, (images, labels) in progress_bar:

                images = images.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                targets = []

                for label in labels:
                    if label == 'cat':
                        targets.append([0.0,1.0])
                    else:
                        targets.append([1.0,0.0])

                targets = torch.tensor(targets).to(self.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                loss_val += loss.item()

                _, predicted = torch.max(outputs, 1)
                _, answer = torch.max(targets, 1)

                total += targets.size(0)
                correct += (predicted == answer).sum().item()

                progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Accuracy': f"{100. * correct / total:.2f}"})         
            
            epoch_accuracy = 100. * correct / total
            train_losses.append(loss_val / len(train_loader))
            train_accuracies.append(epoch_accuracy)

            loss_val_test, accuracy_val_test = self.test_model(test_loader, criterion)
            test_losses.append(loss_val_test)
            test_accuracies.append(accuracy_val_test)
            learning_rate_now = optimizer.param_groups[0]['lr']

            scheduler.step(loss_val_test)

            print(f"Epoch {epoch+1}/{self.epochs}: Learning Rate = {learning_rate_now:.1e} Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_accuracies[-1]:.2f}%, Validation Loss: {test_losses[-1]:.4f}, Validation Accuracy: {test_accuracies[-1]:.2f}%")

            if accuracy_val_test >= best_accuracy:
                best_epoch = epoch + 1
                best_accuracy = accuracy_val_test
                best_model_wts = self.model.state_dict()

        print("Finished Training")    
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if best_model_wts is not None:
            if erasing:
                save_dir = 'Q2_saved_models_erasing'
            else:
                save_dir = 'Q2_saved_models'
            os.makedirs(save_dir, exist_ok=True)
                      
            save_path = os.path.join(save_dir, f'model_{timestamp}.pth')
            
            torch.save(best_model_wts, save_path)
            print(f"Best model is '{best_epoch}' , and weights saved to '{save_path}'")

        if train_losses and test_losses and train_accuracies and test_accuracies is not None:
            df = pd.DataFrame({
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
            })
            if erasing:
                save_dir = 'Q2_saved_lists_erasing'
            else:
                save_dir = 'Q2_saved_lists'
            os.makedirs(save_dir, exist_ok=True)

            csv_save_path = os.path.join(save_dir, f'lists_{timestamp}.csv')
            df.to_csv(csv_save_path, index=False)
            print(f"Training and validation lists saved to '{csv_save_path}'")

            if erasing:
                self.show_accuracy_and_loss(csv_save_path, True, timestamp, 'Q2_saved_figures_erasing')
            else:  
                self.show_accuracy_and_loss(csv_save_path, True, timestamp)
        
    def test_model(self, test_loader, criterion):
        self.model.eval()

        loss_val = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)

                targets = []

                for label in labels:
                    if label == 'cat':
                        targets.append([0.0,1.0])
                    else:
                        targets.append([1.0,0.0])

                targets = torch.tensor(targets).to(self.device)

                loss = criterion(outputs, targets)
                loss_val += loss.item()

                _, predicted = torch.max(outputs, 1) 
                _, answer = torch.max(targets, 1)

                total += targets.size(0)
                correct += (predicted == answer).sum().item()

        test_loss = loss_val / len(test_loader)
        test_accuracy = 100. * correct / total
        return test_loss, test_accuracy
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def predict_model(self, image):
        self.model.eval()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224)
        ])

        image = transforms(image).unsqueeze(0)

        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            _, predicted = torch.max(output.data, 1)

        predicted = 'Cat' if predicted.item() == 1 else 'Dog' 
        
        probabilities = nn.functional.softmax(output, dim=1)
        probabilities = probabilities.cpu().numpy().flatten()
        output = output.cpu().numpy().flatten()

        return predicted, probabilities
    
    def show_accuracy_and_loss(self, csv_path, save_figuere = True, timestamp = datetime.now().strftime('%Y%m%d_%H%M%S'), save_dir = 'Q2_saved_figures'):
        df = pd.read_csv(csv_path)
        
        epochs_range = range(1, len(df) + 1)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, df['train_losses'], label='Training Loss')
        plt.plot(epochs_range, df['test_losses'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, df['train_accuracies'], label='Training Accuracy')
        plt.plot(epochs_range, df['test_accuracies'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()

        if save_figuere:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'figure_{timestamp}.png')
            plt.savefig(save_path)
            print(f"Figure saved to '{save_path}'")

        plt.show()

if __name__ == '__main__':
    cc = CatDog_Classifier()
    cc.load_model('Q2_saved_models\model_20241220_221231.pth')
    print(cc.predict_model(Image.open('Q2_Dataset/inference_dataset/Dog/12051.jpg')))
