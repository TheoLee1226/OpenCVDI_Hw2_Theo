import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


class MNIST_Classifier:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(self.device)
        else:
            device_name = "CPU"
        print(f"Device: {self.device} ({device_name})")
        
        self.num_classes = 10
        self.learning_rate = 0.00005
        self.epochs = 50
        
        self.model = self.build_model().to(self.device)
    
    def build_model(self):
        model = torchvision.models.vgg16_bn(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, self.num_classes)
        return model
    
    def show_model(self):
        return torchsummary.summary(self.model, (1, 32, 32))
    
    def train_model(self):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ])

        train_batch_size = 150
        test_batch_size = 150

        train_ds = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_ds = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                targets = torch.zeros_like(outputs)

                for n in range(len(labels)):
                    targets[n][labels[n]] = 1

                targets = targets.to(self.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                loss_val += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Accuracy': f"{100. * (predicted == labels).sum().item() / labels.size(0):.2f}"})

            epoch_accuracy = 100. * correct / total
            train_losses.append(loss_val / len(train_loader))
            train_accuracies.append(epoch_accuracy)

            loss_val_test, accuracy_val_test = self.test_model(test_loader, criterion)
            test_losses.append(loss_val_test)
            test_accuracies.append(accuracy_val_test)

            print(f"Epoch {epoch+1}/{self.epochs}: Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_accuracies[-1]:.2f}%, Validation Loss: {test_losses[-1]:.4f}, Validation Accuracy: {test_accuracies[-1]:.2f}%")

            if accuracy_val_test >= best_accuracy:
                best_epoch = epoch + 1
                best_accuracy = accuracy_val_test
                best_model_wts = self.model.state_dict()

        print("Finished Training")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if best_model_wts is not None:
            save_dir = 'Q1_saved_models'
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
            save_dir = 'Q1_saved_lists'
            os.makedirs(save_dir, exist_ok=True)

            csv_save_path = os.path.join(save_dir, f'lists_{timestamp}.csv')
            df.to_csv(csv_save_path, index=False)
            print(f"Training and validation lists saved to '{csv_save_path}'")

            self.show_accuracy_and_loss(csv_save_path, True, timestamp)

    def test_model(self, test_loader, criterion):
        self.model.eval()

        loss_val = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
            
                # Convert labels to a matrix with 0 and 1
                targets = torch.zeros_like(outputs)
                for n in range(len(labels)):
                    targets[n][labels[n]] = 1

                loss = criterion(outputs, targets)
                loss_val += loss.item()

                _, predicted = torch.max(outputs, 1)  # Use outputs instead of targets
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = loss_val / len(test_loader)
        test_accuracy = 100. * correct / total
        return test_loss, test_accuracy

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def predict_model(self, image):
        self.model.eval()

        image = image.convert('L')
        image = image.resize((32, 32))

        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Normalize(mean=[0.5], std=[0.5])(image)

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            _, predicted = torch.max(output.data, 1)
            
        return predicted.item(), output

    def show_accuracy_and_loss(self, csv_path, save_figuere = True, timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')):
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
            save_dir = 'Q1_saved_figures'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'figure_{timestamp}.png')
            plt.savefig(save_path)
            print(f"Figure saved to '{save_path}'")

        plt.show()

    def show_probabilities(self, output):

        probabilities = nn.functional.softmax(output, dim=1)
        probabilities = probabilities.cpu().numpy().squeeze()

        return probabilities 

if __name__ == '__main__':
    classifier = MNIST_Classifier()

    #classifier.train_model()

    #classifier.load_model('saved_models\model_20241219_220119.pth')
    classifier.show_model()

    image_paths = ['Q1_InferenceData/test_1.jpg',
                  'Q1_InferenceData/test_2.jpg',
                  'Q1_InferenceData/test_3.jpg']

    for image_path in image_paths:

        image = Image.open(image_path)

        ans, output = classifier.predict_model(image)

        print(ans)

        print(classifier.show_probabilities(output))
