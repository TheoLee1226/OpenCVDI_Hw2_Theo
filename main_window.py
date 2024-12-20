import sys

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from PIL import Image
from PIL import ImageQt

import matplotlib.pyplot as plt   
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import MNIST_classifier
import catdog_classifier
import my_catdog_dataset

import torchvision
import torch

import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")

        self.mc_timestamp = "20241220_100744"
        self.cc_timestamp = "20241220_221231"
        self.cc_timestamp_erasing = "20241220_224341"

        self.mc = MNIST_classifier.MNIST_Classifier()
        self.mc.load_model(f"Q1_saved_models\model_{self.mc_timestamp}.pth")

        self.cc = catdog_classifier.CatDog_Classifier()
        self.cc_erasing = catdog_classifier.CatDog_Classifier()
    
        self.cc.load_model(f"Q2_saved_models\model_{self.cc_timestamp}.pth")
        self.cc_erasing.load_model(f"Q2_saved_models_erasing\model_{self.cc_timestamp_erasing}.pth")

        self.image1 = None
        self.image2 = None

        load_image_button = QPushButton("Load Image")
        load_video_button = QPushButton("Load Video")

        load_image_button.clicked.connect(self.load_image)
        load_video_button.clicked.connect(self.load_video)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(load_image_button)
        vbox1.addWidget(load_video_button)

        vbox1_widget = QWidget()
        vbox1_widget.setLayout(vbox1)
        vbox1_widget.setFixedSize(100, 400)

        show_structure_button = QPushButton("1.1 Show Structure")
        show_acc_loss_button = QPushButton("1.2 Show Acc and Loss")
        predict_button = QPushButton("1.3. Predict")
        self.predict_label = QLabel("predict")
        self.predict_label.setAlignment(Qt.AlignCenter)

        show_structure_button.clicked.connect(self.show_structure)
        show_acc_loss_button.clicked.connect(self.show_acc_loss)
        predict_button.clicked.connect(self.predict)

        vbox2_1 = QVBoxLayout()
        vbox2_1.addWidget(show_structure_button)
        vbox2_1.addWidget(show_acc_loss_button)
        vbox2_1.addWidget(predict_button)
        vbox2_1.addWidget(self.predict_label)

        vbox2_1_widget = QWidget()
        vbox2_1_widget.setLayout(vbox2_1)
        vbox2_1_widget.setFixedSize(200, 200)

        load_image_button2 = QPushButton("Q2 Load Image")
        show_image_button = QPushButton("2.1 Show Image")
        show_model_button = QPushButton("2.2 Show Model Structure")
        show_comparison_button = QPushButton("2.3 Show Comprasion")
        inference_button = QPushButton("2.4 Inference")
        self.inference_label = QLabel("TextLabel")
        self.inference_label.setAlignment(Qt.AlignCenter)

        load_image_button2.clicked.connect(self.load_image2)
        show_image_button.clicked.connect(self.show_image)
        show_model_button.clicked.connect(self.show_model)
        show_comparison_button.clicked.connect(self.show_comparison)
        inference_button.clicked.connect(self.inference)

        vbox2_2 = QVBoxLayout()
        vbox2_2.addWidget(load_image_button2)
        vbox2_2.addWidget(show_image_button)
        vbox2_2.addWidget(show_model_button)
        vbox2_2.addWidget(show_comparison_button)
        vbox2_2.addWidget(inference_button)
        vbox2_2.addWidget(self.inference_label)

        vbox2_2_widget = QWidget()
        vbox2_2_widget.setLayout(vbox2_2)
        vbox2_2_widget.setFixedSize(200, 200)

        vbox2_combined = QVBoxLayout()
        vbox2_combined.addWidget(vbox2_1_widget)
        vbox2_combined.addWidget(vbox2_2_widget)

        vbox2_combined_widget = QWidget()
        vbox2_combined_widget.setLayout(vbox2_combined)

        self.image_label1 = QLabel()
        self.image_label1.setAlignment(Qt.AlignCenter)
        self.image_label1.setScaledContents(True)
        self.image_label1.setFixedSize(200, 200) 

        self.image_label2 = QLabel()
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        self.image_label2.setFixedSize(200, 200)

        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.image_label1)
        vbox3.addWidget(self.image_label2)

        vbox3_widget = QWidget()
        vbox3_widget.setLayout(vbox3)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(vbox1_widget)
        hbox1.addWidget(vbox2_combined_widget)
        hbox1.addWidget(vbox3_widget)

        main_layout = QVBoxLayout()
        main_layout.addLayout(hbox1)

        self.setLayout(main_layout)

    def show_loss_and_acc_image_popup(self, image_path):
        dialog = QDialog(self)
        dialog.setWindowTitle("Image")

        vbox = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        vbox.addWidget(label)

        dialog.setLayout(vbox)
        dialog.exec_()
    
    def show_probabilities_popup(self, data_list, data_labels = None):
        
        if data_labels is None:
            data_labels = [str(i) for i in range(0, len(data_list))]

        fig, ax = plt.subplots()
        ax.bar(range(len(data_list)), data_list)
        ax.set_title('Probabilities')
        ax.set_xlabel('Probabilities')
        ax.set_ylabel('Number')
        ax.set_xticks(range(len(data_list)))
        ax.set_xticklabels(data_labels)

        canvas = FigureCanvas(fig)

        dialog = QDialog(self)
        dialog.setWindowTitle("Probabilities")

        vbox = QVBoxLayout()
        vbox.addWidget(canvas)

        dialog.setLayout(vbox)
        dialog.exec_()

    def show_class_image_popup(self):

        dialog = QDialog(self)
        dialog.setWindowTitle("Class Images")

        cat_image = cv2.imread('Q2_Dataset/training_dataset/Cat/0.jpg')
        dog_image = cv2.imread('Q2_Dataset/training_dataset/Dog/0.jpg')
        cat_image = cv2.resize(cat_image, (224, 224))
        dog_image = cv2.resize(dog_image, (224, 224))

        cat_image_qt = QImage(cat_image.data, cat_image.shape[1], cat_image.shape[0], cat_image.strides[0], QImage.Format_RGB888).rgbSwapped()
        dog_image_qt = QImage(dog_image.data, dog_image.shape[1], dog_image.shape[0], dog_image.strides[0], QImage.Format_RGB888).rgbSwapped()

        cat_pixmap = QPixmap.fromImage(cat_image_qt)
        dog_pixmap = QPixmap.fromImage(dog_image_qt)

        cat_label = QLabel()
        dog_label = QLabel()
        cat_label.setPixmap(cat_pixmap)
        dog_label.setPixmap(dog_pixmap)


        cat_text_label = QLabel("Cat")
        cat_text_label.setAlignment(Qt.AlignCenter)
        dog_text_label = QLabel("Dog")
        dog_text_label.setAlignment(Qt.AlignCenter)

        cat_layout = QVBoxLayout()
        dog_layout = QVBoxLayout()
        hbox = QHBoxLayout()

        cat_layout.addWidget(cat_text_label)
        cat_layout.addWidget(cat_label)
        dog_layout.addWidget(dog_text_label)
        dog_layout.addWidget(dog_label)
        hbox.addLayout(dog_layout)
        hbox.addLayout(cat_layout)

        dialog.setLayout(hbox)
        dialog.exec_()

    def show_comparison_popup(self, data_list):
        
        fig, ax = plt.subplots()
        ax.bar(range(len(data_list)), data_list)
        ax.set_title('Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(len(data_list)))
        ax.set_xticklabels([f"Normal\n({data_list[0]:.4}%)", f"Erasing\n({data_list[1]:.4}%)"])

        canvas = FigureCanvas(fig)

        dialog = QDialog(self)
        dialog.setWindowTitle("Comparison")

        vbox = QVBoxLayout()
        vbox.addWidget(canvas)

        dialog.setLayout(vbox)
        dialog.exec_()

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.image_label1.setPixmap(QPixmap(file_name))
            print(f"Loaded image: {file_name}")
        self.image1 = Image.open(file_name)

    def load_video(self):
        print("Load Video")

    def show_structure(self):
        print("Show Structure")
        model_structure = self.mc.show_model()
        print(model_structure)

    def show_acc_loss(self):
        print("Show Acc and Loss")
        image_path = f"Q1_saved_figures/figure_{self.mc_timestamp}.png"
        self.show_loss_and_acc_image_popup(image_path)

    def predict(self):
        print("Predict")
        ans, output = self.mc.predict_model(self.image1)
        print(f"Ans:{ans}")
        self.predict_label.setText(str(ans))
        probabilities =  self.mc.show_probabilities(output)
        self.show_probabilities_popup(probabilities)

    def load_image2(self):
        print("Q2 Load Image")
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.image_label2.setPixmap(QPixmap(file_name))
            print(f"Loaded image: {file_name}")
        self.image2 = Image.open(file_name)

    def show_image(self):
        print("Show Image")
        self.show_class_image_popup()

    def show_model(self):
        print("Show Model")
        model_structure = self.cc.show_model()
        print(model_structure)

    def show_comparison(self):
        print("Show Comparison")

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
        ])

        test_ds = my_catdog_dataset.My_CatDog_Dataset(train=False, transform=transforms)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)

        test_loss, test_accuracy = self.cc.test_model(test_loader, torch.nn.CrossEntropyLoss())
        test_loss_erasing, test_accuracy_erasing = self.cc_erasing.test_model(test_loader, torch.nn.CrossEntropyLoss())

        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        print(f"Test Loss Erasing: {test_loss_erasing}, Test Accuracy Erasing: {test_accuracy_erasing}")

        self.show_comparison_popup([test_accuracy, test_accuracy_erasing])

    def inference(self):
        print("Inference button")
        ans, output = self.cc_erasing.predict_model(self.image2)
        print(f"Ans:{ans}")
        print(output)
        self.inference_label.setText(str(ans))
        self.show_probabilities_popup(output, ['Dog', 'Cat'])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())