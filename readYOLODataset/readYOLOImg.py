import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class YoloViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)

        self.setLayout(layout)
        self.setWindowTitle('YOLOv8 Dataset Viewer')

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.jpeg)', options=options)
        
        if file_name:
            self.display_image(file_name)

    def display_image(self, image_path):
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the annotation file
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert YOLO format to bounding box coordinates
                    img_height, img_width, _ = image.shape
                    x_min = int((x_center - width / 2) * img_width)
                    x_max = int((x_center + width / 2) * img_width)
                    y_min = int((y_center - height / 2) * img_height)
                    y_max = int((y_center + height / 2) * img_height)

                    # Draw the bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Convert the image to a format suitable for Qt
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)

        # Display the image in the QLabel
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = YoloViewer()
    viewer.show()
    sys.exit(app.exec_())
