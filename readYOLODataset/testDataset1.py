import sys
import os
import cv2
import yaml
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

class YoloViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.load_button = QPushButton('Load Dataset', self)
        self.load_button.clicked.connect(self.load_dataset)

        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.next_image)

        self.prev_button = QPushButton('Previous', self)
        self.prev_button.clicked.connect(self.prev_image)

        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.toggle_play_pause)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.play_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setWindowTitle('YOLOv8 Dataset Viewer')

        self.image_folder = ''
        self.image_files = []
        self.current_index = -1
        self.class_names = []
        self.num_classes = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_event)
        self.play_speed = 100  # milliseconds per image (adjust as needed)
        self.yaml_file_name = ''
        self.is_playing = False

    def list_folders(self, directory):
        return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    def load_dataset(self):
        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(self, 'Select Dataset Folder', '', options=options)
        
        if folder:
            subfolders = self.list_folders(folder)
            if not subfolders:
                QMessageBox.critical(self, "Error", "No subfolders found in the selected directory!")
                return

            selected_folder, ok = QInputDialog.getItem(self, "Select Folder", "Folders:", subfolders, 0, False)
            if ok and selected_folder:
                self.image_folder = os.path.join(folder, selected_folder, 'images')
                self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
                self.current_index = 0
                self.yaml_file_name = ''

                for root, dirs, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".yaml"):
                            self.yaml_file_name = file
                            break

                # Check if yaml_file_name is empty
                if not self.yaml_file_name:
                    QMessageBox.critical(self, "Error", "YAML file not found in the dataset folder!")
                    return
                   
                # Load class names and number of classes from YAML file
                yaml_file = os.path.join(folder, self.yaml_file_name)
                if os.path.exists(yaml_file):
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            self.class_names = data['names']
                            self.num_classes = len(self.class_names)

                # Handle case where YAML file doesn't exist or is invalid
                if self.num_classes == 0:
                    QMessageBox.critical(self, "Error", "Unable to load class names from YAML file!")
                    sys.exit(1)

                if self.image_files:
                    self.display_image()
                else:
                    QMessageBox.critical(self, "Error", "No image files found in the selected folder!")

    def display_image(self):
        if 0 <= self.current_index < len(self.image_files):
            image_file = self.image_files[self.current_index]
            img_path = os.path.join(self.image_folder, image_file)
            
            # Read the image using OpenCV
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get the annotation file
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:  # Ensure the line has exactly 5 parts
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

                            # Display class name if valid class_id
                            if class_id < self.num_classes:
                                class_name = self.class_names[class_id]
                                cv2.putText(image, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert the image to a format suitable for Qt
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap(q_image)

            # Display the image in the QLabel
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    @pyqtSlot()
    def toggle_play_pause(self):
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText('Play')
        else:
            self.timer.start(self.play_speed)
            self.play_button.setText('Pause')
        self.is_playing = not self.is_playing

    @pyqtSlot()
    def timer_event(self):
        # Stop the timer if we reached the end
        if self.current_index >= len(self.image_files) - 1:
            self.timer.stop()
            self.play_button.setText('Play')
            self.is_playing = False
            return

        # Increment index and display next image
        self.current_index += 1
        self.display_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.prev_image()
        else:
            super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = YoloViewer()
    viewer.show()
    sys.exit(app.exec_())
