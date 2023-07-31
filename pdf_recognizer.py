import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QFont
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
import re
import ocrmypdf

# Load the pre-trained model
model = tf.keras.models.load_model("model.h5")
pattern = r"\b8\d{8}\b"


class PDFProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PDF File Processor")
        self.setGeometry(100, 100, 400, 300)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout(main_widget)

        self.instruction_label = QLabel("Please select a PDF file:", self)
        # self.instruction_label.setFont(QFont('Arial', 12))
        layout.addWidget(self.instruction_label)

        self.selected_file_label = QLabel("", self)
        # self.selected_file_label.setFont(QFont('Arial', 12, 'bold'))
        layout.addWidget(self.selected_file_label)

        self.selected_file_value = QLabel("", self)
        # self.selected_file_value.setFont(QFont('Arial', 12))
        layout.addWidget(self.selected_file_value)

        file_layout = QHBoxLayout()
        layout.addLayout(file_layout)

        self.browse_button = QPushButton("Browse", self)
        # self.browse_button.setFont('Arial', 12)
        self.browse_button.clicked.connect(self.choose_file)
        file_layout.addWidget(self.browse_button)

        self.ocr_button = QPushButton("OCR pdf", self)
        # self.ocr_button.setFont('Arial', 12)
        self.ocr_button.clicked.connect(self.ocr_pdf)
        file_layout.addWidget(self.ocr_button)

        self.process_button = QPushButton("Process", self)
        # self.process_button.setFont('Arial', 12)
        self.process_button.clicked.connect(self.process_file)
        layout.addWidget(self.process_button)

        self.modify_button = QPushButton("Modify PDF", self)
        # self.modify_button.setFont('Arial', 12)
        self.modify_button.clicked.connect(self.modify_pdf)
        layout.addWidget(self.modify_button)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.pdf_path = file_path
            self.selected_file_label.setText("Selected PDF file:")
            self.selected_file_value.setText(self.pdf_path)

    def ocr_pdf(self):
        output_file_path = os.path.splitext(self.pdf_path)[0] + '_modified.pdf'
        ocrmypdf.ocr(self.pdf_path, output_file_path, deskew=True,
                     clean=True, rotate_pages=True, skip_text=True)

    def preprocess_pdf_multipage(self, pdf_path):
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=20)
        resized_images = []
        for image in images:
            # Resize image to image_height x image_width
            resized_image = image.resize((224, 224))
            # Append the resized image to the list
            resized_images.append(resized_image)
        return resized_images

    def predict_pdf_multipage(self, pdf_path, model):
        # Preprocess the PDF file
        preprocessed_image = self.preprocess_pdf_multipage(pdf_path)
        image_array = np.array(preprocessed_image)

        page_result = []
        page = []
        for i, img in enumerate(image_array):
            prediction = model.predict(np.expand_dims(img, axis=0))
            page_result.append(np.argmax(prediction))
            page.append(i+1)

        self.gas = pd.DataFrame({'page': page, 'result': page_result})
        self.gas.to_csv('hasil_extract.csv')

    def add_bookmark_to_pdf(self, pdf_path, gas_dataframe):
        output_file_path = os.path.splitext(pdf_path)[0] + '_modified.pdf'

        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            output_pdf = PdfWriter()

            for index, row in gas_dataframe.iterrows():
                page_number = int(row['page'])
                result = row['result']
                page = pdf.pages[page_number - 1]
                output_pdf.add_page(page)
                if result == 0:
                    text = page.extract_text()
                    order_no = re.findall(pattern, text)
                    bookmark_title = f"JC Page {order_no}"
                    output_pdf.add_outline_item(
                        bookmark_title, page_number - 1)
                elif result == 1:
                    page_number_str = str(page_number)
                    bookmark_title = f"MDR Page {page_number_str}"
                    output_pdf.add_outline_item(
                        bookmark_title, page_number - 1)

            with open(output_file_path, 'wb') as output_file:
                output_pdf.write(output_file)

        print(f"Modified PDF file saved as: {output_file_path}")

    def process_file(self):
        if hasattr(self, 'pdf_path'):
            print("Processing PDF file:", self.pdf_path)
            self.predict_pdf_multipage(self.pdf_path, model)

    def modify_pdf(self):
        if hasattr(self, 'pdf_path'):
            self.add_bookmark_to_pdf(self.pdf_path, self.gas)
            print("PDF file modified.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PDFProcessorApp()
    window.show()
    sys.exit(app.exec_())
