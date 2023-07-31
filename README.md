# PDF Page Recognition Application using CNN Model

This project is a PDF page recognition application that uses a Convolutional Neural Network (CNN) model with VGG16 as the base model and an additional Conv2D layer for the recognition layer. The application is built with Python and utilizes PyQt5 for the graphical user interface (GUI).

## Installation

1. Clone the GitHub repository to your local machine:

```bash
git clone https://github.com/yourusername/pdf-page-recognition.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python pdf_recognizer.py
```


2. The PyQt5 GUI window will open. Click on the "Browse" button to select a PDF file.

3. Click on the "OCR PDF" button to perform OCR (Optical Character Recognition) on the selected PDF file. This step is optional but recommended for improved recognition accuracy.

4. Click on the "Process" button to recognize the pages in the PDF using the trained CNN model.

5. Click on the "Modify PDF" button to create bookmarks/outlines in the PDF file based on the recognized pages.

## Model Architecture

The CNN model used for page recognition consists of VGG16 as the base model followed by an additional Conv2D layer. The base VGG16 model is pre-trained on the ImageNet dataset and is used for feature extraction. The additional Conv2D layer is added for recognition purposes.




