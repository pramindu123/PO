# Purchase Order Image Reader

A Python-based tool that extracts data from purchase order images using OCR (Optical Character Recognition) and exports the results to CSV files.

## Overview

This project uses OpenCV for image preprocessing and Tesseract OCR to read and parse purchase order documents from images. It automatically extracts key information such as:

- PO Number
- Date
- Vendor Information
- Total Amount
- Line Items (quantity, description, price)

## Features

- **Image Preprocessing**: Enhances image quality for better OCR accuracy
  - Grayscale conversion
  - Noise reduction
  - Adaptive thresholding
  - Automatic deskewing
  - CLAHE contrast enhancement
- **OCR Text Extraction**: Uses Tesseract OCR engine with optimized settings
- **Data Parsing**: Regex-based extraction of common purchase order fields
- **Batch Processing**: Process multiple images from a folder
- **CSV Export**: Export results in both summary and detailed formats

## Project Structure

```
PO/
├── purchase_order_reader.ipynb  # Main Jupyter notebook
├── images/                      # Input folder for PO images
├── output/                      # Output folder for CSV files
│   ├── purchase_orders_output.csv
│   └── purchase_orders_detailed.csv
├── tessaret/                    # Tesseract OCR files
│   ├── tesseract.exe
│   ├── tessdata/               # Tesseract language data
│   │   ├── eng.traineddata
│   │   └── osd.traineddata
│   └── doc/
└── README.md
```

## Requirements

### Python Packages

- opencv-python
- pytesseract
- pandas
- numpy
- pillow

Install using pip:

```bash
pip install opencv-python pytesseract pandas numpy pillow
```

### Tesseract OCR

This project includes a local Tesseract installation in the `tessaret/` folder. The notebook is configured to use this local installation.

## Usage

1. **Place your purchase order images** in the `images/` folder
   - Supported formats: PNG, JPG, JPEG, TIFF, BMP

2. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook purchase_order_reader.ipynb
   ```

3. **Run all cells** in the notebook to:
   - Import required libraries
   - Define preprocessing and OCR functions
   - Process images from the `images/` folder
   - Export results to CSV files in the `output/` folder

## Output

The notebook generates two types of CSV output:

- **Summary CSV**: Contains main PO information (PO number, date, vendor, total amount)
- **Detailed CSV**: Includes line item details for each purchase order

## Configuration

### Tesseract Path

The Tesseract executable path is configured in the notebook:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\PO\tessaret\tesseract.exe'
```

Modify this path if your Tesseract installation is in a different location.

### OCR Settings

The OCR uses the following Tesseract configuration:

- `--oem 3`: Use default OCR Engine Mode (LSTM + Legacy)
- `--psm 6`: Assume a uniform block of text

## Tips for Best Results

1. **Image Quality**: Use high-resolution scans (300 DPI or higher)
2. **Lighting**: Ensure even lighting when photographing documents
3. **Alignment**: Keep documents as straight as possible
4. **Contrast**: High contrast between text and background improves accuracy

## License

See the Tesseract [LICENSE](tessaret/doc/LICENSE) file for Tesseract OCR licensing information.
