# Trash-Detection-API

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.2-blue.svg)](https://github.com/ultralytics/yolov8)

Trash Detection API is an open-source project leveraging YOLO (You Only Look Once) for detecting trash in images. This project uses FastAPI for the backend server and serves a simple web interface for uploading images and viewing detection results.

## Features

- **Real-time Object Detection**: Detects trash in images using a custom-trained YOLO model optimized for trash detection.
- **Fast and Efficient**: Uses FastAPI for handling requests at lightning speed.
- **User-Friendly Interface**: Simple web interface for uploading images and viewing detection results.
- **Base64 Image Encoding**: Returns images with detected trash in Base64 format for easy integration.

## Installation

To get started with Trash Detection, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Antraxmin/Trash-Detection-API.git
   cd trash-detection-api
   ```

2. **Create a Virtual Environment and Install Dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download the YOLO Model**

   Place your YOLO model (`best.pt`) in the `models` directory.

4. **Run the Server**

   ```bash
   uvicorn main:app --reload
   ```

5. **Access the Web Interface**

   Open your browser and go to `http://127.0.0.1:8000`.

### Example Request

```
curl -X POST "http://localhost:8000/predict/" -F "file=@image.jpg"
```

## Usage

- **Upload Image**: Use the web interface to upload an image.
- **View Results**: The server will return the image with detected trash highlighted in bounding boxes, along with the count of detected objects.

## Contact

For any inquiries or questions, please contact [antraxmin@gmail.com](mailto:antraxmin@gmail.com).
