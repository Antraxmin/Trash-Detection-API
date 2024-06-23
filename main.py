from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = FastAPI()

MODEL_PATH = 'models/best.pt'
model = YOLO(MODEL_PATH)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8) 
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model.predict(source=img)

    detections = results[0].boxes
    count = len(detections)
    
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return JSONResponse(content={
        "count": count,
        "image": img_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
