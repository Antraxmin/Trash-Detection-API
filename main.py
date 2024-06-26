from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from label import label_map 

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

    label_counts = {}
    
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]

        label = int(box.cls[0]) 
        score = box.conf[0] 

        label_text = label_map.get(label, str(label))

        if label_text in label_counts:
            label_counts[label_text] += 1
        else:
            label_counts[label_text] = 1

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        text = f"{label_text}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = int(x1)
        text_y = int(y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10)
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return JSONResponse(content={
        "count": count,
        "label_counts": label_counts,
        "image": img_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
