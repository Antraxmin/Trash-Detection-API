FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]