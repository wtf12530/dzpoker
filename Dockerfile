# Dockerfile for running the suggestion server

FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
# install onnxruntime for fast inference
RUN pip install --no-cache-dir onnxruntime
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
