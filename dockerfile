# 1. Use the official NVIDIA CUDA 12.6 base image
FROM nvidia/cuda:12.6.1-base-ubuntu22.04

# 2. Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Install Python 3.10 and system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Install PyTorch and PaddlePaddle (CUDA 12.6 versions)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 6. Install the remaining Python packages
RUN pip install --no-cache-dir vietocr "paddlex[ocr]" fastapi uvicorn python-multipart

# 7. Copy your project files into the container
COPY uvdoc/ /app/uvdoc/
COPY ocrmodel.py /app/
COPY server.py /app/

# 8. Expose port 8000 for the host machine
EXPOSE 8000

# 9. Run the FastAPI server
CMD ["python", "server.py"]