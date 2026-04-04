# paddlex_vietocr
Bạn có thể chạy Jupyter Notebook trên Google Colab để thử nghiệm với model
## OCR Server
Trước khi chạy, bạn cần cài các dependencies (Cảnh báo: Sẽ tốn rất nhiều dung lượng bộ nhớ (~14GB) và thời gian cài rất lâu). Khuyến khích cài trên venv nhằm tránh conflict với các module có sẵn.
```
pip install vietocr
pip install "paddlex[ocr]"
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install fastapi uvicorn python-multipart
```
Sau khi cài xong, chạy server bằng python
```
cd paddlex_vietocr
python3 server.py
```
## Docker
Yêu cầu: 
- Cần native Docker Engine hoặc Docker Desktop của Windows, Mac
- Máy có hỗ trợ GPU

Cài image bằng lệnh dưới đây
```
docker build -t vietpaddle-server:v1 .
```

Chạy bằng lệnh dưới đây
```
docker run -d --name ocr_api_container --gpus all -p 8000:8000 vietpaddle-server:v1
```

Route /predict của server dùng để tải lên 1 file ảnh (.png, .jpg) và trả về 1 file .json chứa tập hợp các dòng nhận diện được.

**Ví dụ: sử dụng curl để tải ảnh lên**
```
curl -X POST "http://localhost:8000/predict" -F "file=@test_document.jpg"
```