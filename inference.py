import io
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

app = Flask(__name__)

# Путь к обученной модели YOLO
MODEL_PATH = "runs/detect/train/weights/best.pt"

print("Loading model...")
model = YOLO(MODEL_PATH)  # Загружаем модель один раз при старте сервера
print("Model loaded.")

# Hard-coded label map (учтите, что YOLO использует 0-based индексацию классов)
label_map = {
    0: "Head and shoulders bottom",
    1: "Head and shoulders top",
    2: "M_Head",
    3: "StockLine",
    4: "Triangle",
    5: "W_Bottom"
}

def run_inference(image_np):
    """
    Выполняет инференс на изображении (numpy array)
    и возвращает словарь с детекциями: bounding boxes, scores и классы.
    """
    # Запускаем предсказание (результат – список результатов, берем первый)
    results = model.predict(image_np, verbose=False)[0]
    # Извлекаем боксы (формат xyxy, абсолютные координаты), оценки и классы
    boxes = results.boxes.xyxy.cpu().numpy()      # [xmin, ymin, xmax, ymax]
    scores = results.boxes.conf.cpu().numpy()       # confidence score
    classes = results.boxes.cls.cpu().numpy().astype(int)
    return {
        'detection_boxes': boxes,
        'detection_scores': scores,
        'detection_classes': classes
    }


def visualize_detections(image, detections, score_threshold=0.45):
    """
    Рисует рамки и метки детекций на изображении.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for box, score, cls_id in zip(
            detections['detection_boxes'],
            detections['detection_scores'],
            detections['detection_classes']):
        if score < score_threshold:
            continue

        xmin, ymin, xmax, ymax = box.astype(int)
        label = label_map.get(cls_id, f"class_{cls_id}")
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        text = f"{label}: {score:.2f}"
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Рисуем заливку для текста
        draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill="red")
        draw.text((xmin, ymin - text_height), text, fill="white", font=font)
    return image



@app.route('/', methods=['GET'])
def index():
    """Главная страница с формой загрузки изображений."""
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    """Принимает изображения, выполняет инференс и возвращает результаты."""
    if 'images' not in request.files:
        return jsonify({"error": "Изображения не переданы"}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "Изображения не переданы"}), 400

    images_base64 = []
    for file in files:
        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            print(f"Ошибка при открытии изображения: {e}")
            continue

        image_np = np.array(image)
        detections = run_inference(image_np)
        vis_image = image.copy()
        vis_image = visualize_detections(vis_image, detections, score_threshold=0.45)

        buffer = io.BytesIO()
        vis_image.save(buffer, format="JPEG")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images_base64.append(img_str)

    if not images_base64:
        return jsonify({"error": "Не удалось обработать изображения"}), 500

    # Передаем список изображений в шаблон для отображения
    return render_template("result.html", images=images_base64)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
