import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from app.inference import TritonRecognitionClient, crop_rotate_obb

app = FastAPI(title="FastAPI wrapper для Triton", version="1.0")

# Определяем базовую директорию проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(BASE_DIR, "frontend")

# Монтируем статику из папки frontend
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = os.path.join(frontend_dir, "index.html")
    try:
        with open(index_file, "r", encoding="utf8") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки index.html: {e}")


def axis_aligned_bbox_crop(orig_img, bbox: np.ndarray, pad_ratio=0.01, swap_xy=False) -> np.ndarray:
    if bbox.size < 4:
        raise ValueError(f"Некорректный bbox: ожидалось >=4 координат, получено {bbox.size}. bbox={bbox}")
    x1, y1, x2, y2 = bbox[:4].astype(int)
    if swap_xy:
        new_x1 = y1
        new_y1 = x1
        new_x2 = y2
        new_y2 = x2
        x1, x2 = sorted([new_x1, new_x2])
        y1, y2 = sorted([new_y1, new_y2])
    w_box = x2 - x1
    h_box = y2 - y1
    if w_box <= 0 or h_box <= 0:
        raise ValueError(f"Некорректный bbox: x2<=x1 или y2<=y1. (x1,x2,y1,y2)=({x1},{x2},{y1},{y2})")
    pad_x = int(w_box * pad_ratio)
    pad_y = int(h_box * pad_ratio)
    H, W = orig_img.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("После расширения координаты некорректны. Бокс пуст.")
    return orig_img[y1:y2, x1:x2]


# Параметры моделей
TRITON_SERVER_URL = "localhost:8000"
RECOGNITION_MODEL = "my_model"
RECOGNITION_INPUT = "x"
RECOGNITION_OUTPUT = "softmax_0.tmp_0"
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "model_repository", "model_detect", "1", "model.onnx")

yolo_model = YOLO(DETECTION_MODEL_PATH, task="detect")

recognition_client = TritonRecognitionClient(
    server_url=TRITON_SERVER_URL,
    model_name=RECOGNITION_MODEL,
    input_name=RECOGNITION_INPUT,
    output_name=RECOGNITION_OUTPUT
)


@app.post("/infer_image")
async def infer_image_endpoint(file: UploadFile = File(...)):
    """
    Принимает изображение, выполняет детекцию (YOLO),
    затем обрезает область (OBB или axis-aligned) + небольшой паддинг (1%),
    приводит результат к (32, 400), отправляет в Triton,
    а итоговые тексты (и уверенности) возвращает фронтенду.
    Перед возвратом убираем ведущие нули в тексте.
    """
    try:
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось декодировать изображение.")

        preds = yolo_model.predict(source=img, conf=0.75, max_det=1, save=False)
        if len(preds) == 0:
            raise ValueError("Детекция не обнаружила объектов.")
        detect_result = preds[0]

        if detect_result.obb is not None:
            # Пытаемся обрезать OBB
            try:
                cropped_img = crop_rotate_obb(detect_result)
                print("Обрезка по OBB выполнена.")
            except Exception as e:
                print("Ошибка при OBB:", e)
                # fallback на axis-aligned
                if detect_result.boxes.xyxy.shape[0] > 0:
                    bbox = detect_result.boxes.xyxy[0].cpu().numpy()
                    if (bbox[2] - bbox[0]) < (bbox[3] - bbox[1]):
                        cropped_img = axis_aligned_bbox_crop(detect_result.orig_img, bbox, pad_ratio=0.01, swap_xy=True)
                        print("Вертикальный axis-aligned bbox обработан с swap_xy.")
                    else:
                        cropped_img = axis_aligned_bbox_crop(detect_result.orig_img, bbox, pad_ratio=0.01)
                else:
                    raise ValueError("Нет данных для axis-aligned bbox.")
        else:
            # Если нет OBB, берём axis-aligned bbox
            if detect_result.boxes.xyxy.shape[0] > 0:
                bbox = detect_result.boxes.xyxy[0].cpu().numpy()
                if (bbox[2] - bbox[0]) < (bbox[3] - bbox[1]):
                    cropped_img = axis_aligned_bbox_crop(detect_result.orig_img, bbox, pad_ratio=0.01, swap_xy=True)
                    print("Вертикальный axis-aligned bbox обработан с swap_xy.")
                else:
                    cropped_img = axis_aligned_bbox_crop(detect_result.orig_img, bbox, pad_ratio=0.01)
            else:
                raise ValueError("Нет данных для axis-aligned bbox.")

        if cropped_img.size == 0:
            raise ValueError("Обрезанная область пуста.")

        # Приводим к (32, 400)
        layer = torch.nn.AdaptiveAvgPool2d((32, 400))
        t_img = torch.tensor(cropped_img.transpose((2, 0, 1))).unsqueeze(0).float() / 255.
        t_resized = layer(t_img)
        input_for_rec = t_resized.detach().cpu().numpy().astype(np.float32)

        rec_result = recognition_client.run_inference(input_for_rec)

        # rec_result — массив [(text, conf), (text2, conf2), ...]
        final_result = []
        for text, conf in rec_result:
            # Удаляем ведущие нули
            cleaned_text = text.lstrip('0')
            if cleaned_text == "":
                cleaned_text = "0"
            # Возвращаем в том же виде [cleaned_text, conf], чтобы JS мог отобразить item[0]
            final_result.append([cleaned_text, conf])

        # Если final_result пуст, сообщаем "Нет распознанных данных"
        if not final_result:
            final_result = [["Нет распознанных данных", 0.0]]

        return {"result": final_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
