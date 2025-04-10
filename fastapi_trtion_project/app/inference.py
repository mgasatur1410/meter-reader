import cv2
import numpy as np
import torch
import tritonclient.http as httpclient
from ultralytics import YOLO


# ------------------------------------------------
# ФУНКЦИИ ДЛЯ OBB
# ------------------------------------------------
def get_rotated_bbox_corners(center_x, center_y, width, height, rotate):
    cos_angle = np.cos(rotate)
    sin_angle = np.sin(rotate)
    half_width = width / 2
    half_height = height / 2
    corners = np.array([
        [-half_width, -half_height],
        [ half_width, -half_height],
        [ half_width,  half_height],
        [-half_width,  half_height]
    ], dtype=float)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle,  cos_angle]
    ])
    rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([center_x, center_y])
    return rotated_corners

def get_crop_coords(rotated_corners):
    min_x = int(np.min(rotated_corners[:, 0]))
    min_y = int(np.min(rotated_corners[:, 1]))
    max_x = int(np.max(rotated_corners[:, 0]))
    max_y = int(np.max(rotated_corners[:, 1]))
    return min_x, min_y, max_x, max_y

def rotate_image(image, rotation):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.rad2deg(rotation), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def crop_rotate(image, center_x, center_y, width, height, rotation, dwidth=0):
    rotated_image = rotate_image(image, rotation)
    rotated_corners = get_rotated_bbox_corners(center_x, center_y, width + dwidth, height, rotation)
    min_x, min_y, max_x, max_y = get_crop_coords(rotated_corners)
    cropped_image = rotated_image[min_y:max_y, min_x:max_x]
    return cropped_image

def crop_rotate_obb(obb_object):
    """
    Предполагается, что obb_object.obb.xywhr[0] = [x_center, y_center, w, h, theta].
    Расширяем ширину на 5%.
    """
    if obb_object.obb is None:
        raise ValueError("Данные OBB отсутствуют.")
    obb = obb_object.obb.xywhr[0].cpu().numpy()  # [x, y, w, h, theta]
    center_x, center_y, width, height, theta = obb
    cropped = crop_rotate(obb_object.orig_img, center_x, center_y, width, height, theta, dwidth=width * 0.01)
    return cropped


# ------------------------------------------------
# КЛАСС ДЛЯ РАСПОЗНАВАНИЯ (Triton)
# ------------------------------------------------
class TritonRecognitionClient:
    def __init__(self, server_url: str, model_name: str, input_name: str, output_name: str):
        self.server_url = server_url
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.client = httpclient.InferenceServerClient(url=self.server_url)
        # CTC: 0='blank', 1..10='0'..'9', 11='.'
        self.character = ['blank'] + [str(i) for i in range(10)] + ['.']

    def decode_ctc(self, preds_index, preds_prob, remove_duplicates=True):
        results = []
        for b in range(len(preds_index)):
            seq = []
            conf = []
            prev = None
            for i, idx in enumerate(preds_index[b]):
                if idx == 0:
                    continue  # blank
                if remove_duplicates and prev == idx:
                    continue
                if idx < len(self.character):
                    seq.append(self.character[idx])
                    conf.append(preds_prob[b][i])
                prev = idx
            text = ''.join(seq)
            avg_conf = float(np.mean(conf)) if conf else 0.0
            results.append((text, avg_conf))
        return results

    def run_inference(self, input_tensor: np.ndarray):
        inputs = [httpclient.InferInput(self.input_name, input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)
        outputs = [httpclient.InferRequestedOutput(self.output_name)]
        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        output_data = response.as_numpy(self.output_name)  # [B, T, num_classes]
        if output_data.ndim == 2:
            output_data = np.expand_dims(output_data, axis=0)
        preds_idx = output_data.argmax(axis=2)
        preds_prob = output_data.max(axis=2)
        return self.decode_ctc(preds_idx, preds_prob, remove_duplicates=True)


# ------------------------------------------------
# ГЛАВНАЯ ЛОГИКА
# ------------------------------------------------
if __name__ == "__main__":
    # 1) Модель детекции (локальный ONNX)
    model_path = "C:\\Users\\USER\\Desktop\\fastapi_trtion_project\\model_repository\\model_detect\\1\\model.onnx"
    yolo_model = YOLO(model_path, task="detect")

    # 2) Модель распознавания (Triton)
    triton_server_url = "localhost:8000"
    rec_model_name = "my_model"
    rec_input_name = "x"
    rec_output_name = "softmax_0.tmp_0"
    recog_client = TritonRecognitionClient(
        server_url=triton_server_url,
        model_name=rec_model_name,
        input_name=rec_input_name,
        output_name=rec_output_name
    )

    # 3) Изображение
    img_path = "C:\\Users\\USER\\Downloads\\id_216_value_493_122_jpg.rf.e4430bd75a8974d2849787bf15cf391a.jpg"
    preds = yolo_model.predict(
        source=img_path,
        conf=0.75,
        max_det=1,
        save=False
    )
    if len(preds) == 0:
        print("Детекция не нашла объекты.")
        exit(0)

    detect_result = preds[0]

    # 4) Если есть OBB, пробуем. Иначе axis-aligned bbox.
    if detect_result.obb is not None:
        try:
            cropped_img = crop_rotate_obb(detect_result)
            print("Обрезка по OBB (5% расширение).")
        except Exception as e:
            print("Ошибка при OBB:", e)
            if detect_result.boxes.xyxy.shape[0] > 0:
                # axis-aligned fallback
                bbox = detect_result.boxes.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                h_box = y2 - y1
                w_box = x2 - x1
                pad_x = int(w_box * 0.01)
                pad_y = int(h_box * 0.01)
                H, W = detect_result.orig_img.shape[:2]
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(W, x2 + pad_x)
                y2 = min(H, y2 + pad_y)
                # Проверяем, вдруг бокс вертикальный (w_box < h_box)?
                if (x2 - x1) < (y2 - y1):
                    # Меняем местами x и y
                    print("Вынужденный swap XY для вертикального бокса.")
                    new_x1 = y1
                    new_y1 = x1
                    new_x2 = y2
                    new_y2 = x2
                    new_x1, new_x2 = sorted([new_x1, new_x2])
                    new_y1, new_y2 = sorted([new_y1, new_y2])
                    cropped_img = detect_result.orig_img[new_y1:new_y2, new_x1:new_x2]
                else:
                    cropped_img = detect_result.orig_img[y1:y2, x1:x2]
            else:
                print("Нет данных для axis-aligned bbox.")
                exit(0)
    else:
        print("OBB отсутствует. Используем axis-aligned bbox (5% расширение).")
        if detect_result.boxes.xyxy.shape[0] > 0:
            bbox = detect_result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            h_box = y2 - y1
            w_box = x2 - x1
            pad_x = int(w_box * 0.01)
            pad_y = int(h_box * 0.01)
            H, W = detect_result.orig_img.shape[:2]
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(W, x2 + pad_x)
            y2 = min(H, y2 + pad_y)
            # Проверяем вертикальность
            if (x2 - x1) < (y2 - y1):
                print("Swap XY для вертикального bbox (axis-aligned).")
                new_x1 = y1
                new_y1 = x1
                new_x2 = y2
                new_y2 = x2
                new_x1, new_x2 = sorted([new_x1, new_x2])
                new_y1, new_y2 = sorted([new_y1, new_y2])
                cropped_img = detect_result.orig_img[new_y1:new_y2, new_x1:new_x2]
            else:
                cropped_img = detect_result.orig_img[y1:y2, x1:x2]
        else:
            print("Нет данных bbox.")
            exit(0)

    if cropped_img.size == 0:
        print("Обрезанная область пуста.")
        exit(0)

    # 5) Подготовка для распознавания
    layer = torch.nn.AdaptiveAvgPool2d((32, 400))
    t_img = torch.tensor(cropped_img.transpose((2, 0, 1))).unsqueeze(0).float() / 255.
    t_resized = layer(t_img)  # [1, 3, 32, 400]
    input_for_rec = t_resized.detach().cpu().numpy().astype(np.float32)

    # 6) Запуск распознавания
    result_text = recog_client.run_inference(input_for_rec)
    print("Результаты распознавания:", result_text)

    # 7) (Опционально) Показать обрезку
    cv2.imshow("Обрезанная область", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()