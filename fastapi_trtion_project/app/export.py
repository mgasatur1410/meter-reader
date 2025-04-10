from ultralytics import YOLO

# Загружаем модель
model = YOLO("C:\\Users\\USER\\Downloads\\best.pt")  # загрузка официальной модели

# Список для хранения метаданных во время экспорта.
metadata = []

def export_cb(exporter):
    metadata.append(exporter.metadata)

model.add_callback("on_export_end", export_cb)

# Экспорт модели с указанием opset версии 12
onnx_file = model.export(format="onnx", dynamic=True, opset=12)
