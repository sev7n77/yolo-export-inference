from ultralytics import YOLO
model = YOLO ("best.pt")

print ("正在导出onnx模型")
exported_path = model.export (format='onnx')
print(f"导出成功，文件在: {exported_path}")

print("正在导出engine")
engine_path = model.export (format='engine', half=True)
print(f"导出成功，文件在: {engine_path}")