from ultralytics import YOLO

# Load a model
model = YOLO('weights/yolov8m.pt')  # load a pretrained model (recommended for training)

model.train(data="data_staff_removed_20_crop.yaml", epochs=500, batch=4, imgsz=640)  # image size chocie : 640, 1280, and 1920.
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format 