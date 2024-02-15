from ultralytics import YOLO

# Load a model
model = YOLO('weights/yolov8m.pt')  # load a pretrained model (recommended for training)

model.train(data="data_staff_removed.yaml", epochs=1920, batch=1, imgsz = 1280)  # image size chocie : 640, 1280, and 1920.
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format 