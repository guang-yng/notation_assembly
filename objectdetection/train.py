from ultralytics import YOLO
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, help="The yaml config file for dataset.")
    parser.add_argument('-e', '--epochs', default=500, type=int, help="Number of training epochs.")
    parser.add_argument('-b', '--batch_size', default=8, type=int, help="Batch size.")
    parser.add_argument('-m', '--model', default='weights/yolov8l.pt', type=str, help="The model weights to use.")
    parser.add_argument('-i', '--imgsz', default=640, type=int, help="The image size of YOLO. Choose from [640, 1280, 1920].")
    parser.add_argument('--output_dir', default=None, help='The directory to save training info & models.')

    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    model.train(data=args.config, epochs=args.epochs, batch=args.batch_size, imgsz=640, project=args.output_dir)  # image size chocie : 640, 1280, and 1920.
    metrics = model.val()  # evaluate model performance on the validation set
    path = model.export(format="onnx")  # export the model to ONNX format 