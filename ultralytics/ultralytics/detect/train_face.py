from ultralytics import YOLO


def train(epoch):
    model = YOLO("yolov8s-face.yaml")
    model.train(data="face.yaml", epochs=epoch)
    model.val()
    print("face train and val finished !")
