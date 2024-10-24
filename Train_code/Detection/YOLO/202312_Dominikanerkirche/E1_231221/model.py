from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="B:\\01_Study\\Uni-Bamberg\\HIWI\\Holzprojects\\Datasets\\Dominikanerkirche\\export__with_label\\2.Dominikanerkirche.75TS.20VS.5TTS.yolov8\\data.yaml", epochs=10, batch=8)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format
