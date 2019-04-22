import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection
import os

workdir = os.getcwd()

find_objects = ObjectDetection()
find_objects.setModelTypeAsRetinaNet()

find_objects.setModelPath(
    os.path.join(
        workdir,
        "resnet50_coco_best_v2.0.1.h5"))
find_objects.loadModel()

objects_found = find_objects.detectObjectsFromImage(
    input_image=os.path.join(workdir, "ilia.jpg"),
    output_image_path=os.path.join(workdir, "iliaRevealed.jpg"))

for objects in objects_found:
    print(objects["name"] + " : " + str(objects["percentage_probability"]))
