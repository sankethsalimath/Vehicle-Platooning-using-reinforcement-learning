import cv2
import numpy as np
import time

class tiny_yolo_detect:
    def __init__(self):
        # Load Yolo
        self.net = cv2.dnn.readNet("object_detection/yolov3-tiny.weights","object_detection/yolov3-tiny.cfg")
        self.classes = []
        with open("object_detection/coco2.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_img(self, file, SHOW_FPS=False):
        # Loading image
        cap = cv2.VideoCapture(file)
        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0
        while True:
            _, frame = cap.read()
            frame = cv2.resize(frame,(1280,720))
            frame_id += 1

            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    color = self.colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

            if SHOW_FPS == True:
                show_fps()

            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        return boxes

    def show_fps(self):
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)


    def destroy(self):
        cv2.destroyAllWindows()

