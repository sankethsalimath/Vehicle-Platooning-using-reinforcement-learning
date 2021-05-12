import glob
import os
import sys
import numpy as np
import random
import time
import cv2
import math

import weakref

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' %(
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc

SECONDS_PER_EPISODE = 10

spawn_loc = [[-122, 236, 0],
                [-260, 138, -180]]

ACTIONS = {
    0: [1, 0, 0],
    1: [0, 0, -1],
    2: [0, 0, 1],
    3: [1, 0, -1],
    4: [1, 0, 1],
    5: [0, 1, 0]
}

IM_WIDTH = 416
IM_HEIGHT = 416
"""
# Cv2 DNN loading
net = cv2.dnn.readNet("C:/Codes/thesis/sac/yolo/yolov4-tiny.cfg", "C:/Codes/thesis/sac/yolo/yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


classes = []
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

"""
class CarEnv:
    """
    Class for Carla Environment with 2 vehicles
    """
    SHOW_CAM = False
    STEER_AMT = 1.0

    front_camera = None
    def __init__(self):
        # load client and world
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.world = self.client.load_world('Town06')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.mustang = self.blueprint_library.filter("mustang")[0]
        self.face_cascade = cv2.CascadeClassifier('cars.xml')
        self.collision_sensor = None

        # detection paramters
        self.veh_x_pos = []
        self.veh_y_pos = []
        self.distance = []
        self.x_dev, self.y_dev, self.h, self.w = 0, 0, 0, 0
        self.detecting = False


    def reset(self, autoPilot=False):
        print("****RESET****")
        self.collision_hist = []
        self.actor_list = []
        self.y_val = 10
        self.x_dev, self.y_dev = 0, 0
        loc = random.sample(spawn_loc, 1)
        x, y, yaw = loc[0][0], loc[0][1], loc[0][2]
        x2, y2, yaw2 = x, y, yaw
        #print(x, y, yaw, yaw2)
        self.tran1 = carla.Transform(carla.Location(x=x, y=y, z=1.8431), carla.Rotation(0, yaw, 0))


        if yaw2 >= 45 and yaw2 < 135:
            y2 += 10
        elif yaw2 == 0:
            x2 += 10
        elif yaw2 <= -45 and yaw2 > -135:
            y2 -= 10
        elif yaw2 <= -135:
            x2 -= 10


        print(self.tran1)
        self.tran2 = carla.Transform(carla.Location(x=x2, y=y2, z=1.8431), carla.Rotation(0, yaw2, 0))

        print(self.tran2)
        self.vehicle = self.world.spawn_actor(self.model_3, self.tran1)
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(autoPilot)
        time.sleep(0.1)


        self.vehicle2 = self.world.spawn_actor(self.model_3, self.tran2)
        self.actor_list.append(self.vehicle2)

        time.sleep(0.5)
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.actor_list.append(self.collision_sensor.sensor)

        time.sleep(0.5)


        ##### CameraSensor Class #########
        """
        self.rgb_cam = CameraSensor(self.vehicle, self.blueprint_library, self.world)
        self.actor_list.append(self.rgb_cam.sensor)
        time.sleep(2)
        """
        self.vehicle2.set_autopilot(True)
        #while self.front_camera is None:
        #    time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


        self.state = np.array([np.random.uniform(low=0, high=20), np.random.uniform(low=0, high=20), np.random.uniform(low=0, high=10), np.random.uniform(low=0, high=10)])

        return np.array(self.state)


    def step(self, action):
        #print("****ACTION****")
        #print(action)
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(ACTIONS[action][0]), brake=ACTIONS[action][1], steer=float(ACTIONS[action][2])))

        x_diff = self.vehicle.get_location().x - self.vehicle2.get_location().x
        y_diff = self.vehicle.get_location().y - self.vehicle2.get_location().y
        yaw_diff = self.vehicle.get_transform().rotation.yaw - self.vehicle2.get_transform().rotation.yaw
        dist = math.sqrt((x_diff**2)+(y_diff**2))

        space = (x_diff, y_diff, yaw_diff, 10-dist)
        #space = (self.rgb_cam.y_dev, self.rgb_cam.x_dev)
        done = False
        #if self.detecting:
        reward = (10 - dist - abs(0.1*yaw_diff))/100

        #space, reward, done = self.rgb_cam.reward_function()
        if dist > 20:
            print("dist out of bounds :: value=30")
            done = True
            reward = -4

        elif reward < -2.5:
            print("reward low :: value=-1.5")
            done = True
        if len(self.collision_sensor.history) != 0:
            print(':: Collision Detected! ::')
            done = True
            reward = -3


        if yaw_diff > 45:
            print("yaw diff :: 60")
            done = True


        time.sleep(1/50)
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        #print(space, " reward:", reward)
        #print(space, action, reward)
        return np.array(space), reward, done, None

    def collision_data(self, event):
        self.collision_hist.append(event)




class CollisionSensor(object):
    """
    Collision sensor objects with parameters actor, blueprint and world.
    """
    def __init__(self, parent_actor, bp, world):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        #world = self._parent.get_world()
        bp = bp.find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        print('col sensor initialised')

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        self.history.append(event)
        #print('append col')
        if len(self.history) > 4000:
            self.history.pop(0)

class CameraSensor(object):
    def __init__(self, parent_actor, bp, world):

        # rgb cam parameters
        self._parent = parent_actor
        self.sensor = None
        self.rgb_cam = bp.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"110")
        self.front_camera = None
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = world.spawn_actor(self.rgb_cam, transform, attach_to=self._parent)
        print('RGB CAM LOADED')

        # weak self to avoid circular referencing
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: weak_self().load_img(weak_self, image))
        self.x_dev, self.y_dev, self.h, self.w = 0, 0, 0, 0


    @staticmethod
    def load_img(weak_self, image):
        self = weak_self()
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        #i = np.array(i)
        array = np.reshape(array,(IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
        frame = array[:, :, ::-1]
        frame = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_PLAIN
        frame = self.detect(frame)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
        self.frame = frame

    def detect(self, frame):
        height, width = IM_WIDTH, IM_HEIGHT
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (IM_WIDTH, IM_HEIGHT), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and confidence < 1.0 and class_id == 4:
                    self.detecting = True
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

        #print(class_ids, boxes)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                self.x, self.y, self.w, self.h = boxes[i]
                #print(boxes[i])
                label = str(classes[class_ids[i]])

                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, 2)
                self.x_dev, self.y_dev = x + w/2 - 207, h-38
                #cv2.putText(frame, label + " " + str(round(confidence, 2)), (self.x, self.y + 30), font, 3, color, 3)
                frame = cv2.rectangle(frame,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,0,255),2)
                frame = cv2.line(frame,(int(IM_WIDTH/4),210),(int(3*IM_WIDTH/4),210),(0,255,0),1)
                frame = cv2.line(frame,(204,int(IM_HEIGHT/4)),(204,int(3*IM_HEIGHT/4)),(0,255,0),1)
                #frame = cv2.line(frame,(184,int(IM_HEIGHT/4)),(184,int(3*IM_HEIGHT/4)),(0,255,0),1)
                #frame = cv2.line(frame,(224,int(IM_HEIGHT/4)),(224,int(3*IM_HEIGHT/4)),(0,255,0),1)
                frame = cv2.circle(frame,(int(self.x + self.w/2), int(self.y + self.h/2)),5,(255,0,0),2)

        #print(self.x_dev, self.y_dev)
        return frame
