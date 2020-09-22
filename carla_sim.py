import glob
import os
import sys
import numpy as np
import random
import time
import cv2
import math
from object_detection import real_time_yolo as ry

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' %(
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = True

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    IM_WIDTH = 640
    IM_HEIGHT = 480
    front_camera = None
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.det = ry.tiny_yolo_detect()


    def transform(self, x, y, z):
        self.actor_list = []
        self.transform=random.choice(self.world.get_map().get_spawn_points())
        print(self.transform)
        #self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.vehicle = self.world.spawn_actor(self.model_3, carla.Transform(carla.Location(x, y, z), carla.Rotation(0, 90, 0)))
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(True)

    def cam(self):
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        self.camera_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        self.camera_bp.set_attribute("fov", f"110")

        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        print('created %s' % self.camera.type_id)
        #print(self.actor_list)
        self.camera.listen(lambda data: self.process_img(data))

        while self.front_camera is None:
            time.sleep(10)

        return self.front_camera

    def sem_seg(self):
        self.camera_sem = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.camera_sem.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        self.camera_sem.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        self.camera_sem.set_attribute("fov", f"110")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(self.camera_sem, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        print('created %s' % self.camera.type_id)

        self.camera.listen(lambda data: self.process_img_sem(data))

        while self.front_camera is None:
            time.sleep(40)

        return self.front_camera

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i)
        #print(i.shape)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3
        return i3/255.0

    def process_img_sem(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        i = np.array(i)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        r, g, b =  i3[:, :, 0], i3[:, :, 1], i3[:, :, 2]
        mask = (r != 0) & (g != 0) & (b != 142)
        i3[:, :, :3][mask] = [0, 0, 0]
        #print(i3[1, 1, :3])
        if self.SHOW_CAM:
            self.det.load_img(i3)
            #cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3
        return i3/255.0




if __name__ == '__main__':
    env1 = CarEnv()
    env2 = CarEnv()
    env3 = CarEnv()
    car1 = env1.transform(x=-5.93776, y=109.279, z=1.8431)
    car2 = env2.transform(x=-5.93776, y=119.279, z=1.8431)
    car3 = env3.transform(x=-5.93776, y=129.279, z=1.8431)
    env1.sem_seg()
