import glob
import os
import sys
import numpy as np
import random
import time
import cv2
import math
from PIL import Image
from threading import Thread


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' %(
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SECONDS_PER_EPISODE = 10

class CarEnv:
    SHOW_CAM = False
    STEER_AMT = 1.0
    IM_WIDTH = 460
    IM_HEIGHT = 300
    front_camera = None
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.mustang = self.blueprint_library.filter("mustang")[0]
        self.face_cascade = cv2.CascadeClassifier('cars.xml')



    def reset(self, autoPilot=False):
        print("****RESET****")
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.tran1 = carla.Transform(carla.Location(x=-5.93776, y=79.279, z=1.8431), carla.Rotation(0, 90, 0))
        self.tran2 = carla.Transform(carla.Location(x=-5.93776, y=99.279, z=1.8431), carla.Rotation(0, 90, 0))
        self.vehicle = self.world.spawn_actor(self.model_3, self.tran1)
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(autoPilot)

        print(self.transform)



        self.vehicle2 = self.world.spawn_actor(self.mustang, self.tran2)
        self.actor_list.append(self.vehicle2)
        self.vehicle2.set_autopilot(True)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)
        """
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        """
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


        self.state = np.array([np.random.uniform(low=0, high=20), np.random.uniform(low=0, high=50)])
        return np.array(self.state)
        #return self.front_camera

    def step(self, action):
        print("****ACTION****")
        self.space = (self.y_dev+self.h, self.x_dev)

        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
            print("steer left")
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
            print("throttle")
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
            print("steer right")
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            print("brake")

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if self.x_dev < 10 or self.x_dev > -10:
            done = False
            reward = 1
        elif self.y_dev < 10 or self.y_dev > -10:
            done = False
            reward = 3
        elif self.y_dev < -30:
            done = True
            reward = -10
        else:
            done = False
            reward = -2

        time.sleep(1/60)


        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return np.array(self.space), reward, done, None

    def process_img(self, image):
        #image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        i = np.array(i)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        self.frame = cv2.UMat(i2[:, :, :3])
        cars = self.face_cascade.detectMultiScale(self.frame, 1.1, 10)
        ncars = 0

        for self.loc in cars:
            #print(x, y, w, h)
            self.x, self.y, self.w, self.h = self.loc
            if abs(self.x - 197) < 100 and abs(self.y - 122) < 100:
                self.x_dev, self.y_dev = self.x - 197, self.y- 122

            #if w*h > 1200:
            #print(self.x_dev, self.y_dev, self.w, self.h)
            new_img = cv2.rectangle(self.frame,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,0,255),2)
            ncars = ncars + 1

            cv2.imshow("Result", self.frame)
            cv2.waitKey(1000)
        #cv2.destroyAllWindows()
        self.final = self.frame.get()
        self.front_camera = self.final



if __name__ == '__main__':

    env = CarEnv()

    hor_position_min = -200
    hor_position_max = 200
    ver_position_min = -200
    ver_position_max = 200
    low = np.array(
        [hor_position_min, ver_position_min], dtype=np.float32
    )
    high = np.array(
        [hor_position_max, ver_position_max], dtype=np.float32
    )
    observation_space = np.array(
        [[hor_position_min, ver_position_min], [hor_position_min, ver_position_min]], dtype=np.float32
    )
    action_space = 4

    LEARNING_RATE = 0.1

    DISCOUNT = 0.95
    EPISODES = 10
    STATS_EVERY = 1
    SHOW_EVERY = 1
    ep_rewards = []

    DISCRETE_OS_SIZE = [20] * len(high)
    discrete_os_win_size = (high - low)/DISCRETE_OS_SIZE

    epsilon = 1  # not a constant, qoing to be decayed
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES//2
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    env.goal_position_x = 0
    q_table = np.random.uniform(low=-2, high=2, size=(DISCRETE_OS_SIZE + [action_space]))
    #print([20]*len(self.high))


    def get_discrete_state(state):
        discrete_state = (state - low)/discrete_os_win_size
        """
        print('q table size:', self.q_table.shape)
        print('state:', state)
        print('low:', self.low)
        print('os size:', self.discrete_os_win_size)
        print('ds:', discrete_state)
        """
        return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

    for episode in range(EPISODES):
        print("EPISODE:", episode)

        episode_reward = 0
        discrete_state = get_discrete_state(env.reset())
        done = False

        while True:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
                print('argmax action', action)
            else:
                # Get random action
                action = np.random.randint(0, action_space)
                print('random action', action)


            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)
            print("episode reward:", episode_reward)
            if done:
                break
            """
            print("discrete state:")
            print('new state, reward, done:', new_state, reward, done)
            print("ds + a:", discrete_state + (action,))
            """
            #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # If simulation did not end yet after last step - update Q table

            if not done:
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])
                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]
                # And here's our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q
                #print(self.q_table)

            # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
            elif new_state[0] >= env.goal_position_x:
                #q_table[discrete_state + (action,)] = reward
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        ep_rewards.append(episode_reward)
        print('\ndestroying %d actors' % len(env.actor_list))
        env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])

    print("----------------------")
