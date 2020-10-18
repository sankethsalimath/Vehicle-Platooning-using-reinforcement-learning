import numpy as np

class testenv:
    def __init__(self):
        self.goal_position_x = 0
        self.goal_position_y = -10

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 1])
        return np.array(self.state)

    def step(self, action):
        self.space = (np.random.randint(0, 4), np.random.randint(0, 4))

        if action == 0:
            self.pos = -1
        elif action == 1:
            self.pos = 0
        elif action == 2:
            self.pos = 1
        elif action == 3:
            self.pos = 2

        num = np.random.randint(0, 3)
        if num == 0:
            done = True
            reward = -200
        elif num == 1:
            done = False
            reward = -1
        else:
            done = False
            reward = 1


        return np.array([self.pos, self.pos]), reward, done, None


class Qlearn:
    def __init__(self):
        self.hor_position_min = -50
        self.hor_position_max = 50
        self.ver_position_min = 20
        self.ver_position_max = 0
        self.low = np.array(
            [self.hor_position_min, self.ver_position_min], dtype=np.float32
        )
        self.high = np.array(
            [self.hor_position_max, self.ver_position_max], dtype=np.float32
        )
        self.observation_space = np.array(
            [[self.hor_position_min, self.ver_position_min], [self.hor_position_min, self.ver_position_min]], dtype=np.float32
        )
        self.action_space = 4

        self.LEARNING_RATE = 0.1

        self.DISCOUNT = 0.95
        self.EPISODES = 25000
        self.STATS_EVERY = 3000
        self.SHOW_EVERY = 3000
        self.ep_rewards = []

        self.DISCRETE_OS_SIZE = [40] * len(self.high)
        self.discrete_os_win_size = (self.high - self.low)/self.DISCRETE_OS_SIZE

        self.epsilon = 1  # not a constant, qoing to be decayed
        self.START_EPSILON_DECAYING = 1
        self.END_EPSILON_DECAYING = self.EPISODES//2
        self.epsilon_decay_value = self.epsilon/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

        self.q_table = np.random.uniform(low=-2, high=0, size=(self.DISCRETE_OS_SIZE + [self.action_space]))
        #print([20]*len(self.high))


    def get_discrete_state(self, state):
        discrete_state = (state - self.low)/self.discrete_os_win_size
        print('state:', state)
        print('low:', self.low)
        print('os size:', self.discrete_os_win_size)
        print('ds:', discrete_state)
        return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

    def update_Q(self):
        for episode in range(self.EPISODES):
            self.episode_reward = 0
            discrete_state = self.get_discrete_state(env.reset())
            done = False

            while not done:
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.q_table[discrete_state])
                    print('argmax action', action)
                else:
                    # Get random action
                    action = np.random.randint(0, self.action_space)
                    print('random action', action)


                new_state, reward, done, _ = env.step(action)
                print('new state, reward, done:', new_state, reward, done)
                self.episode_reward += reward
                self.new_discrete_state = self.get_discrete_state(new_state)
                print("ds + a:", discrete_state + (action,))
                #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                # If simulation did not end yet after last step - update Q table

                if not done:
                    # Maximum possible Q value in next step (for new state)
                    max_future_q = np.max(self.q_table[self.new_discrete_state])
                    # Current Q value (for current state and performed action)
                    current_q = self.q_table[discrete_state + (action,)]
                    # And here's our equation for a new Q value for current state and action
                    new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
                    # Update Q table with new Q value
                    self.q_table[discrete_state + (action,)] = new_q
                    #print(self.q_table)

                # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
                elif new_state[0] >= env.goal_position_x:
                    #q_table[discrete_state + (action,)] = reward
                    self.q_table[discrete_state + (action,)] = 0

                discrete_state = self.new_discrete_state

            # Decaying is being done every episode if episode number is within decaying range
            if self.END_EPSILON_DECAYING >= episode >= self.START_EPSILON_DECAYING:
                self.epsilon -= self.epsilon_decay_value

            self.ep_rewards.append(self.episode_reward)

env = testenv()
ql = Qlearn()
ql.update_Q()
