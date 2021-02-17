from teleport_gazebo_model import *
import numpy as np
import tensorflow as tf
import sys

CKPT_PATH = './tf_checkpoints/positional_DQN'
EPISODES = 100
MAX_STEPS = 100 # per episode

AGENT_SIZE = 0.1
EXITS = [[0, 4*FOOT]]
EXIT_WIDTH = 0.5

class Environment:
    def __init__(self, xmin, xmax, ymin, ymax, step_size = 0.1*FOOT):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.xmin_traversable = xmin + AGENT_SIZE
        self.xmax_traversable = xmax - AGENT_SIZE
        self.ymin_traversable = ymin + AGENT_SIZE
        self.ymax_traversable = ymax - AGENT_SIZE

        self.teleport = Teleport(ros_rate=80)

        self.step_size = step_size
        self.action_space = np.array([np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4,
                                -np.pi/2, -np.pi/4, 0., np.pi/4])
        
    def reset(self):
        self.x = np.random.uniform(self.xmin_traversable, self.xmax_traversable)
        self.y = np.random.uniform(self.ymin_traversable, self.ymax_traversable)
        self.heading = np.random.uniform(-np.pi,np.pi)

        successful = self.teleport.to(self.x, self.y, self.heading)

        if not successful:
            print("collision on reset, trying again...")
            return self.reset()

        if self.out_of_bounds():
            print("out of bounds on reset, trying again...")
            return self.reset()

        return [self.x, self.y]

    def step(self, action):
        self.heading = self.action_space[action]
        self.x += self.step_size * math.cos(self.heading)
        self.y += self.step_size * math.sin(self.heading)
        
        successful = self.teleport.to(self.x, self.y, self.heading)

        if not successful:
            print("RESULT: wall collision at ({:.2f},{:.2f})".format(self.x, self.y))
            return [self.x, self.y], True

        if self.exited():
            print("RESULT: exit")
            return [self.x, self.y], True
        
        if self.out_of_bounds():
            print("RESULT: out of bounds at ({:.2f},{:.2f})".format(self.x, self.y))
            return [self.x, self.y], True

        return [self.x, self.y], False

    def exited(self):
        for e in EXITS:
            # assuming top centered exit
            if self.y > e[1] and self.x < EXIT_WIDTH/2 and self.x > -EXIT_WIDTH/2:
                return True
        return False
    
    def out_of_bounds(self):
        if self.x < self.xmin or self.x > self.xmax \
            or self.y < self.ymin or self.y > self.ymax:
            return True
        return False

    def normalized(self, position):
        x = position[0]/(self.xmax - self.xmin)
        y = position[1]/(self.ymax - self.ymin)
        return [x,y]

class DQN:
    def __init__(self, name, action_size=8):
        with tf.compat.v1.variable_scope(name):
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, 2], name='inputs')
            self.f1 = tf.compat.v1.layers.dense(self.inputs_, 64, activation=tf.nn.relu, kernel_initializer=tf.compat.v1.initializers.he_normal())
            self.f2 = tf.compat.v1.layers.dense(self.f1, 128, activation=tf.nn.relu, kernel_initializer=tf.compat.v1.initializers.he_normal())
            self.f3 = tf.compat.v1.layers.dense(self.f2, 64, activation=tf.nn.relu, kernel_initializer=tf.compat.v1.initializers.he_normal())
            self.output = tf.compat.v1.layers.dense(self.f3, action_size, activation=None)


if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    env = Environment(-4*FOOT, 4*FOOT, -4*FOOT, 4*FOOT)
    q_network = DQN('main_qn')
    
    with tf.compat.v1.Session() as sess:
        # restore DQN checkpoint
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        if ckpt:
            print("DQN restored from {}".format(ckpt.model_checkpoint_path))
            tf.compat.v1.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        else:
            print("\nERROR: No TF checkpoint files found at {}".format(CKPT_PATH))
            sys.exit()
        
        # run demo
        for episode in range(1, EPISODES+1):
            print("\nEPISODE {}".format(episode))
            state = env.reset()
            step = 0          
            
            while step < MAX_STEPS:
                q_values = sess.run(q_network.output, feed_dict={q_network.inputs_: [env.normalized(state)]})[0]           
                action = np.random.choice([idx for idx, val in enumerate(q_values) if val == np.max(q_values)])   
                state, terminate = env.step(action)
                if terminate:
                    break     
                step += 1

            if step >= MAX_STEPS:
                print("RESULT: took too many steps")
