from teleport_gazebo_model import *
import numpy as np
import tensorflow as tf
from collections import deque
import os
import shutil

USE_GAZEBO = True
RESTORE_CKPT = False

# ----- SIMULATION PARAMETERS -----
# Reinforcement Learning:
EPISODES = 10000
MAX_STEPS = 100 # per episode
GAMMA = 0.999 # future reward discount
EXPLORE_MAX = 1.0 # exploration probability
EXPLORE_MIN = 0.1  
DECAY_PERCENT = 0.5 
DECAY_RATE = 4/DECAY_PERCENT    
LEARN_RATE = 1e-4 
STEP_RWD = -0.1
EXIT_RWD = 0

# Batch Memory:
MEM_SIZE = 1000     
BATCH_SIZE = 50     

# Target Q-Network:
UPDATE_TARGET_EVERY = 1   
TAU = 0.1 # target update factor                  
TRAIN_STEP = 1    

SAVE_STEP = 100
CKPT_PATH = './tf_checkpoints/positional_DQN'

# Enviroment
AGENT_SIZE = 0
EXITS = [[0, 4*FOOT]]
EXIT_WIDTH = 0.5

class Environment:
    def __init__(self, xmin, xmax, ymin, ymax, step_size = 0.5*FOOT):

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.xmin_traversable = xmin + AGENT_SIZE
        self.xmax_traversable = xmax - AGENT_SIZE
        self.ymin_traversable = ymin + AGENT_SIZE
        self.ymax_traversable = ymax - AGENT_SIZE

        if USE_GAZEBO:
            self.teleport = Teleport(ros_rate=100)

        self.step_size = step_size
        self.action_space = np.array([np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4,
                                -np.pi/2, -np.pi/4, 0., np.pi/4])
        
    def reset(self):

        self.x = np.random.uniform(self.xmin_traversable, self.xmax_traversable)
        self.y = np.random.uniform(self.ymin_traversable, self.ymax_traversable)
        self.heading = np.random.uniform(-np.pi,np.pi)

        if USE_GAZEBO:
            successful = self.teleport.to(self.x, self.y, self.heading)

            if not successful:
                print("collision on reset, trying again...")
                return self.reset()

        if self.out_of_bounds():
            print("out of bounds on reset, trying again...")
            return self.reset()

        return [self.x, self.y]

    ####Choose random action from action list
    def choose_random_action(self):
        
        return np.random.choice(len(self.action_space)) 

    def step(self, action):

        last_x = self.x
        last_y = self.y
        last_heading = self.heading

        self.heading = self.action_space[action]
        self.x += self.step_size * math.cos(self.heading)
        self.y += self.step_size * math.sin(self.heading)
        
        if USE_GAZEBO:
            successful = self.teleport.to(self.x, self.y, self.heading)

            if not successful:
                self.x = last_x 
                self.y = last_y
                self.heading = last_heading
                successful = self.teleport.to(self.x, self.y, self.heading)
                if not successful:
                    print("ERROR: failure to recover after collision at ({:.2f},{:.2f})".format(self.x, self.y))
                    return [self.x, self.y], STEP_RWD, False, True
                return [self.x, self.y], STEP_RWD, False, False

        if self.exited():
            print("RESULT: exit")
            return [self.x, self.y], EXIT_RWD, True, True
        
        if self.out_of_bounds():
            print("RESULT: out of bounds at ({:.2f},{:.2f})".format(self.x, self.y))
            return [self.x, self.y], STEP_RWD, False, True

        return [self.x, self.y], STEP_RWD, False, False

    def exited(self):

        for e in EXITS:
            if self.y > e[1] and self.x < EXIT_WIDTH/2 and self.x > -EXIT_WIDTH/2:
                return True
        return False
    
    def out_of_bounds(self):
        if self.x < self.xmin_traversable or self.x > self.xmax_traversable \
            or self.y < self.ymin_traversable or self.y > self.ymax_traversable:
            return True
        return False

    def normalized(self, position):
        x = position[0]/(self.xmax - self.xmin)
        y = position[1]/(self.ymax - self.ymin)
        return [x,y]
        
class trfl:
    def update_target_variables(target_variables, source_variables, tau=1.0, use_locking=False, name="update_target_variables"):
    
        def update_op(target_variable, source_variable, tau):
            return target_variable.assign(tau * source_variable + (1.0 - tau) * target_variable, use_locking)

        with tf.compat.v1.name_scope(name, values=target_variables + source_variables):
            update_ops = [update_op(target_var, source_var, tau) for target_var, source_var in zip(target_variables, source_variables)]
        return tf.compat.v1.group(name="update_all_variables", *update_ops)


class DQN:
    def __init__(self, name, learning_rate=0.0001, batch_size=20, gamma=0.99, action_size=8):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.compat.v1.variable_scope(name):
            
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, 2], name='inputs')  
            self.actions_ = tf.compat.v1.placeholder(tf.int32, [batch_size], name='actions')
            
#            self.is_training = tf.placeholder_with_default(True, shape = (), name = 'is_training')
#            self.keep_prob = 0.5
            
            self.f1 = tf.compat.v1.layers.dense(self.inputs_, 64, activation=tf.nn.relu, kernel_initializer=tf.compat.v1.initializers.he_normal())
            self.f2 = tf.compat.v1.layers.dense(self.f1, 128, activation=tf.nn.relu, kernel_initializer=tf.compat.v1.initializers.he_normal())
            self.f3 = tf.compat.v1.layers.dense(self.f2, 64, activation=tf.nn.relu, kernel_initializer=tf.compat.v1.initializers.he_normal())

            self.output = tf.compat.v1.layers.dense(self.f3, action_size, activation=None)

            #TRFL way
            self.targetQs_ = tf.compat.v1.placeholder(tf.float32, [batch_size,action_size], name='target')
            self.reward = tf.compat.v1.placeholder(tf.float32,[batch_size],name="reward")
            self.discount = tf.compat.v1.constant(gamma,shape=[batch_size],dtype=tf.float32,name="discount")
      
            #TRFL qlearing
            target = tf.stop_gradient(self.reward + self.discount * tf.math.reduce_max(self.targetQs_, axis=1))

            values = tf.convert_to_tensor(self.output)
            indices = tf.convert_to_tensor(self.actions_)
            one_hot_indices = tf.one_hot(indices, tf.shape(values)[-1], dtype=values.dtype)
            qa_tm1 = tf.math.reduce_sum(values * one_hot_indices, axis=-1, keepdims=None)

            td_error = target - qa_tm1
            loss = 0.5 * tf.math.square(td_error)
            qloss = loss
            
#           qloss, q_learning = trfl.qlearning(self.output,self.actions_,self.reward,self.discount,self.targetQs_)            
            self.loss = tf.math.reduce_mean(qloss)
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
    def get_qnetwork_variables(self):
      return [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(self.name)] 

####Memory replay 
class Memory():
    def __init__(self, max_size = 500):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        
        if len(self.buffer) < batch_size:
            return self.buffer
        
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


if __name__ == '__main__':

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    env = Environment(-4*FOOT, 4*FOOT, -4*FOOT, 4*FOOT)
    memory = Memory(MEM_SIZE)
    mainQN = DQN('main_qn', LEARN_RATE, BATCH_SIZE, GAMMA)
    targetQN = DQN('target_qn', LEARN_RATE, BATCH_SIZE, GAMMA)
    
    #TRFL way to update the target network
    target_network_update_ops = trfl.update_target_variables(targetQN.get_qnetwork_variables(),mainQN.get_qnetwork_variables(),tau=TAU)
    
    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())
        
        # configure DQN checkpoints
        checkpoint_manager = tf.compat.v1.train.Saver()
        if not os.path.isdir(CKPT_PATH):
            os.makedirs(CKPT_PATH)
        if RESTORE_CKPT:
            ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
            if ckpt:
                print("DQN restored from {}".format(ckpt.model_checkpoint_path))
                checkpoint_manager.restore(sess, ckpt.model_checkpoint_path)
                shutil.rmtree(CKPT_PATH) # delete old checkpoints
            else:
                print("\nWARNING: No TF checkpoint files found... DQN initialized from scratch\n")
        
        # training
        for episode in range(1, EPISODES+1):

            print("\nEPISODE", episode)

            state = env.reset()
            step = 0          
            
            while step < MAX_STEPS:

                epsilon = EXPLORE_MIN + (EXPLORE_MAX - EXPLORE_MIN)*np.exp(-DECAY_RATE*episode/EPISODES) 

                if np.random.rand() < epsilon: # EXPLORE
                    action = env.choose_random_action()                   
                else:                          # EXPLOIT
                    q_values = sess.run(mainQN.output, feed_dict={mainQN.inputs_: [env.normalized(state)]})[0]           
                    action = np.random.choice([idx for idx, val in enumerate(q_values) if val == np.max(q_values)])   
                
                next_state, reward, exited, terminate = env.step(action)
                memory.add((env.normalized(state), action, reward, env.normalized(next_state), exited))           
                step += 1
                state = next_state          
            
                if len(memory.buffer) == MEM_SIZE and step%TRAIN_STEP==0:
                    # Sample mini-batch from memory
                    batch = memory.sample(BATCH_SIZE)
                    normalized_states = [mem[0] for mem in batch]
                    actions = [mem[1] for mem in batch]
                    rewards = [mem[2] for mem in batch]
                    normalized_next_states = [mem[3] for mem in batch]
                    finish = np.array([mem[4] for mem in batch])
                    
                    # Train network
                    target_Qs = sess.run(targetQN.output, feed_dict={targetQN.inputs_: normalized_next_states})
                    ####End state has 0 action values
                    target_Qs[finish == True] = 0.
                                        
                    #TRFL way, calculate td_error within TRFL
                    loss = sess.run([mainQN.loss, mainQN.opt], feed_dict={mainQN.inputs_: normalized_states, mainQN.targetQs_: target_Qs,
                                                            mainQN.reward: rewards, mainQN.actions_: actions})

                if terminate:
                    break

            if step >= MAX_STEPS:
                print("RESULT: took too many steps")

            if episode % SAVE_STEP ==0:
                checkpoint_manager.save(sess, os.path.join(CKPT_PATH, "positional_DQN.ckpt"), global_step=episode)
            
            #update target q network
            if episode % UPDATE_TARGET_EVERY ==0:
                sess.run(target_network_update_ops)
          
        checkpoint_manager.save(sess, os.path.join(CKPT_PATH, "positional_DQN.ckpt"), global_step=EPISODES)
