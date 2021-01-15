####Evacuation_Continuum_Ob_center_DQN_Gazebo_Position.py
####Training 1 agent for evacuation in Gazebo at 1 exit using Deep Q-Network

from GazeboUtils import *

control = DiffDriveControl()
control.stop()

import numpy as np
import tensorflow as tf
from collections import deque
# math import pi, acos
import os
import shutil
#import time

######Initialize systerm parameters
fm_scalar = 0.3048
agent_size = 0.135       #meters
door_size = 0.5          #meters

dis_lim = (door_size)/2      #set the distance from the exit which the agent is regarded as exited 
reward = -0.1
end_reward = 0.

######Initialize Exit positions range [0, 1]
Exit = []  ###No exit
Exit.append( fm_scalar*np.array([9, 4.5]) )  ##Add Right exit

######Creating Model Saving Directory
model_saved_path = './model'

if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)
    
model_saved_path = model_saved_path + '/Continuum_Ob_Center_DQN_Dense_Hybrid'    
name_mainQN = 'main_qn_ob_center'    
name_targetQN = 'target_qn_ob_center'


class Environment:    
    def __init__(self,  xmin = 0., xmax = 1., ymin = 0., ymax = 1., rot_deg = np.pi/4, step_size = fm_scalar):
        
        ####Robot initialization
        self.size = agent_size
        self.base = rot_deg
        self.robot_pos = control.stop()[:2]
        self.trajectory = 0

        ####Set exit information
        self.Exit = []
        for e in Exit:            
            self.Exit.append(e)
   
        ####Set action space        
        self.action = np.array([np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4,
                                -np.pi/2, -np.pi/4, 0., np.pi/4]) ## 8 actions
        self.step_dis = step_size
        
        ####set reward
        self.reward = reward     ###reward for taking each time step
        self.end_reward = end_reward  ###reward for exit

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
    ####Reset initial configuration the Continuum cell space
    def reset(self):
        
#        self.robot_pos = self.initial_pos

        x = np.random.uniform(self.xmin+self.size, self.xmax-self.size)
        y = np.random.uniform(self.xmin-self.size, self.ymax-self.size)
        theta = np.random.uniform(-np.pi,np.pi)
        theta_degrees = theta*180/np.pi

        self.robot_pos = np.array([x,y])
        self.trajectory = theta

        
        control.set_state(x,y, theta_degrees)

        return (self.robot_pos[0], self.robot_pos[1])        

    ####Choose random action from action list
    def choose_random_action(self):
        
        action = np.random.choice(len(self.action))
        return action

    ###Updates position array and checks if exited   
    def step(self, action):

        reward = self.reward
        done = False
        angle = self.action[action]       ####desired trajectroy 
        
        control.rotate_to(angle)
        self.robot_pos = control.move_forward(self.step_dis)[:2]

        for e in self.Exit:
            exit_dis = self.robot_pos - e
            exit_dis = np.sqrt(np.sum(exit_dis**2))
            
            if exit_dis < dis_lim:
                done = True
                reward = self.end_reward
                break
        
        next_state = (self.robot_pos[0], self.robot_pos[1])
        
        return next_state, reward, done                
        
class trfl:
    def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):
    
        def update_op(target_variable, source_variable, tau):
            if tau == 1.0:
                return target_variable.assign(source_variable, use_locking)
            else:
                return target_variable.assign(tau * source_variable + (1.0 - tau) * target_variable, use_locking)

        with tf.compat.v1.name_scope(name, values=target_variables + source_variables):
            update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var in zip(target_variables, source_variables)]
        return tf.compat.v1.group(name="update_all_variables", *update_ops)


class DQN:
    def __init__(self, name, learning_rate=0.0001, gamma = 0.99,
                 action_size=8, batch_size=20):
        
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
    
    train_episodes = 10        # max number of episodes to learn from
    max_steps = 100                # max steps in an episode
    gamma = 0.999                   # future reward discount

    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.1            # minimum exploration probability 
    decay_percentage = 0.5          
    decay_rate = 4/decay_percentage ####exploration decay rate
            
    # Network parameters
    learning_rate = 1e-4         # Q-network learning rate 
    
    # Memory parameters
    memory_size = 1000          # memory capacity
    batch_size = 50                # experience mini-batch size
    pretrain_length = batch_size   # number experiences to pretrain the memory
    
    #target QN
    update_target_every = 1   ###target update frequency
    tau = 0.1                 ###target update factor
    save_step = 1             ###steps to save the model
    train_step = 1            ###steps to train the model

    #jet = Jetbot()            ###instance of robot movement class
    
    env = Environment(0, 9*fm_scalar, 0, 9*fm_scalar)
    state = env.reset()
    
    memory = Memory(max_size=memory_size)
        
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    
    ####set mainQN for training and targetQN for updating
    mainQN = DQN(name=name_mainQN, learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
    targetQN = DQN(name=name_targetQN,  learning_rate=learning_rate,batch_size=batch_size, gamma = gamma)
 
    #TRFL way to update the target network
    target_network_update_ops = trfl.update_target_variables(targetQN.get_qnetwork_variables(),mainQN.get_qnetwork_variables(),tau=tau)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(max_to_keep= 10) 
    
    ######GPU usage fraction
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    #####pretrain load
#    name_list = []
#    t_list = []
#    for t in mainQN.get_qnetwork_variables():
#        name_list.append('main_qn_2exits' + t.name[17:-2])
#        t_list.append(t)
#    
#    var_dict = dict(zip(name_list, t_list))
#    saver_load_main = tf.train.Saver(var_list = var_dict)
#
#    name_list = []
#    t_list = []
#    for t in targetQN.get_qnetwork_variables():
#        name_list.append('main_qn_2exits' + t.name[19:-2])
#        t_list.append(t)
#    
#    var_dict = dict(zip(name_list, t_list))
#    saver_load_target = tf.train.Saver(var_list = var_dict)    
    
    ##############
    
    with tf.compat.v1.Session(config = config) as sess:
        
        sess.run(init)
        
        ####check saved model to continue or start from initialiation
        if not os.path.isdir(model_saved_path):
            os.mkdir(model_saved_path)
        
        checkpoint = tf.train.get_checkpoint_state(model_saved_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
#            saver_load_main.restore(sess, checkpoint.all_model_checkpoint_paths[1])
#            saver_load_target.restore(sess, checkpoint.all_model_checkpoint_paths[1])
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            
            print("Removing check point and files")
            for filename in os.listdir(model_saved_path):
                filepath = os.path.join(model_saved_path, filename)
                
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)
                
            print("Done")
            
        else:
            print("Could not find old network weights. Run with the initialization")
            sess.run(init)
        ####
        
        step = 0     
        
        # if not os.path.isdir(output_dir):
        #     os.mkdir(output_dir)
        
        for ep in range(1, train_episodes+1):
            total_reward = 0
            t = 0            
            
            while t < max_steps:
                print("time:", t, "position:", env.robot_pos)
                ###### Explore or Exploit 
                epsilon = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*ep/train_episodes) 
                feed_state = np.array(state)
                print("STATE:")
                print(feed_state)
                
                print("EPSILON: ",epsilon)
                testttt = np.random.rand()
                print("RAND: ",testttt)
                if testttt < epsilon:   
                    ####Get random action
                    action= env.choose_random_action()                   
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: feed_state[np.newaxis, :]}
                    Qs = sess.run(mainQN.output, feed_dict=feed)[0]  
                    print("Qs:")  
                    for i in enumerate(Qs):
                        print(i) 

                    
                    action_list = [idx for idx, val in enumerate(Qs) if val == np.max(Qs)]                    
                    action = np.random.choice(action_list)
                #######                
                
                next_state, reward, done = env.step(action)
                                                
                total_reward += reward
                step += 1
                t += 1
                
                state = next_state

                feed_next_state = np.array(next_state)               
                
                memory.add((feed_state, action, reward, feed_next_state, done))
                        
                if done:
                    control.stop()
                    print("Robot has exited")
                    break
            
                if len(memory.buffer) == memory_size and t%train_step==0:
                    # Sample mini-batch from memory
                    batch = memory.sample(batch_size)
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    next_states = np.array([each[3] for each in batch])
                    finish = np.array([each[4] for each in batch])
                    
                    # Train network
                    target_Qs = sess.run(targetQN.output, feed_dict={targetQN.inputs_: next_states})
                    ####End state has 0 action values
                    target_Qs[finish == True] = 0.
                                        
                    #TRFL way, calculate td_error within TRFL
                    loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                        feed_dict={mainQN.inputs_: states,
                                                   mainQN.targetQs_: target_Qs,
                                                   mainQN.reward: rewards,
                                                   mainQN.actions_: actions})
            
         
            if len(memory.buffer) == memory_size:
                print("Episode: {}, Loss: {}, steps per episode: {}".format(ep,loss, t))
                
            if ep % save_step ==0:
                saver.save(sess, os.path.join(model_saved_path, "Evacuation_Continuum_model.ckpt"), global_step = ep)
            
            #update target q network
            if ep % update_target_every ==0:
                sess.run(target_network_update_ops)
#            
#            
        saver.save(sess, os.path.join(model_saved_path, "Evacuation_Continuum_model.ckpt"), global_step= train_episodes)
