
#from environment import Environment, BOARD_SIZE, BOARD_WIDTH
from logger import logger
import World

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import threading
import time
from random import randint

try:
    xrange = xrange
except:
    xrange = range


#env = Environment()

gamma = 0.99
actions = ["up", "down", "left", "right"]
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def do_action(action):
    s = World.ob
    r = -World.score
    reset = True
    if action == actions[0]:
        reset = World.try_move(0, -1)
    elif action == actions[1]:
        reset = World.try_move(0, 1)
    elif action == actions[2]:
        reset = World.try_move(-1, 0)
    elif action == actions[3]:
        reset = World.try_move(1, 0)
    #else:
    #    return
    s2 = World.ob
    r += World.score
    return s, reset, r, s2



class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))




def run():
    tf.reset_default_graph() #Clear the Tensorflow graph.

    myAgent = agent(lr=1e-2,s_size=World.BOARD_SIZE,a_size=4,h_size=8) #Load the agent.

    total_episodes = 70000 #Set total number of episodes to train agent on.
    max_ep = 999
    update_frequency = 5
    init = tf.global_variables_initializer()
    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_lenght = []
            
        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        time.sleep(1)
        while i < total_episodes:
            #s = env.reset()
            s = World.restart_game()
            running_reward = 0
            ep_history = []
            #print ("test....... ", i)
            for j in range(max_ep):
                #Probabilistically pick an action given our network outputs.
                a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in:[s]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)
                (s, d, r, s1) = do_action(actions[a])
                #print ("action ", actions[a])

                #s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
                ep_history.append([s,a,r,s1])
                s = s1
                running_reward += r
                if d == True:
                    #Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:,2] = discount_rewards(ep_history[:,2])
                    feed_dict={myAgent.reward_holder:ep_history[:,2],
                            myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                    grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad

                    if i % update_frequency == 0 and i != 0:
                        feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0
                    
                    total_reward.append(running_reward)
                    total_lenght.append(j)
                    break

            
                #Update our running tally of scores.
            if i % 100 == 0:
                print(np.mean(total_reward[-100:]))
            i += 1
            time.sleep(0.1)
            

def ghost_run():
    time.sleep(1)
    ghost_move = [(0,-1),(0, 1),(-1, 0),(1, 0)]
    while True:
        # Pick the right action
        ghost_random_option = ghost_move[randint(0, 3)]
        World.try_move_ghost(ghost_random_option[0], ghost_random_option[1])

        if World.has_restarted():
            World.restart_game()
            time.sleep(0.01)

        # MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
        time.sleep(0.1)



actions = World.actions
t = threading.Thread(target=run)
ghost_t = threading.Thread(target=ghost_run)
ghost_t.daemon = True
t.daemon = True
ghost_t.start()
t.start()
World.start_game()


