import numpy as np
from logger import logger


ENV_OBSTACLE = 255
ENV_EMPTY = 0
ENV_AGENT = 1
ENV_ENEMY = 2
ENV_GOAL = 3

ACT_STAY = 0
ACT_UP = 1
ACT_LEFT = 2
ACT_DOWN = 3
ACT_RIGHT = 4




BOARD_WIDTH = 4
BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH



def generate_grid_world(num_agent=1, num_enemy=1,num_goal=1):
    sum_up = num_agent+num_enemy+num_goal

    matrix = np.zeros([BOARD_SIZE])
    matrix_idx = np.arrange(BOARD_SIZE)
    
    chosen_idx = np.random.choice(matrix_idx, sum_up, False)

    matrix[chosen_idx[0:num_agent]] = ENV_AGENT
    matrix[chosen_idx[num_agent:num_enemy]] = ENV_ENEMY
    matrix[chosen_idx[num_enemy:num_goal]] = ENV_GOAL


class Environment:
    
    def __init__(self):

        self.enemy_reward = 0
        self.goal_reward = 1
        self.life_reward = -1
        self.agent_pos, self.enemy_pos, self.goal_pos, self.ori_agent_pos= [],[],[], []
        self.board = self.generate_grid_world()
        self.reset()

    def generate_grid_world(self, num_agent=1, num_enemy=1,num_goal=1):

        def position_translate(idxs):
            ret = []
            for each in idxs:
                pos = np.zeros(2)
                pos[0] = each % BOARD_WIDTH  #x
                pos[1] = each // BOARD_WIDTH #y
                ret.append(pos)
            return ret

        sum_up = num_agent+num_enemy+num_goal

        matrix = np.zeros([BOARD_SIZE])
        matrix_idx = np.arange(BOARD_SIZE)
        
        chosen_idx = np.random.choice(matrix_idx, sum_up, False)

        agent_idxs = chosen_idx[0:num_agent]
        enemy_idxs = chosen_idx[num_agent:num_agent+num_enemy]
        goal_idxs = chosen_idx[num_agent+num_enemy:num_agent+num_enemy+num_goal]
        
        matrix[agent_idxs] = ENV_AGENT
        matrix[enemy_idxs] = ENV_ENEMY
        matrix[goal_idxs] = ENV_GOAL


        ret = matrix.reshape((BOARD_WIDTH, BOARD_WIDTH))

        self.agent_pos = position_translate(agent_idxs)
        self.ori_agent_pos = self.agent_pos[:]
        self.enemy_pos = position_translate(enemy_idxs)
        self.goal_pos = position_translate(goal_idxs)
        return ret
    
    def render(self):


        print("")
        for m in self.board:
            for n in m:
                if n == ENV_EMPTY:
                    print(". ",end="")
                elif n == ENV_ENEMY:
                    print("X ", end="")
                elif n == ENV_AGENT:
                    print("@ ", end="")
                elif n == ENV_GOAL:
                    print("O ", end="")
                    
            print("")
        print("")
        
            

    def reset(self):
        
        # self.board = np.zeros([BOARD_WIDTH,BOARD_WIDTH])
        # self.board = self.generate_grid_world()

        #self.render()

        # return self.board
        return np.identity(BOARD_SIZE)[int(self.ori_agent_pos[0][1])*BOARD_WIDTH + int(self.ori_agent_pos[0][0])]


    def step(self,action):

        logger("action: %s" %  action)

        new_board = self.board
        reward = 0
        done = False
        info = ""

        cur_agent_pos = self.agent_pos[0]

        observation = np.identity(BOARD_SIZE)[int(cur_agent_pos[1])*BOARD_WIDTH + int(cur_agent_pos[0])]

                
        
        new_pos = cur_agent_pos.copy()
        new_pos[1] = new_pos[1] - 1 if action == ACT_UP else new_pos[1]
        new_pos[0] = new_pos[0] - 1 if action == ACT_LEFT else new_pos[0]
        new_pos[1] = new_pos[1] + 1 if action == ACT_DOWN else new_pos[1]
        new_pos[0] = new_pos[0] + 1 if action == ACT_RIGHT  else new_pos[0]


        logger(new_pos)

        # condition 1: cannot exceed the boarder

        if (new_pos[0]<0 or new_pos[0]>BOARD_WIDTH-1 or new_pos[1]<0 or new_pos[1]>BOARD_WIDTH-1):
            reward = self.enemy_reward
            done = True
            return observation, reward, done, info

        if new_pos[0] < 0:
            new_pos[0] = 0 
        if new_pos[0] > BOARD_WIDTH-1:
            new_pos[0] = BOARD_WIDTH-1    
        if new_pos[1] < 0:
            new_pos[1] = 0 
        if new_pos[1] > BOARD_WIDTH-1:
            new_pos[1] = BOARD_WIDTH-1    


        logger(new_pos)
        # condition 2: collition detection

        if self.collition(new_pos, self.enemy_pos):
            #meet enemy, done
            reward = self.enemy_reward
            done = True
        elif self.collition(new_pos, self.goal_pos):
            reward = self.goal_reward
            done = True
            # print ("success!")
        else:

            #moving the agent

            new_board[int(cur_agent_pos[1])][int(cur_agent_pos[0])] = ENV_EMPTY
            
            new_board[int(new_pos[1])][int(new_pos[0])] = ENV_AGENT

            self.agent_pos[0]= new_pos.copy()

            observation = np.identity(BOARD_SIZE)[int(new_pos[1])*BOARD_WIDTH + int(new_pos[0])]
            
          #update the board
            self.board = new_board
        

        # if done:
            # print("Done!")

        
        return observation, reward, done, info


    def collition(self, pos1, pos2):
        return np.all(pos1==pos2)

        







if __name__ == "__main__":
    print(__file__)
    print("test mode")

    env = Environment()

    print(env.board)
    print(env.agent_pos)
    print(env.enemy_pos)
    print(env.goal_pos)

    d = False
    for m in range(100):
        ob, r, d, i = env.step(np.random.randint(0,5))
        print(ob)

        if d:
            env.render()
            print("Done in %d times with reward %d" % (m,r))
            break

    print(env.reset())

    

    



