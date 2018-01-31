from tkinter import *
import numpy as np
master = Tk()

triangle_size = 0.1
cell_score_min = -0.2
cell_score_max = 0.2
Width = 64
(x, y) = (8, 8)
actions = ["up", "down", "left", "right"]
BOARD_SIZE = x*y
board = Canvas(master, width=x*Width, height=y*Width)
player = (0, y-1)
score = 1
restart = False
walk_reward = -0.04
ghost = (int((x-1)/2), int((y-1)/2))
trophy = (4, 0)
walls = [(1, 1), (1, 2), (2, 1), (2, 2), (y-2, int((y-1)/2)), (y-3, int((y-1)/2)), (y-2, int((y-1)/2)-1), (y-3, int((y-1)/2)-1) ]
#walls = [(1, 1), (1, 2), (2, 1), (2, 2)]
trophy_info = (4, 0, "green", 30)
ghost_info = (ghost[0], ghost[1], "red", -30)
#specials = [(ghost[0], ghost[1], "red", -1), (4, 0, "green", 1)]
specials = [ghost_info, trophy_info]
cell_scores = {}
ob = np.identity(BOARD_SIZE)[int(player[1])*x + int(player[0])]


def create_triangle(i, j, action):
    if action == actions[0]:
        return board.create_polygon((i+0.5-triangle_size)*Width, (j+triangle_size)*Width,
                                    (i+0.5+triangle_size)*Width, (j+triangle_size)*Width,
                                    (i+0.5)*Width, j*Width,
                                    fill="white", width=1)
    elif action == actions[1]:
        return board.create_polygon((i+0.5-triangle_size)*Width, (j+1-triangle_size)*Width,
                                    (i+0.5+triangle_size)*Width, (j+1-triangle_size)*Width,
                                    (i+0.5)*Width, (j+1)*Width,
                                    fill="white", width=1)
    elif action == actions[2]:
        return board.create_polygon((i+triangle_size)*Width, (j+0.5-triangle_size)*Width,
                                    (i+triangle_size)*Width, (j+0.5+triangle_size)*Width,
                                    i*Width, (j+0.5)*Width,
                                    fill="white", width=1)
    elif action == actions[3]:
        return board.create_polygon((i+1-triangle_size)*Width, (j+0.5-triangle_size)*Width,
                                    (i+1-triangle_size)*Width, (j+0.5+triangle_size)*Width,
                                    (i+1)*Width, (j+0.5)*Width,
                                    fill="white", width=1)


def render_grid():
    global specials, walls, Width, x, y, player
    for i in range(x):
        for j in range(y):
            board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="white", width=1)
            temp = {}
            for action in actions:
                temp[action] = create_triangle(i, j, action)
            cell_scores[(i,j)] = temp
    #for (i, j, c, w) in trophy:
    board.create_rectangle(trophy_info[0]*Width, trophy_info[1]*Width, (trophy_info[0]+1)*Width, (trophy_info[1]+1)*Width, fill=trophy_info[2], width=1)
    for (i, j) in walls:
        board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="black", width=1)

render_grid()


def set_cell_score(state, action, val):
    global cell_score_min, cell_score_max
    triangle = cell_scores[state][action]
    green_dec = int(min(255, max(0, (val - cell_score_min) * 255.0 / (cell_score_max - cell_score_min))))
    green = hex(green_dec)[2:]
    red = hex(255-green_dec)[2:]
    if len(red) == 1:
        red += "0"
    if len(green) == 1:
        green += "0"
    color = "#" + red + green + "00"
    board.itemconfigure(triangle, fill=color)


def try_move_ghost(dx, dy):
    global ghost, x, y, score, walk_reward, ghost_locate, restart
    if restart == True:
        restart_game()
    new_x = ghost[0] + dx
    new_y = ghost[1] + dy
    #score += walk_reward
    if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and not ((new_x, new_y) in walls) and not ((new_x, new_y) == trophy):
        board.coords(ghost_locate, new_x*Width, new_y*Width, (new_x+1)*Width, (new_y+1)*Width)
        ghost = (new_x, new_y)
        '''
    for (i, j, c, w) in specials:
        if new_x == i and new_y == j:
            score -= walk_reward
            score += w
            if score > 0:
                print "Success! score: ", score
            else:
                print "Fail! score: ", score
            restart = True
            return
        '''


def try_move(dx, dy):
    global player, x, y, score, walk_reward, me, restart, ghost
    if restart == True:
        restart_game()
    new_x = player[0] + dx
    new_y = player[1] + dy
    score += walk_reward
    if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and not ((new_x, new_y) in walls):
        board.coords(me, new_x*Width+Width*2/10, new_y*Width+Width*2/10, new_x*Width+Width*8/10, new_y*Width+Width*8/10)
        player = (new_x, new_y)
    specials_x = [(ghost[0], ghost[1], "red", -30), trophy_info]
    for (i, j, c, w) in specials_x:
        if new_x == i and new_y == j:
            score -= walk_reward
            score += w
            if score > 0:
                print ("Success! score: ", score)
            else:
                print ("Fail! score: ", score)
            restart = True
            return restart
    return restart
    #print "score: ", score


def call_up(event):
    try_move(0, -1)


def call_down(event):
    try_move(0, 1)


def call_left(event):
    try_move(-1, 0)


def call_right(event):
    try_move(1, 0)


def restart_game():
    global player, score, me, restart, ghost_locate, ghost
    player = (0, y-1)
    ghost = (int((x-1)/2), int((y-1)/2))
    score = 1
    restart = False
    board.coords(me, player[0]*Width+Width*2/10, player[1]*Width+Width*2/10, player[0]*Width+Width*8/10, player[1]*Width+Width*8/10)
    board.coords(ghost_locate, ghost[0]*Width, ghost[1]*Width, (ghost[0]+1)*Width, (ghost[1]+1)*Width)
    return ob

def has_restarted():
    return restart
    
me = board.create_rectangle(player[0]*Width+Width*2/10, player[1]*Width+Width*2/10,
                            player[0]*Width+Width*8/10, player[1]*Width+Width*8/10, fill="orange", width=1, tag="me")

ghost_locate = board.create_rectangle(ghost[0]*Width, ghost[1]*Width, (ghost[0]+1)*Width, (ghost[1]+1)*Width, fill="red", width=1, tag="ghost")

board.grid(row=0, column=0)


def start_game():
    master.mainloop()
