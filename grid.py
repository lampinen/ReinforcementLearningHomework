class Grid(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.s = 4*5+0
        return self.s
    def step(self,action):
        row = int(self.s/5)
        col = self.s % 5
        print(row,col,action)
        if action == 0:
            col = min(col+1,4)
        elif action == 1:
            row = max(row-1,0)
        elif action == 2:
            col = max(col-1,0)
        elif action == 3:
            row = min(row+1,4)
        print(row,col)
        self.s = row*5+col
        if row == 4 and col ==4:
            reward = 1.0
            term = True
        elif col > 0 and col < 4 and row > 2:
            reward = -1.0
            term = True
        else:
            reward = 0.0
            term = False
        return self.s,reward,term

