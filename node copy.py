class AStarNode():
    def __init__(self, parent, action_name, action, depth, score, dist_to_end, elapsed_time, cost, dead, damaged, save_id):
        self.parent = parent 
        self.action_name = action_name
        self.action = action
        self.depth = depth
        self.score = score
        self.dist_to_end = dist_to_end
        self.elapsed_time = elapsed_time
        self.cost = cost
        self.dead = dead
        self.damaged = damaged
        self.save_id = save_id

    def __lt__(self, other):
        return self.score > other.score
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # def __init__(self, action, repetitions, parent, env):
    #     self.action = action
    #     self.repetitions = repetitions
    #     self.parent = parent  
    #     self.pos_x = self.parent.pos_x
    #     self.env = env

    #     self.is_in_visited_list = False
    #     self.has_been_hurt = False
    
    # def calculate_score(pos_x, damage, death):
    #     if death > 0:
    #         return 0
    #     else:
    #         return pos_x / (damage + 1)


    # def initializeRoot(self, env, save_load_num):
    #     if self.parent == None:
    #         self.starting_save_load_num = save_load_num
    #         self.pos_x = 0
    #         self.remaining_time_estimated = env.episode_length
        
    # def simulate_pos(self):
    #     damage = 0
    #     death = 0

    #     for i in range(self.repetitions):
    #        self.env.step(self.action, 0)
    #        death += self.env.state[37]
    #        damage += self.env.state[19]
    #        self.pos_x += self.env.state[0]

    #     score = self.calculate_score(self.pos_x, damage, death)

    #     if self.is_in_visited_list:
    #         score -= 1500

    #     self.has_been_hurt = (damage > 0)

    #     return score

        

        