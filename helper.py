class Helper():
    # visited_list_penalty = 1500
    visited_list_penalty = 999
    score_reward = 10
    kill_count_reward = 20

    def get_damage(node, prev_node):
        damage = 0
        if node.state[19] > prev_node.state[19]:
            damage = node.state[19] - prev_node.state[19]

            if node.state[37] != 0:
                if node.pos_y < -50:
                    damage += 5
                else:
                    damage += 2

        return damage
    
    @staticmethod
    def can_jump_higher(node, check_parent):
        if node.parent != None and check_parent and Helper.can_jump_higher(node.parent, False):
            return True
        return node.state[10]
    
    @staticmethod
    def create_possible_actions(node):
        actions =  {
                        "stay_still": {"action": (1, 0, 0)},
                        "move_left": {"action": (0, 0, 0)},
                        "move_right": {"action": (2, 0, 0)},
                        "jump_straight": {"action": (1, 1, 0)},
                        "jump_left": {"action": (0, 1, 0)},
                        "jump_right": {"action": (2, 1, 0)},
                    }
        

        possible_actions = []
        can_jump = Helper.can_jump_higher(node, True)

        if can_jump:
            possible_actions.append(actions["jump_straight"]["action"])

        possible_actions.append(actions["move_right"]["action"])
        if can_jump:
            possible_actions.append(actions["jump_right"]["action"])

        possible_actions.append(actions["move_left"]["action"])
        if can_jump:
            possible_actions.append(actions["jump_left"]["action"])

        return possible_actions
        
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
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

        

        