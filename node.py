from helper import Helper
import keyboard
class AStarNode():
    def __init__(self, env, parent, dist_x, dist_y, damage, death, repetitions, action, save_num):
        self.actions_data = {
            "stay_still": {"action": (1, 0, 0), "cost": 0},
            "move_left": {"action": (0, 0, 0), "cost": 0},
            "move_right": {"action": (2, 0, 0), "cost": 0},
            "jump_straight": {"action": (1, 1, 0), "cost": 0},
            "jump_left": {"action": (0, 1, 0), "cost": 0},
            "jump_right": {"action": (2, 1, 0), "cost": 0},
        }
        
        self.reached_end_count = 0
        self.parent = parent

        self.penalty = 0
        self.reward = 0

        self.kill_count_difference = 0
        self.score_difference = 0

        self.remaining_time = 0
        
        self.state = None
        self.save_num = save_num

        self.action = action
        self.repetitions = repetitions

        # if self.parent != None:
        #     self.time_elapsed = self.parent.time_elapsed + self.repetitions
        # else:
        #     self.time_elapsed = 0
        
        self.time_elapsed = (env.episode_length / 600)

        if self.parent != None:
            self.damage = self.parent.damage + damage
            self.death = self.parent.death + death
            
            self.calculate_distance_from_origin((self.parent.distance_from_origin + dist_x))

            self.pos_x = self.parent.pos_x + dist_x
            self.pos_y = self.parent.pos_y + dist_y
        else:
            self.damage = damage
            self.death = death
            
            self.calculate_distance_from_origin(dist_x)

            self.pos_x = dist_x
            self.pos_y = dist_y

        self.has_been_hurt = False
        self.is_in_visited_list = False

        self.calculate_distance_from_origin(dist_x)

    def initialize_root(self, state, save_num):
        if self.parent == None:
            self.state = state
            # dist_x = state[0]
            # dist_y = state[1]

            self.save_num = save_num

            # self.pos_x = dist_x
            # self.pos_y = dist_y

            # self.damage = 0
            # self.death = 0

            

    def calculate_distance_from_origin(self, dist_x):
        self.distance_from_origin = dist_x
        # self.remaining_time_estimated = (1 - (self.distance_from_origin / 600))
        test_time = max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000)))
        if test_time <= 0:
            print("self.distance_from_origin: " + str(self.distance_from_origin))
            print("(self.distance_from_origin / 1000): " + str((self.distance_from_origin / 1000)))
            print("1 - (self.distance_from_origin / 1000): " + str(1 - (self.distance_from_origin / 1000)))
            print("min(1.0, 1 - (self.distance_from_origin / 1000)): " + str(min(1.0, 1 - (self.distance_from_origin / 1000))))
            print("max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000))): " + str(max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000)))))
            keyboard.wait("space")

        self.remaining_time_estimated = max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000)))

        self.calculate_cost()
    
    def calculate_cost(self):
        if self.damage > 0 or self.death > 0:
            self.cost = 9999
        else:
            # self.cost = ((self.remaining_time_estimated + self.time_elapsed * 0.9) + self.penalty + (self.kill_count_difference * Helper.kill_count_reward) + (self.score_difference * Helper.score_reward))
            self.cost = ((self.remaining_time_estimated + self.time_elapsed * 0.9) + self.penalty)
        return self.cost
    
    def extract_plan(self):
        actions = []
        pos_x_list = []
        if self.parent == None:
            return actions, pos_x_list
        
        current = self.parent
        while current.parent != None:
            for i in range(current.repetitions):
                actions.append(current.action)
                pos_x_list.append(current.pos_x)
            
            current = current.parent

        actions.reverse()
        pos_x_list.reverse()
        return actions, pos_x_list

    def simulate_pos(self, env, latest_save_num, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count, best_remaining_time_estimated):
        self.state = self.parent.state

        self.reached_end_count = 0
        self.pos_x = self.parent.pos_x
        self.calculate_distance_from_origin(self.parent.pos_x)
        self.pos_y = self.parent.pos_y
        self.damage = self.parent.damage
        self.death = self.parent.death

        action_plan, pos_x_plan = self.extract_plan()

        if self.parent == None:
            raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, original_save_num)
        
            if reached_end_door:
                self.reached_end_count += 1

            self.pos_x += self.state[0]
            self.pos_y += self.state[1]
            self.damage += self.state[19]
            self.death += self.state[37]

        else:
            raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, self.parent.save_num)
        
            if reached_end_door:
                self.reached_end_count += 1

            self.pos_x += self.state[0]
            self.pos_y += self.state[1]
            self.damage += self.state[19]
            self.death += self.state[37]

        self.calculate_distance_from_origin(self.pos_x)

        latest_save_num += 1
        temp_end_save_num = latest_save_num
        raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], -latest_save_num)
        
        if reached_end_door:
            self.reached_end_count += 1
        
        self.damage += self.state[19]
        self.death += self.state[37]

        raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], 0)
        
        if reached_end_door:
            self.reached_end_count += 1
        
        self.damage += self.state[19]
        self.death += self.state[37]
        # print("save")
        # keyboard.wait("space")
        # for i in range(1):
        #     test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = env.step(self.actions_data["stay_still"]["action"], latest_save_num)
            
        #     if test_reached_end_door:
        #         self.reached_end_count += 1
        #     self.damage += test_state[19]
        #     self.death += test_state[37]
        #     # print("load")
        #     # keyboard.wait("space")
        #     test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = env.step(self.actions_data["stay_still"]["action"], 0)

        #     if test_reached_end_door:
        #         self.reached_end_count += 1
        #     self.damage += test_state[19]
        #     self.death += test_state[37]
        #     # print("move")
        #     # keyboard.wait("space")
        #     latest_save_num += 1
        #     test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = env.step(self.actions_data["stay_still"]["action"], -latest_save_num)
            
        #     if test_reached_end_door:
        #         self.reached_end_count += 1
        #     self.damage += test_state[19]
        #     self.death += test_state[37]
        #     # print("save 2")
        #     # keyboard.wait("space")
            

        
        # print("move")
        # keyboard.wait("space")
        # for i in range(2):
        #     raw_grid, temp_state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], 0)
            
        #     if reached_end_door:
        #         self.reached_end_count += 1
                
        #     self.damage += temp_state[19]
        #     self.death += temp_state[37]
        # print("check")
        # keyboard.wait("space")

        test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = env.step(self.actions_data["stay_still"]["action"], temp_end_save_num)
        # test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = env.step(self.actions_data["stay_still"]["action"], latest_save_num)

        # print("load")
        # keyboard.wait("space")
        damage = Helper.get_damage(self, self.parent)
        self.remaining_time =  (1 -(env.episode_length / 600))

        if self.is_in_visited_list:
            # self.remaining_time += Helper.visited_list_penalty
            self.penalty += Helper.visited_list_penalty

        self.has_been_hurt = damage != 0

        # if best_remaining_time_estimated > self.remaining_time_estimated:
        #     print("Good Actions: " + str(action_plan))
        #     print("Latest Action: " + str(self.action))
        #     print("")
        #     print("Good Pos X's: " + str(pos_x_plan))
        #     print("Latest Pos X: " + str(self.pos_x))
        #     print("")

        #     print("self.pos_x: " + str(self.pos_x))
        #     print("self.pos_y: " + str(self.pos_y))
        #     print("self.damage: " + str(self.damage))
        #     print("self.death: " + str(self.death))
        #     print("----")
        #     print("action_plan: " + str(action_plan))
        #     print("self.action: " + str(self.action))
        #     print("----")
            # keyboard.wait("space")

        return self.remaining_time, latest_save_num

    # def simulate_pos(self, env, latest_save_num, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count, best_remaining_time_estimated):
    #     # print("1 Sim self.pos_x: " + str(self.pos_x))
    #     self.state = self.parent.state

    #     self.reached_end_count = 0
    #     self.pos_x = original_dist_x
    #     self.calculate_distance_from_origin(original_dist_x)
    #     self.pos_y = original_dist_y
    #     self.damage = original_damage
    #     self.death = original_death

    #     self.score_difference = 0
    #     self.kill_count_difference = 0

    #     # print("2 Sim self.pos_x: " + str(self.pos_x))

    #     # for i in range(self.repetitions):
    #         # if i == 0:
    #         #     raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, self.parent.save_num)
    #         # else:
    #         #     raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, 0)

    #     action_plan = self.extract_plan()

    #     if action_plan == None:
    #         raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, original_save_num)
        
    #         if reached_end_door:
    #             self.reached_end_count += 1

    #         self.pos_x += self.state[0]
    #         self.pos_y += self.state[1]
    #         self.damage += self.state[19]
    #         self.death += self.state[37]

    #         self.score_difference = self.state[7] - original_score
    #         self.kill_count_difference = self.state[23] - original_kill_count

    #     else:
    #         for i, part_action in enumerate(action_plan):
    #             if i == 0:
    #                 raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(part_action, original_save_num)
    #             else:
    #                 raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(part_action, 0)
                
    #             if reached_end_door:
    #                 self.reached_end_count += 1

    #             self.pos_x += self.state[0]
    #             self.pos_y += self.state[1]
    #             self.damage += self.state[19]
    #             self.death += self.state[37]

    #             self.score_difference = self.state[7] - original_score
    #             self.kill_count_difference = self.state[23] - original_kill_count
    #             # keyboard.wait("space")  

    #         raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, 0)
            
    #         if reached_end_door:
    #             self.reached_end_count += 1

    #         self.pos_x += self.state[0]
    #         self.pos_y += self.state[1]
    #         self.damage += self.state[19]
    #         self.death += self.state[37]

    #         self.score_difference = self.state[7] - original_score
    #         self.kill_count_difference = self.state[23] - original_kill_count

    #     self.calculate_distance_from_origin(self.pos_x)

    #     latest_save_num += 1
    #     raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], -latest_save_num)
        
    #     if reached_end_door:
    #         self.reached_end_count += 1
        
    #     self.damage += self.state[19]
    #     self.death += self.state[37]
        
    #     for i in range(1):
    #         raw_grid, temp_state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], 0)
            
    #         if reached_end_door:
    #             self.reached_end_count += 1
                
    #         self.damage += temp_state[19]
    #         self.death += temp_state[37]

    #     test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = env.step(self.actions_data["stay_still"]["action"], latest_save_num)

    #     damage = Helper.get_damage(self, self.parent)
    #     self.remaining_time =  (1 -(env.episode_length / 600))

    #     if self.is_in_visited_list:
    #         # self.remaining_time += Helper.visited_list_penalty
    #         self.penalty += Helper.visited_list_penalty

    #     self.has_been_hurt = damage != 0

    #     # if best_remaining_time_estimated > self.remaining_time_estimated:
    #     #     print("self.pos_x: " + str(self.pos_x))
    #     #     print("self.pos_y: " + str(self.pos_y))
    #     #     print("self.damage: " + str(self.damage))
    #     #     print("self.death: " + str(self.death))
    #     #     print("----")
    #     #     print("action_plan: " + str(action_plan))
    #     #     print("self.action: " + str(self.action))
    #     #     print("----")
    #         # keyboard.wait("space")

    #     return self.remaining_time, latest_save_num

    def __lt__(self, other):
        return self.cost < other.cost

    def generate_children(self, env, save_num, latest_save_num):
        children = []
        possible_actions =  Helper.create_possible_actions(self)
        
        if self.is_leaf_node():
            possible_actions = []

        for i in range(len(possible_actions)):
            action = possible_actions[i]
            child_damage = 0
            child_death = 0
            
            raw_grid, state, reached_termination, reached_end_door, reward, done, step_info = env.step(action, save_num)

            child_damage += state[19]
            child_death += state[37]
            
            latest_save_num += 1
            raw_grid, state, reached_termination, reached_end_door, reward, done, step_info = env.step(action, -latest_save_num)

            child_damage += state[19]
            child_death += state[37]

            # print("generate_children, load, save_num: " + str(save_num))
            # print("generate_children, save, latest_save_num: " + str(latest_save_num))

            children.append(AStarNode(env, self, state[0], state[1], child_damage, child_death, self.repetitions, action, latest_save_num))

        # if self.is_leaf_node() or len(children) == 0:
        #     print("self.is_leaf_node(): " + str(self.is_leaf_node()))
        #     print("len(children): " + str(len(children)))
        #     keyboard.wait("space")
        return children, latest_save_num


    def is_leaf_node(self):
        if self.death <= 0:
            return False
        return self.death > 0     
    

    # ---
        # raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, self.parent.save_num)

        # print("action: " + str(self.action))
        # print("old pos_x: " + str(self.pos_x))
        # print("old distance_from_origin: " + str(self.distance_from_origin))
        # print("moved: " + str(self.state[0]))
        # self.pos_x += self.state[0]
        # self.pos_y += self.state[1]
        # self.damage += self.state[19]
        # self.death += self.state[37]
        # print("new pos_x: " + str(self.pos_x))
        
        # print("current action: " + str(self.action))
        # print("3 Sim self.pos_x: " + str(self.pos_x))

        # dist_x = self.state[0]
        # dist_y = self.state[1]

        # self.calculate_distance_from_origin(self.pos_x)
        # print("new distance_from_origin: " + str(self.distance_from_origin))

        # keyboard.wait("space")
        

        # print("4 Sim self.pos_x: " + str(self.pos_x))

        # print("Sim ACTION")
        # keyboard.wait("space")

        # latest_save_num += 1
        # raw_grid, temp_state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], -latest_save_num)
        # self.damage += temp_state[19]
        # self.death += temp_state[37]
        
        # for i in range(1):
        #     raw_grid, temp_state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], 0)
        #     self.damage += temp_state[19]
        #     self.death += temp_state[37]

        # print("simulate_pos, load, self.parent.save_num: " + str(self.parent.save_num))
        # print("simulate_pos, save, latest_save_num: " + str(latest_save_num))

        # damage = Helper.get_damage(self, self.parent)
        # self.remaining_time =  (1 -(env.episode_length / 600))

        # if self.is_in_visited_list:
        #     # self.remaining_time += Helper.visited_list_penalty
        #     self.penalty += Helper.visited_list_penalty

        # self.has_been_hurt = damage != 0


        # if best_remaining_time_estimated > self.remaining_time_estimated:
        #     for i in range(self.repetitions):
        #         if i == 0:
        #             raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, self.parent.save_num)
        #         else:
        #             raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, 0)
        #     print("Sim ACTION")
        #     keyboard.wait("space") 
                

        #     raw_grid, temp_state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, -latest_save_num)