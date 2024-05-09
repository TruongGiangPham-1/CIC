# scafold gridworld with column and beam
from turtle import color
import gymnasium as gym
import random
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
MAX_TIMESTEP = 3000

# import enum


class Action:
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5

    PLACE_BEAM = 6
    PLACE_COLUMN = 7


    PLACE_ELECTRICAL = 8
    PLACE_PIPE = 9

    PLACE_SCAFOLD = 10
    REMOVE_SCAFOLD = 11



action_enum = Action
class ScaffoldGridWorldEnv(gym.Env): 
    """
        3 TUPLE element state
        1. agent position (N, N, N)
        3. building zone (N, N, N, K) K for possibility of K mixtures of column and beams
        4. target zone

        block description:

        ACTION:
        current Rule:

    """
    neighbors = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    SCAFFOLD = -1
    EMPTY = 0

    COL_BLOCK = 0.5
    BEAM_BLOCK = 1


    PIPE_BLOCK = 0.25
    ELECTRICAL_BLOCK = 0.75

    def __init__(self, dimension_size, path: str, mixture_capacity=2, num_agents=1, debug=False):
        super(ScaffoldGridWorldEnv, self).__init__()
        self.action_enum = Action
        #self.reward = Reward()
        agent_pos_space = spaces.Box(low=0, high=1, shape=(dimension_size, dimension_size, dimension_size), dtype=float)
        grid_space = spaces.Box(low=-1, high=1, shape=(dimension_size, dimension_size, dimension_size, mixture_capacity), dtype=float)
        self.observation_space = spaces.Dict({
            'agent_position': agent_pos_space,
            'building_zone': grid_space,
            'target': grid_space

        })
        self.dimension_size = dimension_size
        self.mixture_capacity = mixture_capacity

        self.timestep_elapsed = 0
        self.finished_structure = False
        self.finished_structure_with_scafold = False
        
        self.record_sequence = []
        # 1 for building zone, 1 for target, 1 for each agent position, and 1 for all agents position
        # Order: building zone, agent position(s), target, all other agents position
        self.obs = np.zeros((3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=float) #TOOD: i dont know anything yet

        self.all_targets = None
        self._initialized = False

        #self.building_zone = self.obs[0]
        #self.agent_pos_grid = self.obs[1]
        #self.target = self.obs[2]
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size, self.mixture_capacity), dtype=float)
        self.agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=float)
        self.target = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size, self.mixture_capacity), dtype=float)
       
        self.num_agents = num_agents

        #self.reset()

        self._init_target()
        
        
        if debug:
            self._placeAgentInBuildingZone()

    
    # in multiagent, make sure only one agent is allowed to call this function(meta agent for example)
    def reset(self, seed=None, options=None):
        #self.mutex.acquire()  # get lock to enter critical section
        self.building_zone.fill(0)
        self.finished_structure = False
        self.finished_structure_reward_used = False
        self.finished_structure_with_scafold = False

        self.AgentsPos = np.zeros((self.num_agents, 3), dtype=int)

        random_start_pos = np.zeros(3, dtype=int)
        for i in range(self.num_agents):
            random_start_pos[0] = np.random.randint(0, self.dimension_size)
            random_start_pos[1] = np.random.randint(0, self.dimension_size)
            random_start_pos[2] = 0
            self.AgentsPos[i] = random_start_pos
            #self.building_zone[random_start_pos[0], random_start_pos[1], random_start_pos[2]] = 1  # encode agents position on the building zone
        self.action_space = spaces.Discrete(10)   
        #self._init_target()
        
        self.record_sequence = []
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.timestep_elapsed = 0
        self._init_obs()

        


        np.copyto(self.obs[2], random.choice(self.all_targets))

        # change all instance of 1 to 0.5 in self.obs
        self.obs[2][self.obs[2] == 1] = ScaffoldGridWorldEnv.COL_BLOCK
        # change all instance of 2 to 1 in self.obs
        self.obs[2][self.obs[2] == 2] = ScaffoldGridWorldEnv.BEAM_BLOCK

        obs = self.get_obs(0)
        #self.mutex.release()
        return obs, {}

    # return (3 x N x N x N) tensor for now, 
    def get_obs(self, agent_id=0):
        # clear agent_pos_grid
        self.agent_pos_grid.fill(0)
        #agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        #agent_pos_grid[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1
        self.agent_pos_grid[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1

        other_agents_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        for i in range(self.num_agents):
            if i != agent_id:
                other_agents_pos_grid[self.AgentsPos[i][0], self.AgentsPos[i][1], self.AgentsPos[i][2]] = 1

        #TODO: concat other_agents_pos_grid when doing multiagent
        #return np.stack((self.building_zone, agent_pos_grid, self.target), axis=0)
        return self.obs
        
    def _get_info(self):
        pass

    def _init_obs(self):
        if self._initialized:
            return
        self.all_targets = self.loader.load_all()
        # convert all targets to float
    

        assert len(self.all_targets) > 0, "No target found\n"
        for i in range(len(self.all_targets)):
            assert self.all_targets[i].shape[0] == self.dimension_size, \
                (f"Dimension mismatch: Target: {self.all_targets[i].shape}, "
                 f"Environment: {self.dimension_size}\n"
                 "TODO: more flexibility")
        self._initialized = True

    def _placeBlockInBuildingZone(self):
        # place some block in building zone for testing
        self.building_zone[self.dimension_size // 2, self.dimension_size // 2, 0] = -1
        return

    def _placeAgentInBuildingZone(self):
        # place some agent in building zone for testing
        self.AgentsPos[0] = [self.dimension_size//2, self.dimension_size//2, 0]
        return

    def _init_target(self):
        self.target = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size, self.mixture_capacity), dtype=float)
        
        # Hardcoding 4 columns, and 4 beams across the 4 columns. TO BE CHANGED TO BE MORE DYNAMIC AND USEABLE
        possible_targets = []
        
        points = []
        for i in range(self.dimension_size):  # all column beams
            #points.append([0, 0, i, 0]) # column 1 at position (0, 0)
            #points.append([0, self.dimension_size - 1, i, 0]) # column 2 at position (0, dimension_size - 1)
            #points.append([self.dimension_size - 1, 0, i, 0]) # column 3 at position (dimension_size - 1, 0)
            #points.append([self.dimension_size - 1, self.dimension_size - 1, i, 0]) # column 4 at position (dimension_size - 1, dimension_size - 1)
            
            #points.append([0, i, self.dimension_size - 1, 0]) # beam 1 connecting column 1 and column 2
            #points.append([self.dimension_size - 1, i, self.dimension_size - 1, 0]) # beam 2 connecting column 3 and column 4
            #points.append([i, 0, self.dimension_size - 1, 0]) # beam 3 connecting column 1 and column 3
            #points.append([i, self.dimension_size - 1, self.dimension_size - 1, 0]) # beam 4 connecting column 2 and column 4
            pass
        
        # place 4 beam in 4 corder of grid at [x, y, z, 1]
        points.append([0, 0, 0, 1]) # column 1 at position (0, 0)
        points.append([0, self.dimension_size - 1, 0, 1]) # column 2 at position (0, dimension_size - 1)
        points.append([self.dimension_size - 1, 0, 0, 1]) # column 3 at position (dimension_size - 1, 0)
        points.append([self.dimension_size - 1, self.dimension_size - 1, 0, 1]) # column 1 at position (0, 0)
             
        
        possible_targets.append(points)                      
        for p in points:
            self.target[p[0], p[1], p[2], p[3]] = ScaffoldGridWorldEnv.BEAM_BLOCK  # -1 is block
        points.clear()
        # column
        points.append([0, 0, 0, 0]) # column 1 at position (0, 0)
        points.append([0, self.dimension_size - 1, 0, 0]) # column 2 at position (0, dimension_size - 1)
        points.append([self.dimension_size - 1, 0, 0, 0]) # column 3 at position (dimension_size - 1, 0)
        points.append([self.dimension_size - 1, self.dimension_size - 1, 0, 0]) # column 1 at position (0, 0)
        for p in points:
            self.target[p[0], p[1], p[2], p[3]] = ScaffoldGridWorldEnv.COL_BLOCK  # -1 is block


    def _tryMove(self, action, agent_id):
        #if (action in [0, 1, 2, 3, 4, 5]):
        new_pos = self.AgentsPos[agent_id].copy()
        if (action == 0):  # move forward
            new_pos[1] += 1
        elif (action == 1):  # move backward
            new_pos[1] -= 1
        elif (action == 2):  # move left
            new_pos[0] -= 1
        elif (action == 3):  # move right
            new_pos[0] += 1
        elif (action == 4):  # move up
            new_pos[2] += 1
        elif (action == 5):  # move down
            new_pos[2] -= 1
        """
        if (action in [0, 1, 2, 3]):
            if (self._canClimbDown(new_pos)):
                new_pos[2] -= 1
                return new_pos"""
        return new_pos


    def _isInScaffoldingDomain(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == ScaffoldGridWorldEnv.SCAFFOLD):
            return True
        return False 
    def _isInBlock(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == ScaffoldGridWorldEnv.COL_BLOCK or self.building_zone[pos[0], pos[1], pos[2]] == ScaffoldGridWorldEnv.BEAM_BLOCK):
            return True
        return False
    def _columnExist(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == ScaffoldGridWorldEnv.COL_BLOCK):
            return True
        return False

    def _beamExist(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == ScaffoldGridWorldEnv.BEAM_BLOCK):
            return True
        return False

    # position is not in any block or scaffolding
    def _is_empty(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == 0):
            return True
        return False
    # there is an agent in the position 
    def _thereIsAgent(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == 1):
            return True
        return False

    def _isOutOfBound(self, pos):
        if (pos[0] < 0 or pos[0] >= self.dimension_size or pos[1] < 0 or pos[1] >= self.dimension_size or pos[2] < 0 or pos[2] >= self.dimension_size):
            return True
        return False




    """
    return true if placement is valid

    arg:
        action: the action to be performed
        current_pos: the current position of the agent
        agent_id: the id of the agent
    
    """
    def _isScaffoldValid(self, current_pos):
        sneighbors = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1]]
                       # LEFT       RIGHT       BEHIND      FRONT     DOWN     UP         
        neighbour_direction = [
                [current_pos[0] + delta_x, current_pos[1] + delta_y, current_pos[2] + delta_z]
                for delta_x, delta_y, delta_z in sneighbors
            ]

        # Find if there is any supporting neighbour, or on the ground
        supporting_neighbour = current_pos[2] == 0
        if supporting_neighbour == False:
            for neighbour in neighbour_direction:
                if neighbour[0] < 0 or neighbour[0] >= self.dimension_size \
                        or neighbour[1] < 0 or neighbour[1] >= self.dimension_size \
                        or neighbour[2] < 0 or neighbour[2] >= self.dimension_size:
                    continue

                if self.building_zone[neighbour[0], neighbour[1], neighbour[2]] == ScaffoldGridWorldEnv.SCAFFOLD:
                    supporting_neighbour = True
                    break

        # If the space is already occupied
        duplicate_block = self.building_zone[current_pos[0], current_pos[1], current_pos[2]] != ScaffoldGridWorldEnv.EMPTY
        if supporting_neighbour and not duplicate_block:
            return True
        return False
    

    """
    if there exist supporting neighborblock around currentPos, 
    SUPORTING IF THERE ARE 4 SCAFFOLD 

    arg: 
       currentPos: 3-tuple (x, y, z), the location we want to place the block
    
    """ 
    def _check_support(self, currentPos, beam=False):   
        support = True
        scalffold_direction = [  
            [currentPos[0] + delta_x, currentPos[1] + delta_y, currentPos[2] + delta_z]
            for delta_x, delta_y, delta_z in [[-1, 0, -1], [1, 0, -1], [0, -1, -1], [0, 1, -1], [0, 0, -1]]
        ]                                       # LEFT       RIGHT       BEHIND      FRONT

        adjacent_direction = [
            [currentPos[0] + delta_x, currentPos[1] + delta_y, currentPos[2] + delta_z]
            for delta_x, delta_y, delta_z in [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
        ]                                       # LEFT       RIGHT       BEHIND      FRONT
        
        # Remove invalid directions
        for direction in scalffold_direction:
            if direction[0] < 0 or direction[0] >= self.dimension_size \
                    or direction[1] < 0 or direction[1] >= self.dimension_size \
                    or direction[2] < 0 or direction[2] >= self.dimension_size:
                scalffold_direction.remove(direction)
        for direction in adjacent_direction:
            if direction[0] < 0 or direction[0] >= self.dimension_size \
                    or direction[1] < 0 or direction[1] >= self.dimension_size \
                    or direction[2] < 0 or direction[2] >= self.dimension_size:
                adjacent_direction.remove(direction)
        
        for scaffold_dir in scalffold_direction:
            if self.building_zone[scaffold_dir[0], scaffold_dir[1], scaffold_dir[2]] == ScaffoldGridWorldEnv.EMPTY:
                support = False
                break
        if beam:
            if support:
                if self.building_zone[currentPos[0], currentPos[1], currentPos[2] - 1] == ScaffoldGridWorldEnv.COL_BLOCK:
                    support = True
                else:
                    support = False
                    for adjacent_dir in adjacent_direction:
                        if self.building_zone[adjacent_dir[0], adjacent_dir[1], adjacent_dir[2]] == ScaffoldGridWorldEnv.COL_BLOCK or \
                            self.building_zone[adjacent_dir[0], adjacent_dir[1], adjacent_dir[2]] == ScaffoldGridWorldEnv.BEAM_BLOCK:
                            support = True
                            break
        return support
        
        # Check if there's any supporting block



    # done and there is no more scaffold
    def _isDoneBuildingStructure(self):
        # check if col is finished
        col_done = self._check_finish_columns()
        beam_done = self._isBeamDone()
    
        scaffold_left = np.any(np.isin(self.building_zone[self.target == 0], ScaffoldGridWorldEnv.SCAFFOLD))
        if col_done and beam_done and not scaffold_left:
            return True
        return False
    # done structure but there is scafold left 
    def _isDoneBuildingStructureWithScafold(self):
        # check if col is finished
        col_done = self._check_finish_columns()
        beam_done = self._isBeamDone()
    
        #scaffold_left = np.any(np.isin(self.building_zone[self.target == 0], ScaffoldGridWorldEnv.SCAFFOLD))
        if col_done and beam_done: #and not scaffold_left:
            return True
        return False
        
    def _isBeamDone(self):
        check = np.isin(self.building_zone[self.target == ScaffoldGridWorldEnv.BEAM_BLOCK], ScaffoldGridWorldEnv.BEAM_BLOCK)
        if np.all(check):
            return True
        return False
    
    def _check_finish_columns(self):
        # 
        check = np.isin(self.building_zone[self.target == ScaffoldGridWorldEnv.COL_BLOCK], ScaffoldGridWorldEnv.COL_BLOCK)
        if np.all(check):
            return True
        return False
    

    def step(self, action_tuple):
        if (len(action_tuple) != 2):
            raise ValueError("action_tuple should be a tuple of 2 elements")

        action = action_tuple[0]
        agent_id = action_tuple[1]
        #self.mutex.acquire()  # get lock to enter critical section
        self.timestep_elapsed += 1
        """
        ACTION:
        0: forward, 1: backward, 2: left, 3: right                         [move]
        4: up, 5: down                                                     [move but only in the scaffolding domain]
        6: place scaffold at current position                              [place block]
        7: remove scaffold at current position                             [remove block]
        8-11: place block at the 4 adjacent position of the agent          [place block]
        
        
        REWARD STRUCTURE
        move: -0.5
        place scaffold: -0.5 if valid, -1 if invalid
        remove scaffold: -0.5 if valid, -1 if invalid
        place block: -0.5 if valid, -2.5 if invalid, +0.5 if valid and on the target
        
        TODO: Big positive when the structure is done. Small positive reward for each scaffold removed after the structure
        is finished. 
        
        """
        current_pos = self.AgentsPos[agent_id]

        if (action in [self.action_enum.FORWARD, 
                       self.action_enum.BACKWARD,
                       self.action_enum.RIGHT,
                       self.action_enum.LEFT,
                       self.action_enum.UP,
                       self.action_enum.DOWN]):  # move action
            R = -0.2
            terminated = False
            truncated = False
            is_valid = False
            new_pos = self._tryMove(action, agent_id)
            if (self._isValidMove(new_pos, action, current_pos)):
                #self.building_zone[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 0
                self.AgentsPos[agent_id][0] = new_pos[0]
                self.AgentsPos[agent_id][1] = new_pos[1]
                self.AgentsPos[agent_id][2] = new_pos[2]
                #self.building_zone[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1
            """else:  # case: invalid move, so agent just stay here
                pass"""
            obs = self.get_obs(agent_id)
            #self.mutex.release()
            #if not isValid: R = -1
            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True    
            return obs, R, terminated, truncated, {}

        elif (action == self.action_enum.PLACE_SCAFOLD):
            R = -0.3
            terminated = False
            truncated = False
            is_valid = False
            # agent can only place scaffold if there is nothing in current position
            if (self._isScaffoldValid(current_pos)):
                self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = ScaffoldGridWorldEnv.SCAFFOLD  # place scaffold block
                is_valid = True

            obs = self.get_obs(agent_id)
#            self.mutex.release()
            if not is_valid: R = -1
            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True

            #if is_valid and self.target[current_pos[0], current_pos[1], current_pos[2]] != ScaffoldGridWorldEnv.EMPTY:
            #    # case: if we placed a scafold on the target
            #    R = -2
            #else:
            #    # we place scafold on non target
            #    R = -0.75

            return obs, R, terminated, truncated, {}
            
        elif (action == self.action_enum.REMOVE_SCAFOLD):
            R = -0.3
            if self.finished_structure_with_scafold:
                R = 0.2
            terminated = False
            truncated = False
            is_valid = False
            # agent can only remove scaffold if there is a scaffold in current position and there is no scaffold above or agent above

            
            if (self._isInScaffoldingDomain(current_pos)):
                if (not self._isOutOfBound([current_pos[0], current_pos[1], current_pos[2] + 1]) and  (self._is_empty([current_pos[0], current_pos[1], current_pos[2] + 1]) or 
                                                                                                       self.building_zone[current_pos[0], current_pos[1], current_pos[2] + 1] == ScaffoldGridWorldEnv.BEAM_BLOCK)):
                    # case: remove scaffold is not on the top floor and there is no block above
                    self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = 0
                    is_valid = True
                elif (self._isOutOfBound([current_pos[0], current_pos[1], current_pos[2] + 1])):
                    # case: remove scaffold is on the top floor
                    self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = 0
                    is_valid = True
            else:  # case: invalid remove, so agent just stay here
                pass
                
            
            # return obs, reward, done, info
            obs = self.get_obs(agent_id)
            #self.mutex.release()
            if not is_valid: R = -1
            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True
                return obs, R, terminated, truncated, {}
            else:
                if self.finished_structure and is_valid and not np.any(np.isin(self.building_zone, ScaffoldGridWorldEnv.SCAFFOLD)):
                    R = 1
                    terminated = True
                return obs, R, terminated, truncated, {}
        elif action == self.action_enum.PLACE_COLUMN:  # place command
            R = -0.5
            terminated = False
            truncated = False
            is_valid = False
            
            # if there is already a block or a scaffold in the position
            if self.building_zone[current_pos[0], current_pos[1], current_pos[2]] == ScaffoldGridWorldEnv.SCAFFOLD or self._isInBlock(current_pos):
                is_valid = False
            
            # Check if there is proper support. Case 1, on the floor
            elif current_pos[2] == 0:
                is_valid = True
            # Case 2, on the scaffold and there is column block below
            elif self._check_support(current_pos) and self._columnExist((current_pos[0], current_pos[1], current_pos[2] - 1)):
                is_valid = True
                
            if is_valid:
                self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = ScaffoldGridWorldEnv.COL_BLOCK

                if  self.target[current_pos[0], current_pos[1], current_pos[2]] == self.building_zone[current_pos[0], current_pos[1], current_pos[2]]:
                    R = 0.9  # placing scafold column costs (-0.2 + -0.2) = -0.4. Placing column stack costs -1.2. (6*-0.2) = -1.2. so 0.9 + -1.2 > -0.4
                else:
                    R = -0.5
            else:
                R = -1
            obs = self.get_obs(agent_id)
            #self.mutex.release()
            # check if structure is complete
            if (is_valid and self._check_finish_columns() and not self.finished_structure_reward_used):  #  only do terminal check if we placed a block to save computation
                #terminated = True
                """if np.array_equal(self.building_zone, self.target):
                    terminated = True"""
                
                R = 1
                self.finished_structure_reward_used = True

            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True
                return obs, R, terminated, truncated, {}
            else:
                return obs, R, terminated, truncated, {}
        elif action == self.action_enum.PLACE_BEAM:
            R = -0.5
            terminated = False
            truncated = False
            is_valid = False
            # case: havent fnished column block yet
            if not self._check_finish_columns():
                return self.get_obs(agent_id), -1, False, self.timestep_elapsed > MAX_TIMESTEP, {}
            else:
                # if there is already a block or a scaffold in the position
                if self.building_zone[current_pos[0], current_pos[1], current_pos[2]] == ScaffoldGridWorldEnv.SCAFFOLD or self._isInBlock(current_pos):
                    is_valid = False
                
                # Check if there is proper support. Case 1, on the floor
                elif current_pos[2] == 0:
                    is_valid = True
                # Case 2, on the scaffold
                elif self._check_support(current_pos, beam=True):
                    is_valid = True
                    
                if is_valid:
                    self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = ScaffoldGridWorldEnv.BEAM_BLOCK
                    if  self.target[current_pos[0], current_pos[1], current_pos[2]] == self.building_zone[current_pos[0], current_pos[1], current_pos[2]]:
                        # good placement R = -1.5 + 1.25 = -0.25
                        R = 0.9
                    else:
                        R = -0.5
                else:
                    R = -1

                obs = self.get_obs(agent_id)
                #self.mutex.release()
                # check if structure is complete
                if (is_valid and self._isDoneBuildingStructureWithScafold() and not self.finished_structure_with_scafold):  #  only do terminal check if we placed a block to save computation
                    #terminated = True
                    R = 1
                    self.finished_structure_with_scafold = True
                """elif (is_valid and self._isDoneBuildingStructure()):  #  only do terminal check if we placed a block to save computation
                    #terminated = True
                    R = 1
                    self.finished_structure = True"""
                
                if self.timestep_elapsed > MAX_TIMESTEP:
                    truncated = True
                    return obs, R, terminated, truncated, {}
                else:
                    return obs, R, terminated, truncated, {}

        return

        
 
    

    """
    brute force this (N, N, N, K) space because I am not sure if we can do it in a more efficient way
    
    """
    def render(self):
        # acumulate all agents position

        mask = self.building_zone[..., 0] == ScaffoldGridWorldEnv.COL_BLOCK  # (8, 8, 8)

        print(mask.shape)
        print(mask)

        fig = plt.figure()
        colors = np.empty(mask.shape, dtype=object)
        colors[mask] = 'blue'

        #agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        #agent_pos_grid[self.AgentsPos[0][0], self.AgentsPos[0][1], self.AgentsPos[0][2]] = 1

        ## prepare some coordinates
        #col_cube = self.building_zone == ScaffoldGridWorldEnv.COL_BLOCK
        #beam_cube = self.building_zone == ScaffoldGridWorldEnv.BEAM_BLOCK
        #scaffold_cube = self.building_zone == ScaffoldGridWorldEnv.SCAFFOLD
        #agent_position_cube = agent_pos_grid == 1

        #fig = plt.figure()

        #building_zone_render = col_cube | agent_position_cube | beam_cube | scaffold_cube
        ## set the colors of each object
        #colors = np.empty(building_zone_render.shape, dtype=object)
        #colors[col_cube] = '#7A88CCC0'
        #colors[agent_position_cube] = '#FFD65DC0'
        #colors[beam_cube] = '#FF5733C0'
        #colors[scaffold_cube] = 'pink'
        ## print(colors)

        #ax = fig.add_subplot(1, 2, 1, projection='3d')
        #ax.voxels(building_zone_render, facecolors=colors, edgecolor='k')

        #col_cube = self.target == ScaffoldGridWorldEnv.COL_BLOCK
        #beam_cube = self.target == ScaffoldGridWorldEnv.BEAM_BLOCK
        #target_render = col_cube | beam_cube
        ## set the colors of each object
        #colors = np.empty(target_render.shape, dtype=object)
        #colors[col_cube] = '#7A88CCC0'
        #colors[beam_cube] = '#FF5733C0'
        #ax = fig.add_subplot(1, 2, 2, projection='3d')
        #ax.voxels(target_render, facecolors=colors, edgecolor='k')

        #plt.show()

  

    
    def close(self):
        pass


if __name__ == '__main__':
    env = ScaffoldGridWorldEnv(8, "path", mixture_capacity=2, num_agents=1, debug=False)    
    env.render()



