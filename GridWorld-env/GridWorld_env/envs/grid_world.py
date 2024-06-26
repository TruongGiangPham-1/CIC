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

        self.AgentsPos = np.zeros((self.num_agents, 3), dtype=int)
        #self.reset()

        self._init_target()
        
        
        if debug:
            self._placeAgentInBuildingZone()

    
    # in multiagent, make sure only one agent is allowed to call this function(meta agent for example)
    def reset(self, seed=None, options=None):
        #self.mutex.acquire()  # get lock to enter critical section
        self.building_zone.fill(0)
        self.agent_pos_grid.fill(0)
        self.finished_structure = False
        self.finished_structure_reward_used = False
        self.finished_structure_with_scafold = False

        self.AgentsPos = np.zeros((self.num_agents, 3), dtype=int)

        random_start_pos = np.zeros(3, dtype=int)
        #self._init_target()
        self.record_sequence = []

        self.timestep_elapsed = 0
        self._init_obs()

        random_start_pos = np.zeros(3, dtype=int)
        for i in range(self.num_agents):
            random_start_pos[0] = np.random.randint(0, self.dimension_size)
            random_start_pos[1] = np.random.randint(0, self.dimension_size)
            random_start_pos[2] = 0
            self.AgentsPos[i] = random_start_pos
        np.copyto(self.obs[2], random.choice(self.all_targets))

        # change all instance of 1 to 0.5 in self.obs
        #self.obs[2][self.obs[2] == 1] = ScaffoldGridWorldEnv.COL_BLOCK
        ## change all instance of 2 to 1 in self.obs
        #self.obs[2][self.obs[2] == 2] = ScaffoldGridWorldEnv.BEAM_BLOCK

        #obs = self.get_obs(0)
        return self.get_obs(), {}

    # return (3 x N x N x N) tensor for now, 
    # TODO: implement
    def get_obs(self, agent_id=0):
        # clear agent_pos_grid
        self.agent_pos_grid.fill(0)
        #agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        #agent_pos_grid[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1
        self.agent_pos_grid[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1

        #return np.stack((self.building_zone, agent_pos_grid, self.target), axis=0)

#        return self.obs
        return {
            'agent_position': self.agent_pos_grid,
            'building_zone': self.building_zone,
            'target': self.target
        }
        
    def _get_info(self):
        pass

    # TODO : implement
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
        self.AgentsPos[0] = [0, 0, 0]
        return

    def _init_target(self):
        self.target = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size, self.mixture_capacity), dtype=float)
        
        # Hardcoding 4 columns, and 4 beams across the 4 columns. TO BE CHANGED TO BE MORE DYNAMIC AND USEABLE
        possible_targets = []
        
        col_points = []
        beam_points = []
        pipe_points = []
        electrical_points = []
        for i in range(4):  # place all columns:
            col_points.append([2, 2, i, 0]) # column 1 at position (2, 2)
            col_points.append([2, 4, i, 0]) # column 2 at position (2, 4
            col_points.append([5, 2, i, 0]) # column 3 at position (5, 2)
            col_points.append([5, 4, i, 0]) # column 4 at position (5, 4)

        for p in col_points:
            self.target[p[0], p[1], p[2], p[3]] = ScaffoldGridWorldEnv.COL_BLOCK  # -1 is block
        col_points.clear()

        # add beams on the 1st and 3rd floor between columns
        for i in [0, 2]:
            beam_points.append([2, 2, i, 1]) # column 1 at position (2, 2)
            beam_points.append([2, 4, i, 1]) # column 2 at position (2, 4
            beam_points.append([5, 2, i, 1]) # column 3 at position (5, 2)
            beam_points.append([5, 4, i, 1]) # column 4 at position (5, 4)

            # add beams on the floor between columns
            beam_points.append([3, 2, i, 0]) # column 1 at position (2, 2)
            beam_points.append([3, 4, i, 0]) # column 2 at position (2, 4
            beam_points.append([4, 2, i, 0]) # column 3 at position (5, 2)
            beam_points.append([4, 4, i, 0]) # column 4 at position (5, 4)
            beam_points.append([2, 3, i, 0]) # column 4 at position (5, 4)
            beam_points.append([5, 3, i, 0]) # column 4 at position (5, 4)

        for p in beam_points:
            self.target[p[0], p[1], p[2], p[3]] = ScaffoldGridWorldEnv.BEAM_BLOCK
        # add pipe work
        for i in [1]:  # in the 2nd floor
            pipe_points.append([2, 3, i, 0])
            pipe_points.append([3, 3, i, 0])
            pipe_points.append([4, 3, i, 0])
            pipe_points.append([5, 3, i, 0])
        for p in pipe_points:
            self.target[p[0], p[1], p[2], p[3]] = ScaffoldGridWorldEnv.PIPE_BLOCK

        # add electrical work
        for i in [1, 3]:  # in the 2nd and 4th floor
            if i == 1:
                electrical_points.append([2, 3, i, 1])
                electrical_points.append([3, 3, i, 1])
                electrical_points.append([4, 3, i, 1])
                electrical_points.append([5, 3, i, 1])
            else:
                electrical_points.append([2, 3, i, 0])
                electrical_points.append([3, 3, i, 0])
                electrical_points.append([4, 3, i, 0])
                electrical_points.append([5, 3, i, 0])
        for p in electrical_points:
            self.target[p[0], p[1], p[2], p[3]] = ScaffoldGridWorldEnv.ELECTRICAL_BLOCK



        # place 4 beam in 4 corner of grid at [x, y, z, 1]
    


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


    def scaffold_exist(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2], 0] == ScaffoldGridWorldEnv.SCAFFOLD):
            return True
        return False 
    # TODO: outdated
    def _isInBlock(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == ScaffoldGridWorldEnv.COL_BLOCK or self.building_zone[pos[0], pos[1], pos[2]] == ScaffoldGridWorldEnv.BEAM_BLOCK):
            return True
        return False
    # TODO: outdated
    def _columnExist(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2], 0] == ScaffoldGridWorldEnv.COL_BLOCK):
            return True
        return False


    # TODO: outdated
    def _beamExist(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2], 0] == ScaffoldGridWorldEnv.BEAM_BLOCK):
            return True
        return False

    # position is not in any block or scaffolding
    def _is_empty(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2], 0] == 0):
            return True
        return False
    # when all mixture capacity is exhausted at (X, y, z), we can't place any more block
    def _is_full(self, pos):
        for i in range(self.mixture_capacity):
            if (self.building_zone[pos[0], pos[1], pos[2], i] == 0):
                return False
        return True


    # TODO: outdated
    def _isOutOfBound(self, pos):
        if (pos[0] < 0 or pos[0] >= self.dimension_size or pos[1] < 0 or pos[1] >= self.dimension_size or pos[2] < 0 or pos[2] >= self.dimension_size):
            return True
        return False


    """
    Scaffold can only be placed if it is on the ground, or it is clamped to neighboring scaffold
    
    """
    def _isScaffoldValid(self, current_pos):
        sneighbors = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1]]
                       # LEFT       RIGHT       BEHIND      FRONT     DOWN     UP         
        neighbour_direction = [
                [current_pos[0] + delta_x, current_pos[1] + delta_y, current_pos[2] + delta_z]
                for delta_x, delta_y, delta_z in sneighbors
            ]
        # Find if there is any neighbour clamping(clamping in any direction), or on the ground
        supporting_neighbour = current_pos[2] == 0
        if supporting_neighbour == False:
            for neighbour in neighbour_direction:
                if self._isOutOfBound(neighbour):
                    continue
                if self.scaffold_exist(neighbour):
                    supporting_neighbour = True
                    break
        # If the space is already occupied

        if supporting_neighbour:
            return True
        return False
    
    """
    We can remove scaffold if:
    1. There is a scaffold in the current position
    2. there is no column block above the scaffold since only column block need verticacl stack and we cant make floating column block.
    3. we are the top of the grid
    
    """
    def _check_remove_scaffold(self, current_pos):
        above_coordinate = [current_pos[0], current_pos[1], current_pos[2] + 1]
        if self.scaffold_exist(current_pos):
            col_block_above = not self._isOutOfBound(above_coordinate) and self._columnExist(above_coordinate)
            empty_above = not self._isOutOfBound(above_coordinate) and self._is_empty(above_coordinate)
            if not col_block_above or empty_above:
                return True
            if self._isOutOfBound(above_coordinate):  # case: we are at the top of the grid
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
            if self._isOutOfBound(direction):
                scalffold_direction.remove(direction)
        for direction in adjacent_direction:
                adjacent_direction.remove(direction)
        
        # we doing one scaffold system. If there is any scaffold in the direction, we consider it as supporting
        for scaffold_dir in scalffold_direction:
            if self.scaffold_exist(scaffold_dir):
                support = True
                break
        # TODO: there are electrical and pipe block now. so need to check implementation
        if beam:
            # for beam it is supported by column block below, or clamp to any blocks
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
    
    

    

    """
    Given pos = (x, y, z) add the block of block_type to the building zone
    
    """
    def _add_block(self, pos, block_type):
        x, y, z = pos                
        # Find the index of the leftmost non-zero element at (x, y, z)
        mixture_index_available = np.argmax(self.building_zone[x, y, z] != 0)

        success = False
        if (self._is_empty([x, y, z])):
            self.building_zone[x, y, z, 0] = block_type  # trivially add the block
            success = True
        # case we wan to add a beam block on top of a column block
        elif (self.building_zone[x, y, z, 0] == ScaffoldGridWorldEnv.COL_BLOCK and block_type == ScaffoldGridWorldEnv.BEAM_BLOCK):
            self.building_zone[x, y, z, 1] = ScaffoldGridWorldEnv.BEAM_BLOCK 
            success = True
        # case we wan to add a electrical block on top of a pipe block
        elif (self.building_zone[x, y, z, 0] == ScaffoldGridWorldEnv.PIPE_BLOCK and block_type == ScaffoldGridWorldEnv.ELECTRICAL_BLOCK):
            self.building_zone[x, y, z, 1] = ScaffoldGridWorldEnv.ELECTRICAL_BLOCK
            success = True
        return success
    """
    given a position in the building zone, check if it matches the target zone
    
    """ 
    def _correct_placement(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == self.target[pos[0], pos[1], pos[2]]):
            return True
        return False


    def _isValidMove(self, new_pos, action, current_pos):
        if (self._isOutOfBound(new_pos)):
            return False
        return True

    def step(self, action, agent_id=0):
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
        current_pos = self.AgentsPos[0]

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
            if (self._isScaffoldValid(current_pos) and self._is_empty(current_pos)):
                self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = ScaffoldGridWorldEnv.SCAFFOLD  # place scaffold block
                success = self._add_block(current_pos, ScaffoldGridWorldEnv.SCAFFOLD)
                assert(success, "Failed to add block")
                is_valid = True
            obs = self.get_obs(agent_id)
#            self.mutex.release()
            if not is_valid: R = -1
            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True
            return obs, R, terminated, truncated, {}
        elif (action == self.action_enum.REMOVE_SCAFOLD):
            R = -0.3
            if self.finished_structure_with_scafold:
                R = 0.2
            terminated = False
            truncated = False
            is_valid = False
            # agent can only remove scaffold if there is a scaffold in current position and there is no scaffold above or agent above

            is_valid = self._check_remove_scaffold(current_pos)

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
            
            # if there is already block occupied
            if self._is_full(current_pos):
                is_valid = False
            
            # Check if there is proper support. Case 1, on the floor
            elif current_pos[2] == 0:
                is_valid = True
            # Case 2, not on the floor && on the scaffold and there is column block below
            elif self._check_support(current_pos) and self._columnExist((current_pos[0], current_pos[1], current_pos[2] - 1)):
                is_valid = True
            
            # try adding a block
            block_add_success = self._add_block(current_pos, ScaffoldGridWorldEnv.COL_BLOCK)
            if is_valid and block_add_success:
                if self._correct_placement(current_pos): 
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
        fig = plt.figure()



        ### TARGET RENDER
        # all the mask of the blocks in (x, y, z, 0)
        col_mask_0 = self.target[..., 0] == ScaffoldGridWorldEnv.COL_BLOCK  # (8, 8, 8)
        beam_mask_0 = self.target[..., 0] == ScaffoldGridWorldEnv.BEAM_BLOCK  # (8, 8, 8)
        electrical_mask_0 = self.target[..., 0] == ScaffoldGridWorldEnv.ELECTRICAL_BLOCK  # (8, 8, 8)
        pipe_mask_0 = self.target[..., 0] == ScaffoldGridWorldEnv.PIPE_BLOCK

        # all the mask of the blocks in (x, y, z, 1). only beam or electrial can be mixed on top of column and pipe respectively
        beam_mask_1 = self.target[..., 1] == ScaffoldGridWorldEnv.BEAM_BLOCK  # (8, 8, 8)
        electrical_mask_1 = self.target[..., 1] == ScaffoldGridWorldEnv.ELECTRICAL_BLOCK  # (8, 8, 8)

        beam_colum_mask = col_mask_0 & beam_mask_1  # beam and column can be mixed in same (x, y, z)
        electrical_pipe_mask = pipe_mask_0 & electrical_mask_1  # electrical and pipe can be mixed in same (x, y, z)

        target_render = col_mask_0 | beam_mask_0 | electrical_mask_0 | pipe_mask_0 | beam_colum_mask | electrical_pipe_mask
        # prepare mixture mask
        # set the colors of each object
        colors = np.empty(target_render.shape, dtype=object)
        colors[col_mask_0] = '#7aa6cc'  # blue
        colors[beam_mask_0] = '#FF5733C0'  # red
        colors[electrical_mask_0] = '#ccc87a'  # yellow
        colors[pipe_mask_0] = '#7accbc'   # turoise
        colors[beam_colum_mask] = '#967acc'  # purple = red + blue
        colors[electrical_pipe_mask] = '#b3cc7a'  # yellow green

        #print(colors)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.voxels(target_render, facecolors=colors, edgecolor='k')


        # adding legends (used AI assisted tool to generate this legend code)
        legend_labels = ['column', 'beam', 'electrical', 'pipe', 'beam_column', 'electrical_pipe']
        legend_colors = ['#7aa6cc', '#FF5733C0', '#ccc87a', '#7accbc', '#967acc', '#b3cc7a']
        legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, legend_colors)]
        ax.legend(handles=legend_patches, loc='upper right',  bbox_to_anchor=(1.2, 1.4))





        ### BUILDING ZONE RENDER
        agent_mask  = self.agent_pos_grid == 1
        col_mask_0 = self.building_zone[..., 0] == ScaffoldGridWorldEnv.COL_BLOCK  # (8, 8, 8)
        beam_mask_0 = self.building_zone[..., 0] == ScaffoldGridWorldEnv.BEAM_BLOCK  # (8, 8, 8)
        electrical_mask_0 = self.building_zone[..., 0] == ScaffoldGridWorldEnv.ELECTRICAL_BLOCK  # (8, 8, 8)
        pipe_mask_0 = self.building_zone[..., 0] == ScaffoldGridWorldEnv.PIPE_BLOCK

        beam_mask_1 = self.building_zone[..., 1] == ScaffoldGridWorldEnv.BEAM_BLOCK  # (8, 8, 8)
        electrical_mask_1 = self.building_zone[..., 1] == ScaffoldGridWorldEnv.ELECTRICAL_BLOCK  # (8, 8, 8)

        beam_colum_mask = col_mask_0 & beam_mask_1  # beam and column can be mixed in same (x, y, z)
        electrical_pipe_mask = pipe_mask_0 & electrical_mask_1  # electrical and pipe can be mixed in same (x, y, z)

        scaffold_mask = self.building_zone[..., 0] == ScaffoldGridWorldEnv.SCAFFOLD

        building_render = col_mask_0 | beam_mask_0 | electrical_mask_0 | pipe_mask_0 | beam_colum_mask | electrical_pipe_mask | scaffold_mask | agent_mask

        colors[scaffold_mask] = 'pink'
        colors[agent_mask] = 'black'

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.voxels(building_render, facecolors=colors, edgecolor='k')
        plt.show()

    
    def close(self):
        pass






if __name__ == '__main__':
    env = ScaffoldGridWorldEnv(8, "path", mixture_capacity=2, num_agents=1, debug=False)    
    #print(env.target)
    env.step(Action.FORWARD)
    env.render()



