"""Simple grid world environment for testing PWMK components."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import gymnasium as gym


class SimpleGridWorld(gym.Env):
    """
    Simple 2D grid world for testing multi-agent Theory of Mind.
    
    Features:
    - Multiple agents with partial observability
    - Objects to collect (keys, treasures)
    - Agent positions visible to others within view radius
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        num_agents: int = 2,
        view_radius: int = 2,
        num_treasures: int = 1,
        num_keys: int = 1
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.view_radius = view_radius
        self.num_treasures = num_treasures
        self.num_keys = num_keys
        
        # Action space: 0=up, 1=right, 2=down, 3=left, 4=pickup
        self.action_space = gym.spaces.Discrete(5)
        
        # Observation space: local grid view + agent features
        obs_size = (2 * view_radius + 1) ** 2 + 4  # grid + [x, y, has_key, has_treasure]
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[List[np.ndarray], Dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize grid (0=empty, 1=wall, 2=key, 3=treasure, 4=agent)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Add walls around border
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # Agent positions and states
        self.agent_positions = []
        self.agent_inventories = []
        
        # Place agents randomly
        for i in range(self.num_agents):
            pos = self._get_random_empty_position()
            self.agent_positions.append(pos)
            self.agent_inventories.append({"has_key": False, "has_treasure": False})
            self.grid[pos[0], pos[1]] = 4
        
        # Place objects
        self.key_positions = []
        self.treasure_positions = []
        
        for _ in range(self.num_keys):
            pos = self._get_random_empty_position()
            self.key_positions.append(pos)
            self.grid[pos[0], pos[1]] = 2
            
        for _ in range(self.num_treasures):
            pos = self._get_random_empty_position()
            self.treasure_positions.append(pos)
            self.grid[pos[0], pos[1]] = 3
        
        self.step_count = 0
        
        # Return observations for all agents
        observations = [self._get_observation(i) for i in range(self.num_agents)]
        info = {"step": self.step_count}
        
        return observations, info
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """Execute actions for all agents."""
        rewards = [0.0] * self.num_agents
        
        # Execute actions
        for agent_id, action in enumerate(actions):
            reward = self._execute_action(agent_id, action)
            rewards[agent_id] = reward
        
        # Check termination (all treasures collected)
        treasures_collected = sum(
            inv["has_treasure"] for inv in self.agent_inventories
        )
        terminated = treasures_collected >= self.num_treasures
        
        self.step_count += 1
        truncated = self.step_count >= 100  # Maximum episode length
        
        # Get observations
        observations = [self._get_observation(i) for i in range(self.num_agents)]
        
        info = {
            "step": self.step_count,
            "agent_positions": self.agent_positions.copy(),
            "inventories": self.agent_inventories.copy()
        }
        
        return observations, rewards, terminated, truncated, info
    
    def _execute_action(self, agent_id: int, action: int) -> float:
        """Execute a single agent's action."""
        reward = -0.01  # Small step penalty
        
        current_pos = self.agent_positions[agent_id]
        
        if action < 4:  # Movement actions
            # Calculate new position
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
            dx, dy = moves[action]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check bounds and collisions
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size and
                self.grid[new_pos[0], new_pos[1]] != 1):  # Not a wall
                
                # Check for other agents
                if new_pos not in self.agent_positions:
                    # Clear old position
                    self.grid[current_pos[0], current_pos[1]] = 0
                    
                    # Update position
                    self.agent_positions[agent_id] = new_pos
                    
                    # Set new position (but preserve objects underneath)
                    if self.grid[new_pos[0], new_pos[1]] == 0:
                        self.grid[new_pos[0], new_pos[1]] = 4
                    
        elif action == 4:  # Pickup action
            pos = current_pos
            
            # Check for key
            if pos in self.key_positions:
                self.agent_inventories[agent_id]["has_key"] = True
                self.key_positions.remove(pos)
                self.grid[pos[0], pos[1]] = 4  # Just agent
                reward += 1.0
                
            # Check for treasure (need key first)
            elif pos in self.treasure_positions and self.agent_inventories[agent_id]["has_key"]:
                self.agent_inventories[agent_id]["has_treasure"] = True
                self.treasure_positions.remove(pos)
                self.grid[pos[0], pos[1]] = 4  # Just agent
                reward += 10.0
        
        return reward
    
    def _get_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent."""
        pos = self.agent_positions[agent_id]
        
        # Extract local grid view
        local_view = []
        for dx in range(-self.view_radius, self.view_radius + 1):
            for dy in range(-self.view_radius, self.view_radius + 1):
                x, y = pos[0] + dx, pos[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    local_view.append(self.grid[x, y])
                else:
                    local_view.append(1)  # Wall outside bounds
        
        # Add agent features
        inventory = self.agent_inventories[agent_id]
        features = [
            pos[0] / self.grid_size,  # Normalized x position
            pos[1] / self.grid_size,  # Normalized y position  
            1.0 if inventory["has_key"] else 0.0,
            1.0 if inventory["has_treasure"] else 0.0
        ]
        
        observation = np.array(local_view + features, dtype=np.float32)
        return observation
    
    def _get_random_empty_position(self) -> Tuple[int, int]:
        """Get a random empty position on the grid."""
        while True:
            x = np.random.randint(1, self.grid_size - 1)
            y = np.random.randint(1, self.grid_size - 1)
            if self.grid[x, y] == 0:
                return (x, y)
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            symbols = {0: ".", 1: "#", 2: "K", 3: "T", 4: "A"}
            for row in self.grid:
                print("".join(symbols[cell] for cell in row))
            print(f"Step: {self.step_count}")
            print(f"Inventories: {self.agent_inventories}")
            print()
        elif mode == "rgb_array":
            # Simple color mapping for visualization
            rgb = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            rgb[self.grid == 1] = [100, 100, 100]  # Walls: gray
            rgb[self.grid == 2] = [255, 255, 0]    # Keys: yellow
            rgb[self.grid == 3] = [0, 255, 0]      # Treasures: green  
            rgb[self.grid == 4] = [255, 0, 0]      # Agents: red
            return rgb
        
        return None
    
    def get_belief_ground_truth(self) -> Dict[str, Dict[str, Any]]:
        """Get ground truth beliefs for all agents."""
        beliefs = {}
        
        for agent_id in range(self.num_agents):
            agent_beliefs = {}
            pos = self.agent_positions[agent_id]
            
            # What this agent can see
            visible_agents = []
            for other_id in range(self.num_agents):
                if other_id != agent_id:
                    other_pos = self.agent_positions[other_id]
                    distance = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                    if distance <= self.view_radius:
                        visible_agents.append(other_id)
            
            agent_beliefs["visible_agents"] = visible_agents
            agent_beliefs["own_inventory"] = self.agent_inventories[agent_id].copy()
            agent_beliefs["position"] = pos
            
            # Simple belief about others (what they might know)
            for other_id in visible_agents:
                other_pos = self.agent_positions[other_id]
                agent_beliefs[f"believes_agent_{other_id}_at"] = other_pos
                
            beliefs[f"agent_{agent_id}"] = agent_beliefs
        
        return beliefs