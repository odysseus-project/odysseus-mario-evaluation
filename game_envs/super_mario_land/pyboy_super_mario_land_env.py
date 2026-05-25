from datetime import datetime
import os
from pyboy import PyBoy
from gymnasium import Env, spaces
from typing import Tuple, Any, Dict, List
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from game_envs.base_env import BaseEnv

class PyBoySuperMarioLandEnv(BaseEnv):
    metadata = {"render_modes": []}
    BUTTONS = ['a', 'b', 'left', 'right', 'up', 'down', 'noop']
    MATRIX_SHAPE = (160, 144)
    MAX_STEP = 100000
    
    # Tile descriptions mapping based on compressed_list from game wrapper
    TILE_DESCRIPTIONS = {
        0: "empty",
        # 1: "mario/daisy",
        1: "mario",
        2: "plane (mario is inside)",
        3: "submarine (mario is inside)",
        4: "shoots",
        5: "coin",
        6: "mushroom",
        7: "heart",
        8: "star",
        9: "lever",
        10: "neutral_blocks (solid ground - you can safely stand on it or, if it is obstructing your path, jump over it)",
        11: "moving_blocks",
        12: "pushable_blocks",
        13: "question_block (can contain items - jump to hit)",
        14: "pipes (obstacles - may need to jump over)",
        15: "goomba (enemy - jump on or avoid)",
        16: "koopa (enemy - jump on or avoid)",
        17: "plant (enemy - avoid)",
        18: "moth (enemy - jump on or avoid)",
        19: "flying_moth (enemy - avoid)",
        20: "sphinx (enemy - jump on or avoid)",
        21: "big_sphinx (enemy - avoid)",
        22: "fist (enemy - avoid)",
        23: "bill (enemy - avoid)",
        24: "projectiles (enemy attack - avoid)",
        25: "shell (enemy - avoid)",
        26: "explosion",
        27: "spike (obstacle - avoid)",
    }
    
    def __init__(
        self, 
        rom_path: str, 
        init_state: str = None, 
        world: int = 1, 
        level: int = 1, 
        headless: bool = True, 
        sound: bool = False, 
        with_text_info: bool = True,
        with_game_area: bool = False,
        # resize_ratio: int = 1,
        use_old_tick: bool = False,
        use_simple_reward: bool = False,
        epsilon: float = 0.0,
        cgb: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.headless = headless
        self.with_text_info = with_text_info
        self.with_game_area = with_game_area
        # self.resize_ratio = resize_ratio
        self.use_old_tick = use_old_tick
        self.use_simple_reward = use_simple_reward
        self.epsilon = float(epsilon)
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if headless:
            self.pyboy = PyBoy(
                rom_path,
                window="null",
                cgb=cgb,
            )
        else:
            self.pyboy = PyBoy(
                rom_path,
                cgb=cgb,
            )
        self.pyboy.set_emulation_speed(0)
        """Initialize the emulator."""
        self.world = world
        self.level = level
        self.mario = self.pyboy.game_wrapper
        self.mario.game_area_mapping(self.mario.mapping_compressed, 0)
        self.mario.start_game(world_level = (world, level))
        if init_state:
            self.init_state = init_state
            with open(init_state, "rb") as f:
                self.pyboy.load_state(f)
        else:
            self.init_state = None
        self.tick(1, True)
        self.step_count = 0
        self._button_is_pressed = {button: False for button in self.BUTTONS}
    
    def initial_obs(self) -> Dict:
        return self.get_observation()
    
    def step(self, action: List[str]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = [str(btn).lower() for btn in action]
        if random.random() < self.epsilon:
            action = [random.choice(self.BUTTONS)]
        invalid_buttons = []
        if not all(btn in self.BUTTONS for btn in action):
            invalid_buttons = [btn for btn in action if btn not in self.BUTTONS]
            action = [btn for btn in action if btn in self.BUTTONS]
        self.press_buttons(action)
        
        if len(action) > 0:
            self.step_count += 1
        # Here, self.pyboy.memory[0xFFA6] accesses the byte in Game Boy memory
        # at address 0xFFA6, which is used by Super Mario Land to track the
        # 'death animation' timer. When this value exceeds 0x80 (i.e., 128 in decimal),
        # it indicates that Mario is dead or the death sequence is active, so the episode should terminate.
        done = (
            self.step_count >= self.MAX_STEP
            or self.mario.game_over()
            or self.pyboy.memory[0xFFA6] > 0x80  # 0xFFA6 > 0x80 means Mario is dead or in death animation
            or (self.mario.lives_left < 2 if self.init_lives_left == 2 else False)
        )
        reward = self.compute_reward()

        obs = self.get_observation()
        self.set_prev_state()
        return obs, reward, done, False, {"invalid_buttons": invalid_buttons, "level_progress": float(self.mario.level_progress), "game_score": float(self.mario.score)}
    
    def compute_reward(self):
        if self.pyboy.memory[0xFFA6] > 0x80:
            lives_diff = -1
        else:
            # lives_diff = 0
            lives_diff = self.mario.lives_left - self.prev_lives_left
        score_diff = self.mario.score - self.prev_score
        level_progress_diff = (self.mario.level_progress - self.prev_level_progress) * float(lives_diff == 0)
        if self.use_simple_reward:
            reward = 10.0 * level_progress_diff
        else:
            reward = 1000.0 * lives_diff + score_diff + 10.0 * level_progress_diff
        return reward
    
    def set_prev_state(self):
        self.prev_world = self.mario.world
        self.prev_coins = self.mario.coins
        self.prev_lives_left = self.mario.lives_left
        self.prev_score = self.mario.score
        self.prev_time_left = self.mario.time_left
        self.prev_level_progress = self.mario.level_progress
    
    def reset(self, seed) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.mario.reset_game()
        if self.init_state:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
        self.tick(1, True)
        self.step_count = 0
        self._button_is_pressed = {button: False for button in self.BUTTONS}
        self.set_prev_state()
        self.init_lives_left = self.mario.lives_left
        obs = self.get_observation()
        return obs, {}
    
    def close(self):
        self.pyboy.stop(save=False)
    
    # Helper functions
    def get_screenshot(self):
        """Get the current screenshot."""
        return Image.fromarray(self.pyboy.screen.ndarray)
    
    def get_state_from_memory(self) -> str:
        """
        Reads the game state from memory and returns a string representation of it.
        """
        info = ""
        if self.with_text_info:
            info += (
                f"Super Mario Land: World {'-'.join([str(i) for i in self.mario.world])}\n"
                f"Coins: {self.mario.coins}\n"
                f"lives_left: {self.mario.lives_left}\n"
                f"Score: {self.mario.score}\n"
                f"Time left: {self.mario.time_left}\n"
                f"Level progress: {self.mario.level_progress}\n"
            )
        if self.with_game_area:
            info += f"\n{self.get_game_area_summary()}"
        return info
    
    def get_observation(self):
        # Get image observation (screenshot already creates a new Image object)
        obs_img = self.get_screenshot()
        # obs_img = self.resize_image(obs_img, self.resize_ratio)
        # Read state memory (string is immutable, no need for deepcopy)
        obs_memo = self.get_state_from_memory()

        # No deepcopy needed - obs_img is already a new object, obs_memo is a string
        obs = {"image": obs_img, "state": obs_memo}
        return obs
    
    def tick(self, *args, **kwargs):
        """Advance the emulator by the specified number of frames."""
        self.pyboy.tick(*args, **kwargs)
    
    def get_game_area_summary(self) -> str:
        """
        Get a human-readable summary of the game area tiles.
        Only shows legend entries for tiles that are actually present in the current game area.
        
        Returns:
            str: Formatted string representation of the game area with descriptions
        """
        tiles = self.mario.game_area()
        
        # Get unique tile values present in the current game area
        unique_tiles = set(tiles.flatten())
        
        # Build a formatted string
        lines = []
        lines.append("Game Area Tiles (16 rows x 20 columns):")
        lines.append("=" * 60)

        adjust = 4
        
        tiles_header = (
            " " * 4 + "".join([f"{i: >4}" for i in range(tiles.shape[1])]) + "\n" + "_" * (adjust * tiles.shape[1] + 4)
        )

        tiles_str = "\n".join(
            [
                (f"{i: <3}|" + "".join([str(tile).rjust(adjust) for tile in line])).strip()
                for i, line in enumerate(tiles)
            ]
        )

        lines.append(tiles_header + "\n" + tiles_str)
        
        lines.append("\n" + "=" * 60)
        lines.append("Legend (tiles present in current area):")
        for idx in sorted(unique_tiles):
            desc = self.TILE_DESCRIPTIONS.get(idx, "unknown")
            lines.append(f"  {idx:2d}: {desc}")

        lines.append(
            "Notes:\n"
            "- The game screen is divided into a grid of 16 rows and 20 columns (16x20).\n"
            "- Row 0 is at the top of the screen; Row 15 is at the bottom.\n"
            "- Column 0 is the leftmost column; Column 19 is the rightmost.\n"
            "- Mario occupies a 2x2 cell area (spans 2 rows vertically and 2 columns horizontally)."
        )
        
        return "\n".join(lines)
    
    # def press_buttons(self, buttons):
    #     """Press a sequence of buttons on the Game Boy.
        
    #     Args:
    #         buttons (list[str]): List of buttons to press in sequence
    #         wait (bool): Whether to wait after each button press
            
    #     Returns:
    #         str: Result of the button presses
    #     """
    #     for button, status in self._button_is_pressed.items():
    #         if status and button not in buttons:
    #             self.pyboy.button_release(button)
    #             self._button_is_pressed[button] = False
    #     for button in buttons:
    #         self.pyboy.button_press(button)
    #         self._button_is_pressed[button] = True
    #     self.tick(10) # Immediate for real time games

    def press_buttons(self, buttons):
        """Press a sequence of buttons on the Game Boy.
        
        Args:
            buttons (list[str]): List of buttons to press in sequence
            wait (bool): Whether to wait after each button press
            
        Returns:
            str: Result of the button presses
        """
        for button in buttons:
            if button == "noop":
                continue
            self.pyboy.button_press(button)
        # self.tick(10, False) # Immediate for real time games
        if self.use_old_tick:
            self.tick(10, False)
        else:
            if "a" in buttons:
                self.tick(15, False)
            else:
                self.tick(5, False)
        for button in buttons:
            if button == "noop":
                continue
            self.pyboy.button_release(button)
        self.tick(1, True) # Immediate for real time games
