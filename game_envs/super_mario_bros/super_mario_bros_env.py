import ast
import re
import logging
import gym_super_mario_bros
#from nes_py.wrappers import JoypadSpace
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from PIL import Image
from typing import List, Tuple, Any, Dict
from game_envs.base_env import BaseEnv
import gymnasium as gym 

logger = logging.getLogger(__name__)


class SuperMarioBrosEnv(BaseEnv):

    BUTTONS = ['A', 'B', 'left', 'right', 'up', 'down', 'NOOP']
    
    def __init__(
        self,
        world: int = 1,
        stage: int = 1,
        frame_skip: int = 4,
        frame_stack: int = 1, 
        with_text_info: bool = True,
        render_mode: str = None,
        **kwargs
    ):
        super().__init__()
        self.env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage),apply_api_compatibility=True,render_mode=render_mode  )
        #joypad wrapper action space

        # if action_type == "simple":
        #     actions = SIMPLE_MOVEMENT
        # elif action_type == "complex":
        #     actions = COMPLEX_MOVEMENT
        # elif action_type == "right_only":
        #     actions = RIGHT_ONLY
        # else:
        #     raise ValueError(f"Invalid action type: {action_type}")
        
        # self.env = JoypadSpace(self.env, actions)
        # self.action_lookup = {tuple(action_item): index for index, action_item in enumerate(actions)}

        self.BUTTON_MAP = {
        'A': 0, 'B': 1, 'select': 2, 'start': 3,'up': 4, 'down': 5, 'left': 6, 'right': 7}
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.with_text_info = with_text_info
        self.frame_history = []
        # self.image_resize = image_resize
        
        # get observation space and action space from the environment
        self.last_info = {}

    
    def initial_obs(self) -> Dict:
        return self.get_observation()
    
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        state, info = self.env.reset()
        self.frame_history = [self.process_frame(state) for _ in range(self.frame_stack)]
        self.last_info = info
        obs = self.get_observation()
        return obs, info


    def step(self, action: List[str]) -> Tuple[Dict, float, bool, bool, Dict[str, Any]]:
        action = [str(btn) for btn in action]
        invalid_buttons = []
        
        # button to bit map
        button_to_bit = {
            'right': 0b10000000,  # 128
            'left':  0b01000000,  # 64
            'down':  0b00100000,  # 32
            'up':    0b00010000,  # 16
            'start': 0b00001000,  # 8
            'select':0b00000100,  # 4
            'b':     0b00000010,  # 2
            'a':     0b00000001,  # 1
            'noop':  0b00000000,  # 0
        }
        
        # byte action
        byte_action = 0
        valid_buttons = []
        
        for btn in action:
            btn_lower = btn.lower()
            if btn_lower in button_to_bit:
                byte_action |= button_to_bit[btn_lower]
                valid_buttons.append(btn_lower)
            else:
                invalid_buttons.append(btn)
        
        total_reward = 0
        terminated = False
        truncated = False
        # skip frames based on action type
        has_a = 'a' in valid_buttons
        has_movement = any(b in ['left', 'right', 'up', 'down'] for b in valid_buttons)
        
        if has_a:  
            actual_skip = 30 
            # print(f"\n[跳跃动作] 执行 {actual_skip} 帧")
        elif has_movement:     
            actual_skip = 10 
            # print(f"\n[移动动作] 执行 {actual_skip} 帧")
        else:                  
            actual_skip = 1
            # print(f"\n[空动作] 执行 {actual_skip} 帧")
        
        for i in range(actual_skip):
            state, reward, terminated, truncated, step_info = self.env.step(byte_action)
            total_reward += reward
            self.frame_history.append(self.process_frame(state))
            self.last_info = step_info 
            
            if terminated or truncated:
                break

        # noop: align nespy simulator with pyboy simulator
        if not terminated and not truncated and actual_skip > 1:
            state, reward, terminated, truncated, step_info = self.env.step(button_to_bit['noop'])
            total_reward += reward
            self.frame_history.append(self.process_frame(state))
            self.last_info = step_info
        
        obs = self.get_observation()
        
        info = {
            'invalid_buttons': invalid_buttons,
            'frame_skip': actual_skip,  
            'byte_action': byte_action, 
            **self.last_info  
        }

        return obs, total_reward, terminated, truncated, info
    

	#with frame stack ver
    # def get_observation(self) -> Dict:
    #     current_frames = self.frame_history[-self.frame_stack:]
    #     if self.frame_stack == 1:
    #         obs_img = current_frames[0]
    #     else:
    #         obs_img = current_frames
        
    #     state_text = self._format_state_info(self.last_info)
        
    #     return {"image": obs_img, "state": state_text}
    
	#without frame stack ver

    def get_observation(self) -> Dict:
        obs_img = self.frame_history[-1] 
        if self.with_text_info:
            state_text = self._format_state_info(self.last_info)
        else:
            state_text = ""
        return {"image": obs_img, "state": state_text}

    # def resize_image(self, image: Image.Image, resize_ratio: float = 1.0) -> Image.Image:
    #     if resize_ratio != 1.0:
    #         """Resize the image to the specified ratio."""
    #         new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
    #         resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    #         return resized_image
    #     else:   
    #         """Return the original image."""
    #         return image

    def process_frame(self, state):
        screen = Image.fromarray(state)
        # screen = self.resize_image(screen, self.image_resize)
        return screen
    
    def _format_state_info(self, info: Dict) -> str:
        if not info:
            return "Super Mario Bros - No state info available"
        
        world = info.get('world', '?')
        stage = info.get('stage', '?')
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        time = info.get('time', 0)
        status = info.get('status', 'small')
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 0)
        life = info.get('life', 3)  
        flag_get = info.get('flag_get', False)  
        
        state_str = f"""Super Mario Bros: World {world}-{stage}
		Score: {score}
		Coins: {coins}
		Time: {time}
		Mario Status: {status}
		Position: ({x_pos}, {y_pos})
		Lives: {life}
		Flag reached: {flag_get}"""
        
        return state_str
    
    def close(self):
        self.env.close()
    
    def render(self):
        return self.env.render()


def parse_button_sequence(text: str) -> List[str]:
    """Parse button sequence from text with <answer> tags.
    
    Expected format: <answer>["button1", "button2"]</answer>
    or: <answer>['button1', 'button2']</answer>
    
    Available actions in Super Mario Bros:
    - ['NOOP']
    - ['right']
    - ['right', 'A']
    - ['right', 'B']
    - ['right', 'A', 'B']
    - ['A']
    - ['left']
    
    Args:
        text: Text to parse containing <answer> tags with a list of buttons
        
    Returns:
        List of button strings. Invalid buttons will be handled
        by the game environment and reported in info['invalid_buttons'].
    """
    try:
        # Extract content from <answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, text, re.DOTALL)
        
        if match:
            answer_content = match.group(1).strip()
        else:
            answer_content = text.strip()
            # raise ValueError(f"No <answer> tag found in text")
        
        # Parse as Python list literal (handles both single and double quotes)
        buttons = ast.literal_eval(answer_content)
        
        if not isinstance(buttons, list):
            raise ValueError(f"Answer content is not a list: {answer_content}")
        
        return [str(btn) for btn in buttons]
    
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse answer as list: {answer_content}, error: {e}")
