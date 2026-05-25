import os
import base64
import json
import logging
import argparse
import re
import random
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
from omegaconf import OmegaConf
from openai import OpenAI
from google import genai
from anthropic import Anthropic
from google.genai.types import GenerateContentConfig, Part, Content

from utils import parse_button_sequence, get_class_from_name, resize_image
from game_envs import BaseEnv
from game_envs.super_mario_land.env_prompts import SYSTEM_PROMPT, OLD_SYSTEM_PROMPT, FORMAT_PROMPT


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

def get_image_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_bytes(image: Image.Image) -> bytes:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the game agent evaluation.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Game Agent Evaluation Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--game_config_path", 
        type=str, 
        required=True, 
        help="Path to the game environment config file"
    )
    parser.add_argument(
        "--interaction_model_config_path", 
        type=str, 
        required=True, 
        help="Path to the interaction model config file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./logs", 
        help="Directory to save evaluation logs"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=10, 
        help="Number of evaluation steps to run"
    )
    parser.add_argument(
        "--max_history", 
        type=int, 
        default=2, 
        help="Maximum number of history turns to keep in the prompt"
    )
    
    # Boolean flags (positive integers mean true, 0 means false)
    parser.add_argument(
        "--with_text_info", 
        type=int,
        default=0,
        help="Whether to use text info (positive integer = true, 0 = false)"
    )
    parser.add_argument(
        "--with_game_area", 
        type=int,
        default=0,
        help="Whether to use game area information (positive integer = true, 0 = false)"
    )
    
    # Game-specific arguments
    parser.add_argument(
        "--game_world", 
        type=int, 
        default=0, # 0 means use config default
        help="World value for super mario land"
    )
    parser.add_argument(
        "--game_level", 
        type=int, 
        default=0, # 0 means use config default
        help="Level value for super mario land"
    )
    parser.add_argument(
        "--game_state",
        type=str,
        default=None,
        help="Game state to load and initialize the environment"
    )
    parser.add_argument(
        "--image_resize_ratio", 
        type=float, 
        default=1.0, 
        help="Resize ratio for observation image"
    )
    parser.add_argument(
        "--image_resize_width", 
        type=int, 
        default=None, 
        help="Resize width for observation image"
    )
    parser.add_argument(
        "--image_resize_height", 
        type=int, 
        default=None, 
        help="Resize height for observation image"
    )
    parser.add_argument(
        "--use_old_prompt", 
        type=int,
        default=0,
        help="Whether to use old prompt (positive integer = true, 0 = false)"
    )
    parser.add_argument(
        "--use_old_tick", 
        type=int,
        default=0,
        help="Whether to use old tick (positive integer = true, 0 = false)"
    )
    # arguments: random noisy action
    parser.add_argument(
        "--epsilon_noise",
        type=float,
        default=0,
        help="Epsilon probability for random noisy action (0 = always following model response, positive float = probability of including random action)"
    )
    parser.add_argument(
        "--noise_actions",
        type=str,
        nargs='+',  # Accept one or more arguments
        default=["a", "b", "up", "down", "left", "right"],
        help="List of actions to include in random noisy action (space-separated)"
    )
    parser.add_argument(
        "--run_count",
        type=int,
        default=None,
        help="Run count/ID for this evaluation (will be included in log directory name)"
    )
    
    return parser.parse_args()


class GameAgentEvaluationLoop:
    """An agent loop that evaluates a game-playing agent in a game environment.
    
    This class handles the complete evaluation pipeline including:
    - Game environment initialization
    - Model setup (interaction)
    - Evaluation loop execution
    - Logging and result saving
    
    Args:
        game_config_path: Path to game environment configuration
        interaction_config_path: Path to interaction model configuration
        log_dir: Directory to save evaluation logs
        steps: Number of evaluation steps to run
        max_history: Maximum conversation history to keep
        with_text_info: Whether to include text information in observations
        with_game_area: Whether to include game area information
        game_world: World number for Super Mario Land
        game_level: Level number for Super Mario Land
        game_state: Game state to load and initialize the environment
        image_resize_ratio: Factor to resize observation images
        image_resize_width: Target width to resize observation images
        image_resize_height: Target height to resize observation images
    """

    def __init__(
        self,
        game_config_path: str,
        interaction_config_path: str,
        log_dir: str, 
        steps: int, 
        max_history: int,
        with_text_info: bool,
        with_game_area: bool,
        game_world: int,
        game_level: int,
        game_state: str,
        image_resize_ratio: float,
        image_resize_width: int,
        image_resize_height: int,
        use_old_prompt: bool,
        use_old_tick: bool,
        epsilon_noise: float,
        noise_actions: Optional[list[str]] = None,
        run_count: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Initialize game environment from config file
        try:
            self.game_env = self._init_game(game_config_path, game_world, game_level, game_state)
            self.game_env.with_game_area = with_game_area
            self.game_env.with_text_info = with_text_info
            self.game_env.use_old_tick = use_old_tick
        except Exception as e:
            logger.error(f"Failed to initialize game environment: {e}")
            raise

        # Initialize interaction model from config file
        try:
            self.interaction_prompt = OLD_SYSTEM_PROMPT if use_old_prompt else SYSTEM_PROMPT
            self.interaction_client, self.interaction_type, self.interaction_params = self._init_model(interaction_config_path)
        except Exception as e:
            logger.error(f"Failed to initialize interaction model: {e}")
            raise

        if self.interaction_type == "llm" and not with_game_area:
            raise ValueError(
                "An interaction model of type LLM requires game area information to be enabled."
            )
        
        # Store configuration
        self.steps = steps
        self.max_history = max_history
        self.image_resize_ratio = image_resize_ratio
        self.image_resize_width = image_resize_width
        self.image_resize_height = image_resize_height
        # Setup logging directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if run_count is not None:
            self.log_dir = os.path.join(log_dir, f"eval_{timestamp}_run{run_count}")
        else:
            self.log_dir = os.path.join(log_dir, f"eval_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        # Noisy actions
        self.epsilon_noise = epsilon_noise
        self.noise_actions = noise_actions

        # Save configuration
        self.save_config(game_world, game_level, game_state, with_text_info, with_game_area, use_old_prompt, use_old_tick, epsilon_noise, noise_actions)

    @staticmethod
    def _init_game(game_config_path: str, game_world: int, game_level: int, game_state: str) -> BaseEnv:
        """Initialize game environment from configuration file.
        
        Args:
            game_config_path: Path to game configuration file
            game_world: World number for Super Mario Land (0 = use config default)
            game_level: Level number for Super Mario Land (0 = use config default)
            game_state: Game state to load and initialize the environment (None = start from game default initialization)
            
        Returns:
            Initialized game environment
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
            RuntimeError: If game environment initialization fails
        """
        if not os.path.exists(game_config_path):
            raise FileNotFoundError(f"Game config file not found: {game_config_path}")
            
        try:
            game_config = OmegaConf.load(game_config_path)
        except Exception as e:
            raise ValueError(f"Failed to load game config: {e}")

        game_class_name = game_config.get("class_name")
        if not game_class_name:
            raise ValueError("Game config must specify 'class_name'")
            
        try:
            game_class = get_class_from_name(game_class_name)
        except Exception as e:
            raise ValueError(f"Failed to get game class '{game_class_name}': {e}")
            
        game_env_config = OmegaConf.to_container(game_config.get("config", {}), resolve=True)
        if not isinstance(game_env_config, dict):
            raise ValueError("Game config 'config' section must be a dictionary")
            
        # Override world/level if specified
        if game_world > 0:
            game_env_config["world"] = game_world
        if game_level > 0:
            game_env_config["level"] = game_level
        if game_state is not None:
            game_env_config["init_state"] = game_state
        
        try:
            game_env = game_class(**game_env_config)
            return game_env
        except Exception as e:
            raise RuntimeError(f"Failed to initialize game environment: {e}")
    
    @staticmethod
    def _init_model(config_path: str) -> Tuple[Any, str, Dict[str, Any]]:
        """Initialize model client from configuration file.
        
        Args:
            config_path: Path to model configuration file
            
        Returns:
            Tuple of (client, model_type, parameters)
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
            RuntimeError: If client initialization fails
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config file not found: {config_path}")
            
        try:
            model_config = OmegaConf.load(config_path)
        except Exception as e:
            raise ValueError(f"Failed to load model config: {e}")

        model_name = model_config.get("model_name")
        if not model_name:
            raise ValueError("Model config must specify 'model_name'")

        model_type = model_config.get("model_type", "vlm").lower().strip()
        if model_type not in ["llm", "vlm"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Supported types are 'llm' and 'vlm'.")

        # Extract configuration parameters with defaults
        api_key = model_config.get("api_key", "dummy_key")
        base_url = model_config.get("base_url", None)
        timeout = model_config.get("timeout", 60)
        temperature = model_config.get("temperature", 1.0)
        top_p = model_config.get("top_p", 1.0)
        max_tokens = model_config.get("max_tokens", 8192)
        top_k = model_config.get("top_k", -1)
        enable_thinking = model_config.get("enable_thinking", None)
        reasoning_effort = model_config.get("reasoning_effort", None)
        text_verbosity = model_config.get("text_verbosity", None)

        # Validate required parameters
        if not api_key:
            raise ValueError("Model config must specify 'api_key'")

        try:
            if "gemini-3" in model_name:
                client = genai.Client(api_key=api_key)
            elif "claude" in model_name:
                print(f"Initializing Claude client with API key: {api_key}")
                client = Anthropic(api_key=api_key)
            else:
                client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model client: {e}")

        # Build parameters dictionary
        params: Dict[str, Any] = {
            "model": model_name,
            "top_p": top_p,
        }

        if "gpt" in model_name or model_name == "o3":
            if "gpt" in model_name:
                params["top_p"] = top_p
                params["temperature"] = temperature
            params["max_completion_tokens"] = max_tokens
            params["reasoning_effort"] = reasoning_effort
            params["verbosity"] = text_verbosity
        elif "gemini-3" in model_name:
            params["max_output_tokens"] = max_tokens
            params["temperature"] = temperature
            params["top_p"] = top_p
            params["top_k"] = top_k
        elif "claude" in model_name:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature
            params["top_p"] = top_p
            params["top_k"] = top_k
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature
            extra_params = {"top_k": top_k}
            if enable_thinking is not None:
                extra_params["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
            params["extra_body"] = extra_params

        return client, model_type, params
    
    def save_config(self, world: int, level: int, game_state: str, with_text_info: bool, with_game_area: bool, use_old_prompt: bool, use_old_tick: bool, epsilon_noise: float, noise_actions: list[str]) -> None:
        """Save evaluation configuration to JSON file.
        
        Args:
            world: Game world number
            level: Game level number
            game_state: Game state to load and initialize the environment
            with_text_info: Whether to use text info
            with_game_area: Whether to use game area
            use_old_prompt: Whether to use old prompt
            use_old_tick: Whether to use old tick
        """
        config_data = {
            "interaction_model": {
                "type": self.interaction_type,
                "params": self.interaction_params,
            },
            "game_env": str(self.game_env),
            "game_world": world,
            "game_level": level,
            "game_state": game_state,
            "steps": self.steps,
            "max_history": self.max_history,
            "with_text_info": with_text_info,
            "with_game_area": with_game_area,
            "image_resize_ratio": self.image_resize_ratio,
            "image_resize_width": self.image_resize_width,
            "image_resize_height": self.image_resize_height,
            "use_old_prompt": use_old_prompt,
            "use_old_tick": use_old_tick,
            "epsilon_noise": epsilon_noise,
            "noise_actions": noise_actions,
            "timestamp": datetime.now().isoformat(),
        }
        
        config_path = os.path.join(self.log_dir, "evaluation_config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def save_interaction_log(
        self, 
        step: int, 
        obs_image: Image.Image, 
        messages: List[Dict[str, Any]], 
        button_sequence: Optional[List[str]] = None,
        button_sequence_eps: Optional[List[str]] = None,
        score: Optional[float] = None,
        progress: Optional[float] = None,
        done: Optional[bool] = None,
        error: Optional[str] = None
    ) -> None:
        """Save interaction log for a single evaluation step.
        
        Args:
            step: Current step number
            obs_image: Observation image
            messages: Conversation messages
            button_sequence: Button sequence pressed (if any)
            score: Game score received
            progress: Level progress achieved
            done: Whether episode is finished
            error: Error message (if any)
        """
        try:
            # Save image (post-action observation for this step)
            image_path = os.path.join(self.log_dir, f"step_{step:03d}_observation.png")
            obs_image.save(image_path, optimize=True)
            # image_path in JSON points to step-1 (observation model actually saw for this interaction)
            image_path = os.path.join(self.log_dir, f"step_{step - 1:03d}_observation.png")
            
            # Convert messages to text format for logging
            messages_text = []
            for message in messages:
                if message["role"] == "user":
                    # Handle both text-only and multimodal user messages
                    if "content" in message.keys():
                        content = message["content"]
                    elif "parts" in message.keys():
                        content = message["parts"]
                    else:
                        raise ValueError(f"Invalid message format: {message}")
                    if isinstance(content, list) and len(content) > 0:
                        messages_text.append(content[0]["text"])
                    else:
                        messages_text.append(str(content))
                else:
                    if "content" in message.keys():
                        content = message["content"]
                    elif "parts" in message.keys():
                        content = message["parts"]
                    else:
                        raise ValueError(f"Invalid message format: {message}")
                    messages_text.append(str(content))
                    
            interaction_data = {
                "step": step,
                "messages": messages_text,
                "button_sequence": button_sequence,
                "game_score": score,
                "level_progress": progress,
                "done": done,
                "error": error,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add epsilon noise button sequence if epsilon_noise > 0
            if self.epsilon_noise > 0 and button_sequence_eps is not None:
                interaction_data["button_sequence_eps"] = button_sequence_eps

            json_path = os.path.join(self.log_dir, f"step_{step:03d}_interaction.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(interaction_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved interaction log to {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save interaction log for step {step}: {e}")
            # Don't raise here to avoid breaking the evaluation loop
    
    def get_obs_text(self, obs: Dict[str, Any]) -> str:
        """Generate observation text with formatting instructions.
        
        Args:
            obs: Observation dictionary containing game state
            
        Returns:
            Formatted observation text with instructions
        """
        obs_text = obs.get("state", "")
        
        return self.interaction_prompt

    def run_evaluation(self) -> None:
        """Run the complete evaluation loop."""
        try:
            self._initialize_evaluation()
            
            for step in range(self.steps):
                logger.info(f"Starting evaluation step {step + 1}/{self.steps}")
                
                # Get interaction response
                interaction_response_text = self._get_interaction_response()
                
                # Execute action and get next observation
                obs, score, progress, done, button_sequence, button_sequence_eps = self._execute_action(interaction_response_text)

                # Add assistant response to messages for saving (before truncation)
                # This ensures we save the full interaction context even when max_history == 0
                self._add_assistant_response(interaction_response_text)
                
                # Save interaction log before truncating conversation history
                # (to preserve full interaction context before truncation)
                self._save_step_log(step + 1, obs, score, progress, done, interaction_response_text, button_sequence, button_sequence_eps)
                
                # Truncate conversation history if needed
                self._truncate_conversation_history()
                
                if done:
                    logger.info(f"Episode finished at step {step + 1}")
                    break
            
            logger.info(f"Evaluation completed. Final game_score: {score}. Final level_progress: {progress}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        finally:
            self.cleanup()
    
    def _get_initial_level_progress_and_score(self) -> Tuple[float, float]:
        """Get level_progress and game score at current env state (before any step)."""
        env = self.game_env
        class_name = env.__class__.__name__
        if class_name == "PyBoySuperMarioLandEnv":
            return float(env.mario.level_progress), float(env.mario.score)
        if class_name == "SuperMarioBrosEnv":
            info = getattr(env, "last_info", {}) or {}
            return float(info.get("x_pos", 0.0)), float(info.get("score", 0.0))
        return 0.0, 0.0

    def _initialize_evaluation(self) -> None:
        """Initialize the evaluation environment and first observation."""
        obs, info = self.game_env.reset(seed=1)  # seed for reproducibility

        self.obs = obs if obs is not None else self.game_env.initial_obs()
        self.obs_text = self.get_obs_text(self.obs)
        self.obs_image = self.obs["image"]
        self.obs_image = resize_image(self.obs_image, self.image_resize_ratio, self.image_resize_width, self.image_resize_height)  # resize image handled before input, environment returns original size from simulator
        self.obs_image_url = f"data:image/png;base64,{get_image_base64(self.obs_image)}"
        self.obs_image_b64 = get_image_base64(self.obs_image)
        self.obs_image_bytes = get_image_bytes(self.obs_image)
        image_path = os.path.join(self.log_dir, f"step_{0:03d}_observation.png")
        self.obs_image.save(image_path, optimize=True)
        logger.info(f"Saved initial observation to {image_path}")

        # Save step_000_interaction.json with actual initial level_progress/score (no action yet)
        initial_progress, initial_score = self._get_initial_level_progress_and_score()
        step0_data = {
            "step": 0,
            "messages": [],
            "button_sequence": None,
            "game_score": initial_score,
            "level_progress": initial_progress,
            "done": False,
            "error": None,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
        }
        step0_json = os.path.join(self.log_dir, "step_000_interaction.json")
        with open(step0_json, "w", encoding="utf-8") as f:
            json.dump(step0_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved initial state to {step0_json} (level_progress={initial_progress}, game_score={initial_score})")

        self.interaction_messages = []
    
    def _get_interaction_response(self) -> str:
        """Get interaction response from the model."""
        # Add user message to conversation
        if self.interaction_type == "vlm" and "gemini-3" in self.interaction_params["model"]:
            interaction_user_message_content = [
                {"text": self.obs_text},
                {"inline_data": {"mime_type": "image/png", "data": self.obs_image_bytes}},
            ]
        elif self.interaction_type == "vlm" and "claude" in self.interaction_params["model"]:
            interaction_user_message_content = [
                {"type": "text", "text": self.obs_text},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.obs_image_b64}},
            ]
        elif self.interaction_type == "vlm":
            interaction_user_message_content = [
                {"type": "text", "text": self.obs_text},
                {"type": "image_url", "image_url": {"url": self.obs_image_url}},
            ]
        else:
            interaction_user_message_content = [
                {"type": "text", "text": self.obs_text},
            ]
        if "gemini-3" in self.interaction_params["model"]:
            self.interaction_messages.append({
                "role": "user",
                "parts": interaction_user_message_content,
            })
        else:
            self.interaction_messages.append({
                "role": "user", 
                "content": interaction_user_message_content
            })

        # Get interaction response
        if "gemini-3" in self.interaction_params["model"]:
            interaction_response = self.interaction_client.models.generate_content(
                contents=self.interaction_messages,
                model=self.interaction_params["model"],
                config=GenerateContentConfig(
                    max_output_tokens=self.interaction_params["max_output_tokens"],
                    temperature=self.interaction_params["temperature"],
                    top_p=self.interaction_params["top_p"],
                    top_k=self.interaction_params["top_k"],
                )
            )
            content = interaction_response.text
        elif "claude" in self.interaction_params["model"]:
            interaction_response = self.interaction_client.messages.create(
                model=self.interaction_params["model"],
                max_tokens=self.interaction_params["max_tokens"],
                temperature=self.interaction_params["temperature"],
                # top_p=self.interaction_params["top_p"],
                top_k=self.interaction_params["top_k"],
                # thinking={"type": "adaptive"},
                messages=self.interaction_messages,
            )
            content = interaction_response.content[0].text
        else:
            interaction_response = self.interaction_client.chat.completions.create(
                messages=self.interaction_messages,
                **self.interaction_params,
            )
            content = interaction_response.choices[0].message.content

        # Handle content as list (vision models) or string
        if isinstance(content, list):
            # Extract text from content parts
            text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
            content = "".join(text_parts)
        
        return content
    
    def _execute_action(self, interaction_response_text: str) -> Tuple[Dict[str, Any], float, float, bool, List[str], List[str]]:
        """Execute action in game environment and return new observation.
        
        Args:
            interaction_response_text: Interaction response text
            
        Returns:
            Tuple of (observation, score, progress, done, button_sequence, button_sequence_eps)
        """
        try:
            button_sequence = parse_button_sequence(interaction_response_text)

            # With epsilon probability, add a rndom noisy action
            if self.epsilon_noise > 0 and random.random() < self.epsilon_noise:
                if len(self.noise_actions) == 0:
                    button_sequence_eps = random.sample(self.game_env.action_space, 2)
                elif len(self.noise_actions) == 1:
                    button_sequence_eps = [self.noise_actions[0]]
                else:
                    button_sequence_eps = random.sample(self.noise_actions, 2)
            else:
                button_sequence_eps = button_sequence

            # Execute action in the game environment
            obs, reward, terminated, truncated, info = self.game_env.step(button_sequence_eps)
            # Dual environment support: Super Mario Land vs Super Mario Bros
            if self.game_env.__class__.__name__ == "PyBoySuperMarioLandEnv":
                progress = info.get("level_progress", 0.0)
                score = info.get("game_score", 0.0)
            elif self.game_env.__class__.__name__ == "SuperMarioBrosEnv":
                progress = float(info.get("x_pos", 0.0))
                score = float(info.get("score", 0.0))
            else:
                raise ValueError(f"Unsupported game environment class: {self.game_env.__class__.__name__}")
            done = terminated or truncated

            # Handle invalid buttons
            if info.get("invalid_buttons"):
                self.obs_text = f"Invalid buttons: {info['invalid_buttons']}\n" + self.get_obs_text(obs)
            else:
                self.obs_text = self.get_obs_text(obs)
                
            self.obs = obs
            self.obs_image = obs["image"]
            self.obs_image = resize_image(self.obs_image, self.image_resize_ratio, self.image_resize_width, self.image_resize_height)  # resize image handled before input, environment returns original size from simulator
            self.obs_image_url = f"data:image/png;base64,{get_image_base64(self.obs_image)}"
            self.obs_image_b64 = get_image_base64(self.obs_image)
            self.obs_image_bytes = get_image_bytes(self.obs_image)
            
            return obs, score, progress, done, button_sequence, button_sequence_eps
            
        except Exception as e:
            logger.warning(f"Error when interacting with game environment: {e}")
            
            # Return error state
            obs = self.game_env.get_observation()
            score = -100.0  # Error penalty
            progress = 0.0
            done = False
            button_sequence = None
            button_sequence_eps = None
            
            self.obs_text = f"Error when interacting with game environment: {e}\n" + self.get_obs_text(obs)
            self.obs = obs
            self.obs_image = obs["image"]
            self.obs_image = resize_image(self.obs_image, self.image_resize_ratio, self.image_resize_width, self.image_resize_height)  # resize image handled before input, environment returns original size from simulator
            self.obs_image_url = f"data:image/png;base64,{get_image_base64(self.obs_image)}"
            self.obs_image_b64 = get_image_base64(self.obs_image)
            self.obs_image_bytes = get_image_bytes(self.obs_image)
            
            return obs, score, progress, done, button_sequence, button_sequence_eps
    
    def _save_step_log(self, step: int, obs: Dict[str, Any], score: float, progress: float, done: bool, interaction_response_text: str, button_sequence: Optional[List[str]] = None, button_sequence_eps: Optional[List[str]] = None) -> None:
        """Save interaction log for current step."""
        # Try to parse button sequence from interaction response text if not provided
        if button_sequence is None:
            try:
                button_sequence = parse_button_sequence(interaction_response_text)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse button sequence from response: {e}")
        
        self.save_interaction_log(
            step=step,
            obs_image=self.obs_image,
            messages=self.interaction_messages,
            button_sequence=button_sequence,
            button_sequence_eps=button_sequence_eps if self.epsilon_noise > 0 else None,
            score=score,
            progress=progress,
            done=done,
        )
    
    def _add_assistant_response(self, interaction_response_text: str) -> None:
        """Add assistant response to conversation history.
        
        Args:
            interaction_response_text: The assistant's response text to add
        """
        # Add assistant response to conversation history only if not already present
        if "gemini-3" in self.interaction_params["model"]:
            self.interaction_messages.append({
                "role": "model",
                "parts": interaction_response_text
            })
        elif self.interaction_messages[-1]["role"] == "assistant":
            # Check if response is already in the message to avoid duplication
            last_content = str(self.interaction_messages[-1]["content"])
            if interaction_response_text not in last_content:
                self.interaction_messages[-1]["content"] += interaction_response_text
        else:
            self.interaction_messages.append({
                "role": "assistant", 
                "content": interaction_response_text
            })
    
    def _truncate_conversation_history(self) -> None:
        """Truncate conversation history based on max_history setting."""
        # Truncate history if exceeds max_history
        if self.max_history == 0:
            self.interaction_messages = []  # Empty message list
        elif len(self.interaction_messages) >= self.max_history * 2:
            self.interaction_messages = self.interaction_messages[-self.max_history * 2:]
    
    def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            if hasattr(self.game_env, 'close'):
                self.game_env.close()
                logger.info("Game environment closed successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        # Reset instance variables to free memory
        self.obs = None
        self.obs_text = ""
        self.obs_image = None
        self.obs_image_url = ""
        self.obs_image_b64 = ""
        self.obs_image_bytes = None
        self.interaction_messages = []


def main() -> None:
    """Main function to run game agent evaluation."""
    try:
        args = parse_args()
        
        # Validate required files exist
        if not os.path.exists(args.game_config_path):
            raise FileNotFoundError(f"Game config file not found: {args.game_config_path}")
        if not os.path.exists(args.interaction_model_config_path):
            raise FileNotFoundError(f"Interaction model config file not found: {args.interaction_model_config_path}")
        logger.info(f"Starting evaluation with args: {args}")
        
        evaluation_loop = GameAgentEvaluationLoop(
            game_config_path=args.game_config_path,
            interaction_config_path=args.interaction_model_config_path,
            log_dir=args.log_dir,
            steps=args.steps,
            max_history=args.max_history,
            with_text_info=bool(args.with_text_info > 0),
            with_game_area=bool(args.with_game_area > 0),
            game_world=args.game_world,
            game_level=args.game_level,
            game_state=args.game_state,
            image_resize_ratio=args.image_resize_ratio,
            image_resize_width=args.image_resize_width,
            image_resize_height=args.image_resize_height,
            use_old_prompt=bool(args.use_old_prompt > 0),
            use_old_tick=bool(args.use_old_tick > 0),
            epsilon_noise=args.epsilon_noise,
            noise_actions=args.noise_actions,
            run_count=args.run_count
        )
        
        evaluation_loop.run_evaluation()
        logger.info("Evaluation completed successfully.")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
