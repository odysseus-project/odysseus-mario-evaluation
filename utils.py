import ast
import importlib
import re
from typing import List

from PIL import Image


def parse_button_sequence(message: str) -> List[str]:
    """Parse button sequence from assistant message with <answer> tags.
    
    Args:
        message: Assistant message containing <answer>["button1", "button2", ...]</answer>
        or <answer>['button1', 'button2', ...]</answer>
        
    Returns:
        List of button strings to press
    """
    # Extract content from <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, message, re.DOTALL)
    
    if not match:
        raise ValueError(f"No <answer> tag found in the assistant response")
    
    answer_content = match.group(1).strip()
    
    # Try to parse as Python list literal (handles both single and double quotes)
    try:
        buttons = ast.literal_eval(answer_content)
        if isinstance(buttons, list):
            if len(buttons) > 2:
                raise ValueError(f"Too many buttons: {buttons}. Maximum 2 buttons are allowed per turn.")
            return [str(btn).lower() for btn in buttons]
        else:
            raise ValueError(f"Answer content is not a list: {answer_content}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse answer as list: {answer_content}, error: {e}")


def get_class_from_name(class_name: str):
    """Dynamically import a class from a string."""

    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


def resize_image(image: Image.Image, resize_ratio: float = 1.0, resize_width: int = None, resize_height: int = None) -> Image.Image:
    if resize_ratio != 1.0:
        """Resize the image to the specified ratio."""
        new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        return resized_image
    if resize_width is not None and resize_height is not None:
        """Resize the image to the specified height and width."""
        resized_image = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        return resized_image
    return image
