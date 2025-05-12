import sys
import types
import gymnasium
import numpy as np

original_make = gymnasium.make

def patched_make(id, **kwargs):
    """A patched version of gymnasium.make that removes render_mode and prevents rendering."""
    if 'render_mode' in kwargs:
        del kwargs['render_mode']
    
    # Create the environment without render_mode
    env = original_make(id, **kwargs)
    
    # Patch the render method of the environment
    original_render = env.render
    
    def safe_render(*args, **kwargs):
        """A safe render method that returns a blank frame instead of actually rendering."""
        try:
            # Try the original render, but catch any exceptions
            return original_render(*args, **kwargs)
        except Exception as e:
            # If rendering fails, return a blank frame
            print(f"Rendering failed with error: {e}")
            # Return a blank frame with the right dimensions
            if hasattr(env.observation_space, 'shape'):
                height, width = 480, 640  # Default size
                return np.zeros((height, width, 3), dtype=np.uint8)
            return None
    
    # Replace the render method
    env.render = safe_render
    
    # Also try to disable any internal rendering
    if hasattr(env, 'unwrapped'):
        if hasattr(env.unwrapped, 'render_mode'):
            env.unwrapped.render_mode = None
        if hasattr(env.unwrapped, 'mujoco_renderer'):
            # Disable the mujoco renderer if it exists
            env.unwrapped.mujoco_renderer = None
    
    return env

def apply_patch():
    """Apply the patch to disable rendering."""
    gymnasium.make = patched_make
    print("Rendering disabled via monkey patch")
