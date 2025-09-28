# This module will contain the functions for controlling the local machine (laptop).

def set_local_volume(level: float):
    """Sets the system volume. Level should be between 0.0 and 1.0."""
    print(f"ACTION: Setting local volume to {level * 100}%")
    # TODO: Add platform-specific code here.
    # For macOS: Use osascript subprocess
    # For Windows: Use a library like pycaw

def set_local_brightness(level: float):
    """Sets the screen brightness. Level should be between 0.0 and 1.0."""
    print(f"ACTION: Setting local brightness to {level * 100}%")
    # TODO: Add platform-specific code here.
    # For macOS: Use a command-line tool via subprocess
    # For Windows: Use a library like wmi