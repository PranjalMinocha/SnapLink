from pydantic import BaseModel
from typing import Literal

# Use Literal to define a strict set of allowed command actions.
# FastAPI will automatically validate that the 'action' field is one of these strings.
ActionType = Literal["set_dnd", "set_volume", "find_my_device"]


class DeviceCommand(BaseModel):
    """Defines the data structure for a command."""
    device_id: str  # A unique identifier for the target device, e.g., "purvansh-iphone"
    action: ActionType
    value: float | bool | None = None # An optional value, e.g., 0.8 for volume or True for DND