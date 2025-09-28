from fastapi import FastAPI,Request
from . import models
import requests

# A simple in-memory dictionary to act as a queue for commands.
# In a real app, this might be a more robust database like Redis.
COMMAND_QUEUE = {}

app = FastAPI(title="SnapLink API")


USER_KEY = "uzmshrgkdjw8dxbvm2d8poeykc5x6n"
APP_TOKEN = "axcch6rp1wpjft2f48xvjj7fiio2nw"  

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the server is running."""
    return {"status": "SnapLink API is running!"}


@app.post("/command/send", status_code=202)
def send_command(command: models.DeviceCommand):
    """
    Receives a command (e.g., from the Gesture Controller) and queues it
    for a specific device.
    """
    print(f"Received command: {command.action} for device: {command.device_id}")
    COMMAND_QUEUE[command.device_id] = command
    return {"message": "Command queued successfully"}


@app.get("/command/receive/{device_id}", response_model=models.DeviceCommand | None)
def receive_command(device_id: str):
    """
    Called by a supplementary app (e.g., on a phone) to poll for any
    pending commands.
    """
    if device_id in COMMAND_QUEUE:
        # Retrieve the command and remove it from the queue to ensure it's only executed once.
        command = COMMAND_QUEUE.pop(device_id)
        print(f"Sending command: {command.action} to device: {device_id}")
        return command

    # If no command is found for the device, return nothing.
    print(f"No command for device: {device_id}")
    return None


def send_pushover_message(message):
    data = {
        "token": APP_TOKEN,
        "user": USER_KEY,
        "message": message,
        "title": "Alert from Python API",
        "sound": "magic"  
    }
    response = requests.post("https://api.pushover.net/1/messages.json", data=data)
    print(response.status_code, response.text)


@app.post("/alert")
async def alert(payload: models.AlertPayload):
    message = payload.message.lower()

    if message == "sound":
        send_pushover_message("Sound detected!")
        return {"status": "Pushover notification sent"}
    else:
        return {"status": "No action taken"}
