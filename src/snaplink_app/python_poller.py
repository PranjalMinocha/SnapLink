import requests
import time
import subprocess

# IMPORTANT: Replace with your laptop's local WiFi IP address and the port your server is running on.
# Example: "http://192.168.1.15:8000"
API_BASE_URL = "https://unburnt-franklin-exciting.ngrok-free.dev"

# This ID must match what your web app sends as the target device.
DEVICE_ID = "my-android-phone"


def set_silent_mode(state: bool):
    """
    Uses Termux:API to set all critical volume streams to 0 for silence,
    or restores them to a default level. This is a reliable alternative to DND.
    """
    # The list of audio streams we want to control.
    # 'music', 'ring', 'notification', 'system', 'alarm'
    streams = ["ring", "notification", "music", "system"]

    if state:
        # State is True, so we want to SILENCE the phone.
        print("ACTION: Setting Silent Mode ON")
        volume_level = "0"
    else:
        # State is False, so we want to UNSILENCE the phone.
        print("ACTION: Setting Silent Mode OFF")
        volume_level = "10"  # Set to a reasonable default (max is 15).

    # Loop through each stream and set its volume.
    for stream in streams:
        try:
            print(f"...setting {stream} volume to {volume_level}")
            subprocess.run(["termux-volume", stream, volume_level])
        except FileNotFoundError:
            print(f"Error: 'termux-volume' command not found. Is Termux:API installed?")
            return  # Exit the function if the command doesn't exist

    print("ACTION: Volume control finished.")


def set_volume(level: float):
    """Uses Termux:API to set media volume (0-15)."""
    # Ensure the level is between 0.0 and 1.0.
    level = max(0.0, min(1.0, level))
    volume_level = int(level * 15)
    print(f"Setting music volume to {volume_level}")
    subprocess.run(["termux-volume", "music", str(volume_level)])


def find_my_device():
    """Plays a loud sound using Termux:API."""
    print("ACTION: Playing 'Find My Device' sound!")
    # A simple way is to max the notification volume and play the default notification sound.
    subprocess.run(["termux-volume", "notification", "15"])
    subprocess.run(["termux-notification"])  # This plays the sound.


def poll_for_commands():
    """The main loop to check the server for commands."""
    url = f"{API_BASE_URL}/command/receive/{DEVICE_ID}"
    while True:
        try:
            print(f"Checking for command at {url}...")
            response = requests.get(url, timeout=10)  # Increased timeout slightly
            response.raise_for_status()

            command = response.json()

            if command:
                print(f"Received command: {command}")
                action = command.get("action")
                value = command.get("value")

                # The server sends an action, and we decide which function to call.
                if action == "set_dnd":  # Your server can still send "set_dnd" for simplicity.
                    set_silent_mode(value)
                elif action == "set_volume":
                    set_volume(value)
                elif action == "find_my_device":
                    find_my_device()

        except requests.exceptions.RequestException as e:
            print(f"Could not connect to server: {e}")
            print("Will retry in 10 seconds...")
            time.sleep(7)  # Wait a bit longer if the network is down.

        time.sleep(3)  # Wait 3 seconds before checking again.


if __name__ == "__main__":
    # Ensure Termux:API is installed before starting.
    try:
        subprocess.run(["termux-volume"], capture_output=True, check=True, timeout=3)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("=" * 40)
        print("!!! CRITICAL ERROR !!!")
        print("'termux-volume' command failed. Please ensure:")
        print("1. Termux:API app is installed from F-Droid.")
        print("2. You have run 'pkg install termux-api' in Termux.")
        print("=" * 40)
    else:
        print("Termux:API check successful. Starting poller...")
        poll_for_commands()