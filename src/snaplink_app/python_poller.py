import requests
import time
import subprocess
import json

# IMPORTANT: Replace with your laptop's local WiFi IP address and the port your server is running on.
# Example: "http://192.168.1.15:8000"
API_BASE_URL = "https://unburnt-franklin-exciting.ngrok-free.dev"

# This ID must match what your web app sends as the target device.
DEVICE_ID = "my-android-phone"


def toggle_silent_mode():
    """
    Checks the phone's current volume state and does the opposite.
    If the phone is currently loud, it silences it.
    If the phone is currently silent, it restores volume.
    """
    print("ACTION: Toggling Silent Mode...")

    try:
        # --- Step 1: Get the current volume state from Termux ---
        # We use check_output to capture the JSON string the command returns.
        current_state_json = subprocess.check_output(["termux-volume"])

        # --- Step 2: Parse the JSON to read the volume levels ---
        # The output is a list of dictionaries, e.g., [{"stream": "ring", "volume": 10}, ...]
        volume_streams = json.loads(current_state_json)

        # Find the current volume of the 'ring' stream.
        current_ring_volume = 0
        for stream in volume_streams:
            if stream.get("stream") == "ring":
                current_ring_volume = stream.get("volume", 0)
                break  # Stop once we've found it

        print(f"...Current ring volume is {current_ring_volume}")

        # --- Step 3: Decide whether to mute or unmute ---
        streams_to_change = ["ring", "notification", "music", "system"]

        if current_ring_volume > 0:
            # The phone is NOT silent, so our action is to SILENCE it.
            print("...Phone is loud. Setting to SILENT.")
            new_volume_level = "0"
        else:
            # The phone IS silent, so our action is to UNSILENCE it.
            print("...Phone is silent. Setting to LOUD.")
            new_volume_level = "10"  # Restore to a default volume

        # --- Step 4: Apply the new volume level ---
        for stream in streams_to_change:
            subprocess.run(["termux-volume", stream, new_volume_level])

        print("ACTION: Toggle complete.")

    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error getting or parsing volume state: {e}")
        # Fallback to a simple mute if we can't read the state.
        for stream in ["ring", "notification"]:
            subprocess.run(["termux-volume", stream, "0"])


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
                if action == "set_dnd" or action == "set-dnd":  # Your server can still send "set_dnd" for simplicity.
                    toggle_silent_mode()
                # elif action == "set_volume":
                #     set_volume(value)
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