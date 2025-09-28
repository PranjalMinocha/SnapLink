import requests

USER_KEY = "uzmshrgkdjw8dxbvm2d8poeykc5x6n"
APP_TOKEN = "axcch6rp1wpjft2f48xvjj7fiio2nw"  # create a Pushover app to get this


# Sending notification to just raise the volume
def send_pushover_message(message):
    data = {
        "token": APP_TOKEN,
        "user": USER_KEY,
        "message": message,
        "title": "Alert from Python API",
        "sound": "magic"   # optional, see Pushover sounds
    }
    response = requests.post("https://api.pushover.net/1/messages.json", data=data)
    print(response.status_code, response.text)

# Test
send_pushover_message("Hello! This is a test notification.")




