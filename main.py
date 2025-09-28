from fastapi import FastAPI,Request
import requests

app = FastAPI()

USER_KEY = "uzmshrgkdjw8dxbvm2d8poeykc5x6n"
APP_TOKEN = "axcch6rp1wpjft2f48xvjj7fiio2nw"  


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
async def alert(request: Request):
    payload = await request.json()
    message = payload.get("message", "").lower()  

    if message == "sound":
        send_pushover_message("Sound detected!")
        return {"status": "Pushover notification sent"}
    else:
        return {"status": "No action taken"}

