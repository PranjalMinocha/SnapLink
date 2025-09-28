import uvicorn
import sys
from pathlib import Path

# Add the 'src' directory to Python's path to ensure the reloader can find the module.
SRC_PATH = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_PATH))

if __name__ == "__main__":
    uvicorn.run("snaplink_app.main:app", host="0.0.0.0", port=8000, reload=True)
