import os
import sys

# Running `python src/app.py` from the repo root puts only `src/` on sys.path by default.
# The `embeddings/` package lives next to `src/`, so add the repo root first.
_current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from flask import Flask

load_dotenv()
from flask_cors import CORS
try:
    from routes import register_routes
except ImportError:  # pragma: no cover
    from src.routes import register_routes

current_directory = _current_dir

# Serve React build files from <project_root>/frontend/dist
app = Flask(__name__,
    static_folder=os.path.join(project_root, 'frontend', 'dist'),
    static_url_path='')
CORS(app)

# Register routes
register_routes(app)

if __name__ == '__main__':
    # threaded=True helps the dev server keep accepting requests
    # while an SSE/streaming response is in-flight.
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
