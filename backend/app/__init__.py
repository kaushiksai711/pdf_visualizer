# Initialization file for Flask/Django apps
# __init__.py
from flask import Flask
from app.routes import api_blueprint
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Load configurations if any (optional)
    app.config.from_pyfile("config.py", silent=True)

    # Register Blueprints
    app.register_blueprint(api_blueprint)

    return app