# Main app entry point
from flask import Flask
from flask_cors import CORS
import os
from app.routes import api_blueprint

def create_app():
    """Initialize the Flask app."""
    app = Flask(__name__)
    CORS(app)  # Enable Cross-Origin Resource Sharing

    # Configure for large file uploads
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
    app.config['UPLOAD_CHUNK_SIZE'] = 1024 * 1024  # 1MB chunks

    # Ensure the uploads directory exists
    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Register blueprints
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app

if __name__ == "__main__":
    app = create_app()

    # Run the application
    app.run(port=5000, debug=True)
