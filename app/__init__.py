"""
Black Sheep Ventures - Collective Intelligence & Simulation Engine
Flask Application Factory
"""

import os
import warnings

# Suppress resource_tracker warnings from multiprocessing/transformers
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask application factory function."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure JSON responses are formatted cleanly
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # Configure logging
    logger = setup_logger('blacksheep')
    
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 60)
        logger.info("Black Sheep Ventures Prediction Engine Initializing...")
        logger.info("=" * 60)
    
    # Enable CORS for frontend integrations
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register simulation process cleanup hooks
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Registered simulation process cleanup handler")
    
    # Request logging middleware
    @app.before_request
    def log_request():
        req_logger = get_logger('blacksheep.request')
        req_logger.debug(f"Request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            req_logger.debug(f"Payload: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        res_logger = get_logger('blacksheep.request')
        res_logger.debug(f"Response: {response.status_code}")
        return response
    
    # Register API Blueprints
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # Health check endpoint
    @app.route('/health')
    def health():
        return {
            'status': 'ok',
            'service': 'Black Sheep Ventures - Collective Intelligence Engine',
            'version': '1.0.0'
        }
    
    if should_log_startup:
        logger.info("Black Sheep Ventures Prediction Engine Ready")
    
    return app
