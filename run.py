"""
Black Sheep Ventures - Collective Intelligence & Simulation Engine
Server Entrypoint
"""

import os
import sys

# Windows UTF-8 console output setup
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def main():
    """Start Black Sheep Ventures backend service."""
    # Validate configuration
    errors = Config.validate()
    if errors:
        print("[!] Configuration Error:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease check your .env configuration file.")
        sys.exit(1)

    # Initialize application
    app = create_app()

    # Network configuration
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG

    print("=" * 60)
    print("  BLACK SHEEP VENTURES - PREDICTION & SIMULATION ENGINE")
    print(f"  Listening on: http://{host}:{port}")
    print("=" * 60)

    # Start Flask server
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()
