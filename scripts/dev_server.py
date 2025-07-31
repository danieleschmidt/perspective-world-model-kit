#!/usr/bin/env python3
"""Development server for PWMK with hot reloading and debugging."""

import argparse
import os
import sys
from pathlib import Path
import uvicorn
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeReloadHandler(FileSystemEventHandler):
    """Handle file changes for hot reloading."""
    
    def __init__(self, server_process):
        self.server_process = server_process
        self.patterns = {'.py', '.yaml', '.yml', '.json', '.toml'}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in self.patterns:
            logger.info(f"Code change detected in {file_path}")
            # Server will auto-reload with uvicorn's reload flag


def main():
    """Start development server with optional features."""
    parser = argparse.ArgumentParser(description='PWMK Development Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--reload', action='store_true', default=True, help='Enable auto-reload')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--jupyter', action='store_true', help='Start Jupyter Lab alongside server')
    parser.add_argument('--tensorboard', action='store_true', help='Start TensorBoard')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    if args.debug:
        os.environ['DEBUG'] = '1'
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    logger.info(f"Starting PWMK development server on {args.host}:{args.port}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path: {sys.path[0]}")
    
    # Start additional services if requested
    if args.jupyter:
        import subprocess
        jupyter_cmd = [
            sys.executable, '-m', 'jupyter', 'lab',
            '--ip=0.0.0.0', '--port=8888',
            '--no-browser', '--allow-root',
            f'--notebook-dir={project_root}'
        ]
        logger.info("Starting Jupyter Lab on port 8888...")
        subprocess.Popen(jupyter_cmd)
    
    if args.tensorboard:
        import subprocess
        tb_cmd = [
            sys.executable, '-m', 'tensorboard.main',
            '--logdir=logs/',
            '--host=0.0.0.0',
            '--port=6006'
        ]
        logger.info("Starting TensorBoard on port 6006...")
        subprocess.Popen(tb_cmd)
    
    # Start file watcher for additional logging
    if args.reload:
        observer = Observer()
        handler = CodeReloadHandler(None)
        observer.schedule(handler, str(project_root / 'pwmk'), recursive=True)
        observer.start()
        logger.info("File watcher started for hot reloading")
    
    try:
        # Check if we have a web server module, otherwise just run CLI
        try:
            from pwmk.server import app  # Hypothetical web interface
            uvicorn.run(
                "pwmk.server:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
                log_level="debug" if args.debug else "info",
                reload_dirs=[str(project_root / 'pwmk')]
            )
        except ImportError:
            logger.info("No web server module found, running CLI interface")
            from pwmk.cli import main as cli_main
            cli_main()
            
    except KeyboardInterrupt:
        logger.info("Development server stopped")
    finally:
        if args.reload:
            observer.stop()
            observer.join()


if __name__ == '__main__':
    main()