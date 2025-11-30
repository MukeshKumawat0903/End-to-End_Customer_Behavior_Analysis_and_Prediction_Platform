#!/usr/bin/env python
"""
Script to start the API server.
Usage: python run_api.py [--host HOST] [--port PORT] [--reload]
"""
import argparse
import uvicorn

from src.config import get_config


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="Run the Customer Behavior Prediction API")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Override with command-line arguments
    host = args.host or config.api.host
    port = args.port or config.api.port
    reload = args.reload if args.reload else config.api.reload
    workers = args.workers or (1 if reload else config.api.workers)
    
    print(f"🚀 Starting API server on {host}:{port}")
    print(f"📚 API documentation: http://{host}:{port}/docs")
    print(f"🔄 Auto-reload: {reload}")
    print(f"👥 Workers: {workers}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
