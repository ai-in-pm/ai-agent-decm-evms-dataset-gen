import os
import sys
import signal
import logging
import psutil
from src.visualization.dashboard import EVMSDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_existing_process(port):
    """Clean up any existing process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    logger.info(f"Terminating existing process using port {port}")
                    proc.terminate()
                    proc.wait(timeout=5)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    return False

def main():
    port = 8501
    try:
        # Clean up any existing process
        cleanup_existing_process(port)
        
        # Create and run the dashboard
        dashboard = EVMSDashboard()
        logger.info(f"Starting dashboard on http://127.0.0.1:{port}")
        dashboard.run_server(debug=True, port=port)
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    main()
