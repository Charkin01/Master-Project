import os
import subprocess
import webbrowser
from time import sleep

def start_tensorboard(log_dir, port=6006):
    """
    Start TensorBoard server.
    
    Args:
        log_dir (str): Path to the directory containing TensorBoard logs.
        port (int): Port to run TensorBoard server on.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory '{log_dir}' does not exist.")
    
    # Print the contents of the log directory for verification
    print(f"Contents of log directory ({log_dir}):")
    for root, dirs, files in os.walk(log_dir):
        for name in files:
            print(os.path.join(root, name))
    
    # Start TensorBoard
    tensorboard_command = f"tensorboard --logdir={log_dir} --port={port} --reload_interval 5"
    process = subprocess.Popen(tensorboard_command, shell=True)

    # Wait a few seconds to ensure TensorBoard starts
    sleep(5)

    # Open TensorBoard in the default web browser
    webbrowser.open(f"http://localhost:{port}")

    return process

def main():
    log_dir = r'C:\Users\chirk\Downloads\Python\Master-Project\logs\fit'  # Update this path to your actual logs directory
    port = 6006  # Default TensorBoard port

    # Start TensorBoard server and open in browser
    process = start_tensorboard(log_dir, port)

    # Keep the script running to keep TensorBoard alive
    try:
        while True:
            sleep(10)
    except KeyboardInterrupt:
        print("Stopping TensorBoard...")
        process.terminate()

if __name__ == "__main__":
    main()
