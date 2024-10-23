import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class StreamlitHandler(FileSystemEventHandler):
    def __init__(self, command):
        self.command = command
        self.process = None

    def start_process(self):
        self.process = subprocess.Popen(self.command, shell=True)

    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"{event.src_path} has been modified. Restarting Streamlit...")
            self.stop_process()
            self.start_process()

if __name__ == "__main__":
    command = "streamlit run streamlit_app_demo.py --browser.gatherUsageStats=false"
    event_handler = StreamlitHandler(command)
    observer = Observer()
    
    # Ensure the paths are correct
    project_dir = os.path.dirname(os.path.abspath(__file__))
    observer.schedule(event_handler, path=project_dir, recursive=True)
    observer.schedule(event_handler, path=os.path.join(project_dir, 'tabs'), recursive=True)
    
    observer.start()

    try:
        event_handler.start_process()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
