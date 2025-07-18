"""
Spot Instance Monitoring Utility - Detects spot instance termination notices and performs graceful shutdown
"""
import requests
import threading
import time

class SpotInstanceMonitor:
    def __init__(self, checkpoint_save_fn, check_interval=5):
        """
        Initialize a spot instance monitor
        
        Args:
            checkpoint_save_fn: Function to call when spot termination is detected
            check_interval: Interval in seconds between termination checks
        """
        self.checkpoint_save_fn = checkpoint_save_fn
        self.check_interval = check_interval
        self.should_stop = False
        self.monitor_thread = None
        
    def _monitor_loop(self):
        """Background monitoring thread main function"""
        print("Spot instance monitoring started")
        while not self.should_stop:
            try:
                # Check for termination notice
                response = requests.get(
                    "http://169.254.169.254/latest/meta-data/spot/termination-time",
                    timeout=2
                )
                
                # If we get HTTP 200, the instance is scheduled for termination
                if response.status_code == 200:
                    termination_time = response.text
                    print(f"SPOT TERMINATION NOTICE DETECTED! Termination time: {termination_time}")
                    print("Running emergency checkpoint save...")
                    
                    # Call the provided checkpoint function
                    self.checkpoint_save_fn()
                    
                    print("Emergency checkpoint saved. Shutting down monitoring.")
                    self.should_stop = True
                    break
                    
            except requests.exceptions.RequestException:
                # No termination notice if the request fails
                pass
                
            time.sleep(self.check_interval)
        
        print("Spot instance monitoring stopped")
    
    def start(self):
        """Start the monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.should_stop = False
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            return True
        return False
    
    def stop(self):
        """Stop the monitoring thread"""
        self.should_stop = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        return True
    
    def is_running(self):
        """Check if the monitoring thread is running"""
        return self.monitor_thread is not None and self.monitor_thread.is_alive()