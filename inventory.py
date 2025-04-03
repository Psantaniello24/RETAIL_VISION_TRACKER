import os
import datetime
import pandas as pd
from config import PRODUCTS, REPORTS_DIR

class InventoryTracker:
    def __init__(self):
        self.inventory = {product: 0 for product in PRODUCTS}
        self.alerts = []
        self.history = []
        self._setup_reports_dir()
        
    def _setup_reports_dir(self):
        """Create reports directory if it doesn't exist."""
        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
            
    def update(self, detections):
        """Update inventory based on object detections.
        
        Args:
            detections: Dictionary with class names as keys and counts as values
        """
        timestamp = datetime.datetime.now()
        
        # Reset counts for this update
        for product in self.inventory:
            self.inventory[product] = 0
            
        # Update with new counts
        for class_name, count in detections.items():
            if class_name in self.inventory:
                self.inventory[class_name] = count
                
        # Record history
        history_entry = {
            "timestamp": timestamp,
            **{product: count for product, count in self.inventory.items()}
        }
        self.history.append(history_entry)
        
        # Check for alerts
        self.check_alerts()
        
    def check_alerts(self):
        """Check for products below threshold and generate alerts."""
        self.alerts = []
        for product, count in self.inventory.items():
            if product in PRODUCTS and PRODUCTS[product]["alert"]:
                threshold = PRODUCTS[product]["threshold"]
                if count < threshold:
                    self.alerts.append({
                        "product": PRODUCTS[product]["name"],
                        "count": count,
                        "threshold": threshold,
                        "timestamp": datetime.datetime.now()
                    })
    
    def get_inventory(self):
        """Get current inventory with product display names."""
        return {PRODUCTS[product]["name"]: count 
                for product, count in self.inventory.items() 
                if product in PRODUCTS}
    
    def save_report(self):
        """Save current inventory to CSV file."""
        if not self.history:
            return
            
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(REPORTS_DIR, f"inventory_report_{today}.csv")
        
        df = pd.DataFrame(self.history)
        
        # If file exists, append without headers
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
            
        return filename 