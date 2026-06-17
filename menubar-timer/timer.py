"""python setup.py py2app
"""

import rumps
from datetime import datetime
import json
import os
from pathlib import Path
from appdirs import user_data_dir

class DaysUntilApp(rumps.App):
    def __init__(self):
        super(DaysUntilApp, self).__init__("Days Until")
        self.target_date = self.load_target_date()
        self.update_title()
        self.timer = rumps.Timer(self.update_title, 3600)  # Update every hour
        self.timer.start()

    def get_data_file_path(self):
        # Get the standard macOS app data directory
        app_name = "DaysUntil"
        data_dir = user_data_dir(app_name, appauthor=False)
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'target_date.json')

    def load_target_date(self):
        try:
            data_file = self.get_data_file_path()
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data['target_date'])
        except Exception:
            pass
        return datetime(2025, 7, 30)  # Default target date

    def save_target_date(self):
        try:
            data_file = self.get_data_file_path()
            with open(data_file, 'w') as f:
                json.dump({
                    'target_date': self.target_date.isoformat()
                }, f)
        except Exception as e:
            rumps.notification(
                title='Error',
                subtitle='Failed to save target date',
                message=str(e)
            )

    @rumps.clicked("Set Target Date")
    def set_target_date(self, _):
        window = rumps.Window(
            message='Enter target date (YYYY-MM-DD):',
            title='Set Target Date',
            default_text=self.target_date.strftime('%Y-%m-%d')
        )
        response = window.run()
        if response.clicked:
            try:
                new_date = datetime.strptime(response.text, '%Y-%m-%d')
                self.target_date = new_date
                self.save_target_date()
                self.update_title()
            except ValueError:
                rumps.notification(
                    title='Invalid Date',
                    subtitle='Please use YYYY-MM-DD format',
                    message='Example: 2025-07-30'
                )

    def update_title(self, _=None):
        now = datetime.now()
        delta = self.target_date - now
        days_remaining = delta.days
        self.title = f"{days_remaining} D"

if __name__ == "__main__":
    DaysUntilApp().run()

