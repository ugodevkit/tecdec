# Copyright (C) 2026  Falconio Ugo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org>.

"""
Technocratic/Deliberative Classifier Workbench
==============================================

This is the main entry point for the workbench application.
It combines the Training/Download tools and the Testing/Analysis tools into a single tabbed interface.

Tabs:
1. Train / Download: Manage data assets and retrain models.
2. Test / Analyze: Load models and classify text.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
import train
import test

class TecDecTrial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Technocratic/Deliberative Classifier - Trial Workbench')
        self.resize(800, 600)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.trainer_tab = train.TrainerGUI()
        self.tester_tab = test.TesterGUI()
        
        self.tabs.addTab(self.trainer_tab, "Train / Download")
        self.tabs.addTab(self.tester_tab, "Test / Analyze")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TecDecTrial()
    window.show()
    sys.exit(app.exec())
