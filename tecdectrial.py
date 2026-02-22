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
3. Raw Dataset: Manage raw LLM-jp corpus files.
4. Benchmark: Compare models.
5. Keys: Configure API keys for external services.
"""

import sys
import os
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
import train
import test
import rawdataset
import benchmark

class KeysTab(QWidget):
    def __init__(self):
        super().__init__()
        self.config_file = 'keys_config.json'
        self.load_config()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Configure API Keys for External Services"))
        
        # Gemini Key
        gemini_layout = QVBoxLayout()
        gemini_layout.addWidget(QLabel("Gemini API Key File:"))
        self.gemini_path = QLabel(self.gemini_file or "Not set")
        gemini_layout.addWidget(self.gemini_path)
        btn_gemini = QPushButton("Select Gemini Key File")
        btn_gemini.clicked.connect(self.select_gemini_key)
        gemini_layout.addWidget(btn_gemini)
        layout.addLayout(gemini_layout)
        
        # Groq Key
        groq_layout = QVBoxLayout()
        groq_layout.addWidget(QLabel("Groq API Key File:"))
        self.groq_path = QLabel(self.groq_file or "Not set")
        groq_layout.addWidget(self.groq_path)
        btn_groq = QPushButton("Select Groq Key File")
        btn_groq.clicked.connect(self.select_groq_key)
        groq_layout.addWidget(btn_groq)
        layout.addLayout(groq_layout)
        
        btn_load = QPushButton("Load Keys")
        btn_load.clicked.connect(self.load_keys)
        layout.addWidget(btn_load)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_config(self):
        self.gemini_file = None
        self.groq_file = None
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.gemini_file = config.get('gemini_key_file')
                    self.groq_file = config.get('groq_key_file')
                    
                    # Automatically load keys if files are set
                    if self.gemini_file:
                        try:
                            with open(self.gemini_file, 'r') as f:
                                key = f.read().strip()
                                os.environ['GOOGLE_API_KEY'] = key
                            print(f"Gemini key loaded from {self.gemini_file}")
                        except Exception as e:
                            print(f"Failed to load Gemini key: {e}")
                    
                    if self.groq_file:
                        try:
                            with open(self.groq_file, 'r') as f:
                                key = f.read().strip()
                                os.environ['GROQ_API_KEY'] = key
                            print(f"Groq key loaded from {self.groq_file}")
                        except Exception as e:
                            print(f"Failed to load Groq key: {e}")
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save_config(self):
        config = {
            'gemini_key_file': self.gemini_file,
            'groq_key_file': self.groq_file
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
    
    def select_gemini_key(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Gemini API Key File")
        if file:
            self.gemini_file = file
            self.gemini_path.setText(file)
            self.save_config()
    
    def select_groq_key(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Groq API Key File")
        if file:
            self.groq_file = file
            self.groq_path.setText(file)
            self.save_config()
    
    def load_keys(self):
        gemini_file = self.gemini_file
        groq_file = self.groq_file
        
        if gemini_file:
            try:
                with open(gemini_file, 'r') as f:
                    key = f.read().strip()
                    os.environ['GOOGLE_API_KEY'] = key
                QMessageBox.information(self, "Success", "Gemini key loaded.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load Gemini key: {e}")
        
        if groq_file:
            try:
                with open(groq_file, 'r') as f:
                    key = f.read().strip()
                    os.environ['GROQ_API_KEY'] = key
                QMessageBox.information(self, "Success", "Groq key loaded.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load Groq key: {e}")

class TecDecTrial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Technocratic/Deliberative Classifier - Trial Workbench')
        self.resize(1000, 700)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.trainer_tab = train.TrainerGUI()
        self.tester_tab = test.TesterGUI()
        self.rawdataset_tab = rawdataset.RawDatasetGUI()
        self.bench_tab = benchmark.BenchmarkGUI()
        self.keys_tab = KeysTab()
        
        self.tabs.addTab(self.trainer_tab, "Train / Download")
        self.tabs.addTab(self.tester_tab, "Test / Analyze")
        self.tabs.addTab(self.rawdataset_tab, "Raw Dataset")
        self.tabs.addTab(self.bench_tab, "Benchmark")
        self.tabs.addTab(self.keys_tab, "Keys")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TecDecTrial()
    window.show()
    sys.exit(app.exec())
