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
Technocratic/Deliberative Classifier Tester
===========================================

This PySide6 GUI application allows users to test trained SpaCy text classification models.

Features:
1. Model Loading:
   - From local directory (e.g., ./model-best)
   - From installed Python packages (pip install <model.whl>)
2. Input Methods:
   - Manual text entry
   - File loading (*.txt)
   - Random sampling from LLM-jp corpus (via selector.py)
3. Analysis:
   - Displays classification categories and scores.
   - Extracts and lists named entities.

Usage:
Run this script directly or via the workbench (tecdectrial.py).
"""

import sys
import os
import spacy
import selector
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QMessageBox, QTextEdit, QFileDialog, QRadioButton, QButtonGroup, QProgressDialog)
from PySide6.QtCore import Qt, QThread, Signal

class SampleFetcher(QThread):
    finished_signal = Signal(str)
    
    def __init__(self, active_mode=False):
        super().__init__()
        self.active_mode = active_mode

    def run(self):
        try:
            text = selector.get_random_sample(active_mode=self.active_mode)
            self.finished_signal.emit(text)
        except Exception as e:
            self.finished_signal.emit(f"Error: {str(e)}")

class TesterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Tester')
        self.resize(600, 500)
        self.nlp = None
        
        layout = QVBoxLayout()
        
        # --- Model Selection ---
        layout.addWidget(QLabel('<b>1. Select Model Source:</b>'))
        self.radio_group = QButtonGroup()
        
        self.rb_dir = QRadioButton('Load from directory (./model_output/model-best or ./model-best)')
        self.rb_dir.setChecked(True)
        self.radio_group.addButton(self.rb_dir)
        layout.addWidget(self.rb_dir)
        
        self.rb_pkg = QRadioButton('Load from installed package (pip install .whl)')
        self.radio_group.addButton(self.rb_pkg)
        layout.addWidget(self.rb_pkg)
        
        self.btn_load = QPushButton('Load Model')
        self.btn_load.clicked.connect(self.load_model)
        layout.addWidget(self.btn_load)
        
        self.lbl_status = QLabel('Status: No model loaded')
        layout.addWidget(self.lbl_status)
        
        layout.addWidget(QLabel('------------------------------------------------'))
        
        # --- Input ---
        layout.addWidget(QLabel('<b>2. Input Text:</b>'))
        
        input_btns = QHBoxLayout()
        self.btn_file = QPushButton('Load from File...')
        self.btn_file.clicked.connect(self.load_file)
        input_btns.addWidget(self.btn_file)
        
        self.btn_random = QPushButton('Fetch Random from LLM-jp')
        self.btn_random.clicked.connect(lambda: self.fetch_random(active_mode=False))
        input_btns.addWidget(self.btn_random)
        
        input_btns.addStretch()
        layout.addLayout(input_btns)
        
        self.txt_input = QTextEdit()
        self.txt_input.setPlaceholderText('Type or paste text here...')
        layout.addWidget(self.txt_input)
        
        self.btn_test = QPushButton('Analyze')
        self.btn_test.clicked.connect(self.analyze)
        self.btn_test.setEnabled(False)
        layout.addWidget(self.btn_test)
        
        # --- Output ---
        layout.addWidget(QLabel('<b>3. Analysis Result:</b>'))
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        layout.addWidget(self.txt_output)
        
        self.setLayout(layout)

    def load_model(self):
        try:
            if self.rb_dir.isChecked():
                # Check common paths
                paths = ['./model_output/model-best', './model-best']
                found = None
                for p in paths:
                    if os.path.exists(p):
                        found = p
                        break
                
                if found:
                    self.nlp = spacy.load(found)
                    self.lbl_status.setText(f'Status: Loaded from {found}')
                else:
                    # Ask user to pick
                    d = QFileDialog.getExistingDirectory(self, 'Select Model Directory (model-best)')
                    if d:
                        self.nlp = spacy.load(d)
                        self.lbl_status.setText(f'Status: Loaded from {d}')
                    else:
                        return
            else:
                # Load via package name - requires input or config
                # Assuming standard naming convention or ask user
                text, ok = QFileDialog.getOpenFileName(self, 'Select Wheel File (optional) or Cancel to type name', filter='*.whl')
                model_name = ''
                if ok and text:
                    QMessageBox.information(self, 'Info', 'Please install the wheel via pip first!\n\npip install ' + text)
                    return
                else:
                    # Default fallback or hardcoded name from export
                    # In a real scenario we might know the package name from ModelExporter
                    from PySide6.QtWidgets import QInputDialog
                    name, ok = QInputDialog.getText(self, 'Package Name', 'Enter installed model package name (e.g. ja_pipeline):')
                    if ok and name:
                         self.nlp = spacy.load(name)
                         self.lbl_status.setText(f'Status: Loaded package {name}')
                    else:
                         return
            
            self.btn_test.setEnabled(True)
            QMessageBox.information(self, 'Success', 'Model loaded successfully!')

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load model: {str(e)}')
            self.lbl_status.setText('Status: Load failed')
            self.btn_test.setEnabled(False)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Text File', filter='Text files (*.txt);;All files (*)')
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    self.txt_input.setText(f.read())
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Could not read file: {e}')

    def fetch_random(self, active_mode=False):
        if active_mode:
            reply = QMessageBox.warning(self, 'Large Download Warning',
                                      "You are about to download a single file from the LLM-jp corpus.\n" +
                                      "These files are approximately 1GB in size.\n\n" +
                                      "This may take several minutes depending on your internet connection.",
                                      QMessageBox.Ok | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return

        self.btn_random.setEnabled(False)
        msg = "Fetching random sample (Local Cache)..." if not active_mode else "Downloading ~1GB file from LFS (Please Wait)..."
        self.progress = QProgressDialog(msg, "Cancel", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        
        self.fetcher = SampleFetcher(active_mode)
        self.fetcher.finished_signal.connect(lambda t: self.on_fetch_finished(t, active_mode))
        self.fetcher.start()

    def on_fetch_finished(self, text, was_active):
        self.btn_random.setEnabled(True)
        self.progress.close()
        
        if text.startswith("Error:"):
             if not was_active:
                 # Conservative failed, ask user to go active
                 reply = QMessageBox.question(self, 'No Local Data', 
                        "No locally downloaded samples found. Do you want to download a file from the repository? (Requires Internet)",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                 if reply == QMessageBox.Yes:
                     self.fetch_random(active_mode=True)
             else:
                 QMessageBox.critical(self, 'Error', text)
        else:
             self.txt_input.setText(text)

    def analyze(self):
        if not self.nlp:
             return
        
        text = self.txt_input.toPlainText()
        if not text.strip():
             return
        
        try:
            doc = self.nlp(text)
            
            result = '<b>Text Classification (Categories):</b><br>'
            if doc.cats:
                # Sort categories by score
                sorted_cats = sorted(doc.cats.items(), key=lambda item: item[1], reverse=True)
                for cat, score in sorted_cats:
                    result += f'{cat}: {score:.4f}<br>'
            else:
                result += 'No categories found.<br>'
            
            result += '<br><b>Entities:</b><br>'
            if doc.ents:
                 for ent in doc.ents:
                     result += f'{ent.text} ({ent.label_})<br>'
            else:
                 result += 'No entities found.<br>'

            self.txt_output.setHtml(result)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Analysis failed: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TesterGUI()
    window.show()
    sys.exit(app.exec())
