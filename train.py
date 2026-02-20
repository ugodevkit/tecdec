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
Technocratic/Deliberative Classifier Trainer
============================================

This tool handles the full lifecycle of the SpaCy text classification model outside the Java environment.

Capabilities:
1. Asset Management:
   - Download training data (.spacy) and config from Hugging Face.
   - Download pre-trained models (unpacked or wheel) from Hugging Face.
2. Data Generation:
   - Convert raw JSONL data into SpaCy binary format (.spacy).
   - Splits data into training (80%) and development (20%) sets.
3. Training:
   - Downloads the required base language model (e.g., ja_core_news_lg).
   - Executes the SpaCy training pipeline using `config.cfg`.

Note: Requires internet access for downloading assets and base models.
"""

import sys
import os
import json
import random
import spacy
import subprocess
from spacy.tokens import DocBin
from huggingface_hub import hf_hub_download
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QProgressDialog, QTextEdit
from PySide6.QtCore import QThread, Signal, Qt

BASE_MODEL = 'ja_core_news_lg'
MAX_TEXT_LENGTH = 30000

DATASET_REPO = 'ugo86/ja_tecdec_labels'
MODEL_REPO = 'ugo86/ja_tecdec_labeler'

class Worker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def run(self):
        try:
            if self.mode == 'download_data':
                self.download_data()
                self.finished_signal.emit(True, 'Datasets downloaded successfully.')
                return
            elif self.mode == 'download_model':
                self.download_model()
                self.finished_signal.emit(True, 'Model downloaded successfully.')
                return
            elif self.mode == 'download_wheel':
                self.download_wheel()
                self.finished_signal.emit(True, 'Wheel downloaded successfully.')
                return
            elif self.mode == 'generate':
                self.generate_spacy()
            
            self.run_training()
            self.finished_signal.emit(True, 'Process completed successfully.')
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def download_data(self):
        self.log_signal.emit(f'Downloading config from {MODEL_REPO}...')
        try:
            hf_hub_download(repo_id=MODEL_REPO, filename='config.cfg', local_dir='.', local_dir_use_symlinks=False)
            self.log_signal.emit('Downloaded config.cfg')
        except Exception as e:
            self.log_signal.emit(f'Warning: Could not download config.cfg: {e}')

        self.log_signal.emit(f'Downloading datasets from {DATASET_REPO}...')
        files = ['train.spacy', 'dev.spacy', 'training_data.jsonl']
        for f in files:
            try:
                hf_hub_download(repo_id=DATASET_REPO, filename=f, repo_type='dataset', local_dir='.', local_dir_use_symlinks=False)
                self.log_signal.emit(f'Downloaded {f}')
            except Exception as e:
                self.log_signal.emit(f'Warning: Could not download {f}: {e}')

    def download_model(self):
        self.log_signal.emit(f'Downloading unpacked model from {MODEL_REPO}...')
        # Common Spacy model artifacts
        artifacts = ['config.cfg', 'meta.json', 'tokenizer', 'vocab', 'textcat']
        # snapshot_download might be easier but let's stick to specific files/folders if possible or just snapshot
        # Using snapshot_download to get the full folder structure
        from huggingface_hub import snapshot_download
        local_dir = os.path.join(os.getcwd(), 'model-best')
        snapshot_download(repo_id=MODEL_REPO, local_dir=local_dir, local_dir_use_symlinks=False, ignore_patterns=['*.whl', '.git*'])
        self.log_signal.emit(f'Model downloaded to {local_dir}')

    def download_wheel(self):
        self.log_signal.emit(f'Searching for .whl in {MODEL_REPO}...')
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id=MODEL_REPO)
        wheels = [f for f in files if f.endswith('.whl')]
        if not wheels:
            raise Exception('No .whl file found in repository.')
        
        target_whl = wheels[0]
        self.log_signal.emit(f'Downloading {target_whl}...')
        hf_hub_download(repo_id=MODEL_REPO, filename=target_whl, local_dir='.', local_dir_use_symlinks=False)
        self.log_signal.emit(f'Downloaded {target_whl}')

    def generate_spacy(self):
        self.log_signal.emit('Loading data from training_data.jsonl...')
        nlp = spacy.blank('ja')
        db_train = DocBin()
        db_dev = DocBin()
        
        categories = set()
        data_buffer = []
        
        if not os.path.exists('training_data.jsonl'):
            raise Exception('training_data.jsonl not found!')

        with open('training_data.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                categories.add(entry['label'])
                data_buffer.append(entry)
        
        self.log_signal.emit(f'Found categories: {categories}')
        
        count = 0
        for entry in data_buffer:
            text = entry['text']
            label = entry['label']
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            
            try:
                doc = nlp.make_doc(text)
                doc.cats = {cat: 0 for cat in categories}
                doc.cats[label] = 1
                
                if count % 5 == 0:
                    db_dev.add(doc)
                else:
                    db_train.add(doc)
                count += 1
            except Exception as e:
                self.log_signal.emit(f'Skipping document due to error: {e}')
                self.log_signal.emit(f'Text length: {len(text)}, Snippet: {text[:50]}...')
        
        db_train.to_disk('./train.spacy')
        db_dev.to_disk('./dev.spacy')
        self.log_signal.emit(f'Created train.spacy ({len(db_train)}) and dev.spacy ({len(db_dev)})')

    def run_training(self):
        if not os.path.exists('train.spacy') or not os.path.exists('dev.spacy') or not os.path.exists('config.cfg'):
            raise Exception('Missing train.spacy, dev.spacy or config.cfg')
        
        self.log_signal.emit(f'Downloading base model {BASE_MODEL}...')
        self.run_command(f'python -m spacy download {BASE_MODEL}')
        
        cmd = 'python -m spacy train config.cfg --paths.train ./train.spacy --paths.dev ./dev.spacy --output ./model_output'
        self.log_signal.emit(f'Starting training: {cmd}')
        self.run_command(cmd)

    def run_command(self, cmd):
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            self.log_signal.emit(line.strip())
        
        process.wait()
        if process.returncode != 0:
            raise Exception(f'Command failed with return code {process.returncode}')

class TrainerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Trainer')
        self.resize(500, 400)
        layout = QVBoxLayout()
        
        self.label = QLabel('Choose training mode:')
        layout.addWidget(self.label)
        
        self.btn_download = QPushButton('0a. Download Data (train.spacy/dev.spacy)')
        self.btn_download.clicked.connect(lambda: self.start_worker('download_data'))
        layout.addWidget(self.btn_download)
        
        self.btn_model = QPushButton('0b. Download Model (model-best)')
        self.btn_model.clicked.connect(lambda: self.start_worker('download_model'))
        layout.addWidget(self.btn_model)
        
        self.btn_wheel = QPushButton('0c. Download Wheel (.whl)')
        self.btn_wheel.clicked.connect(lambda: self.start_worker('download_wheel'))
        layout.addWidget(self.btn_wheel)
        
        self.btn_jsonl = QPushButton('1. Generate .spacy from JSONL and train')
        self.btn_jsonl.clicked.connect(lambda: self.start_worker('generate'))
        layout.addWidget(self.btn_jsonl)
        
        self.btn_existing = QPushButton('2. Train from existing .spacy files')
        self.btn_existing.clicked.connect(lambda: self.start_worker('train'))
        layout.addWidget(self.btn_existing)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        
        self.setLayout(layout)
        self.worker = None

    def start_worker(self, mode):
        self.btn_download.setEnabled(False)
        self.btn_model.setEnabled(False)
        self.btn_wheel.setEnabled(False)
        self.btn_jsonl.setEnabled(False)
        self.btn_existing.setEnabled(False)
        self.log_area.clear()
        
        self.worker = Worker(mode)
        self.worker.log_signal.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, message):
        self.btn_download.setEnabled(True)
        self.btn_model.setEnabled(True)
        self.btn_wheel.setEnabled(True)
        self.btn_jsonl.setEnabled(True)
        self.btn_existing.setEnabled(True)
        if success:
            QMessageBox.information(self, 'Success', message)
        else:
            QMessageBox.critical(self, 'Error', message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrainerGUI()
    window.show()
    sys.exit(app.exec())
