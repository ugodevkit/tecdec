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
Raw Dataset Manager
===================

This tool manages the raw LLM-jp Corpus data files.
It provides a file tree view of the repository, allowing users to:
- View which files are downloaded vs. pointers
- Download specific files on demand
- Add new files to the local cache

Dependencies: selector.py
"""

import sys
import os
import selector
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QTreeWidget, QTreeWidgetItem, QMessageBox, QProgressDialog, QSplitter)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

class FileFetcher(QThread):
    finished_signal = Signal(bool, str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            selector.fetch_lfs_file(self.file_path)
            self.finished_signal.emit(True, f"Downloaded {self.file_path.name}")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class RawDatasetGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Raw Dataset Manager')
        self.resize(800, 600)
        layout = QVBoxLayout()
        
        # Top controls
        controls = QHBoxLayout()
        
        btn_refresh = QPushButton('Refresh Tree')
        btn_refresh.clicked.connect(self.refresh_tree)
        controls.addWidget(btn_refresh)
        
        btn_download_selected = QPushButton('Download Selected')
        btn_download_selected.clicked.connect(self.download_selected)
        controls.addWidget(btn_download_selected)
        
        btn_add_random = QPushButton('Add Random File')
        btn_add_random.clicked.connect(self.add_random_file)
        controls.addWidget(btn_add_random)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['File', 'Status', 'Size'])
        self.tree.setColumnWidth(0, 400)
        self.tree.setColumnWidth(1, 100)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.tree)
        
        # Status label
        self.status_label = QLabel('Status: Ready')
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        self.refresh_tree()
    
    def refresh_tree(self):
        self.tree.clear()
        self.status_label.setText('Status: Refreshing...')
        
        try:
            selector.ensure_repo()
            ja_dir = selector.REPO_DIR / 'ja'
            if not ja_dir.exists():
                self.status_label.setText('Status: Repository not initialized')
                return
            
            # Build tree structure
            root_item = QTreeWidgetItem(['ja', '', ''])
            self.tree.addTopLevelItem(root_item)
            self.total_files = 0
            self.downloaded = 0
            self.populate_tree(root_item, ja_dir)
            root_item.setExpanded(True)
            
            self.status_label.setText(f'Status: Tree refreshed. {self.downloaded}/{self.total_files} files downloaded.')
        except Exception as e:
            self.status_label.setText(f'Status: Error - {e}')
    
    def populate_tree(self, parent_item, dir_path):
        try:
            for item in sorted(dir_path.iterdir()):
                if item.is_file() and item.suffix == '.gz':
                    status = 'Downloaded' if not selector.is_lfs_pointer(item) else 'Pointer'
                    size = f"{item.stat().st_size} bytes" if item.exists() else 'N/A'
                    child = QTreeWidgetItem([item.name, status, size])
                    if status == 'Pointer':
                        child.setForeground(1, Qt.red)
                    else:
                        child.setForeground(1, Qt.green)
                    parent_item.addChild(child)
                    self.total_files += 1
                    if status == 'Downloaded':
                        self.downloaded += 1
                elif item.is_dir():
                    child = QTreeWidgetItem([item.name, 'Directory', ''])
                    parent_item.addChild(child)
                    self.populate_tree(child, item)
        except Exception as e:
            print(f"Error populating tree: {e}")
    
    def download_selected(self):
        selected = self.tree.selectedItems()
        if not selected:
            QMessageBox.warning(self, 'No Selection', 'Please select a file to download.')
            return
        
        item = selected[0]
        if item.text(1) == 'Directory':
            QMessageBox.warning(self, 'Invalid Selection', 'Cannot download directories.')
            return
        
        file_name = item.text(0)
        # Reconstruct path
        path_parts = []
        current = item
        while current:
            path_parts.insert(0, current.text(0))
            current = current.parent()
        
        rel_path = Path(*path_parts[1:])  # Skip 'ja'
        full_path = selector.REPO_DIR / 'ja' / rel_path
        
        self.status_label.setText(f'Status: Downloading {file_name}...')
        self.progress = QProgressDialog(f"Downloading {file_name}...", "Cancel", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        
        self.fetcher = FileFetcher(full_path)
        self.fetcher.finished_signal.connect(lambda success, msg: self.on_download_finished(success, msg, item))
        self.fetcher.start()
    
    def on_download_finished(self, success, msg, item):
        self.progress.close()
        if success:
            item.setText(1, 'Downloaded')
            item.setForeground(1, Qt.green)
            self.downloaded += 1
            self.status_label.setText(f'Status: Ready. {self.downloaded}/{self.total_files} files downloaded.')
            QMessageBox.information(self, 'Success', msg)
        else:
            QMessageBox.critical(self, 'Error', msg)
            self.status_label.setText(f'Status: Ready. {self.downloaded}/{self.total_files} files downloaded.')
    
    def add_random_file(self):
        try:
            selector.ensure_repo()
            ja_dir = selector.REPO_DIR / 'ja'
            files = list(ja_dir.rglob('*.jsonl.gz'))
            pointers = [f for f in files if selector.is_lfs_pointer(f)]
            if not pointers:
                QMessageBox.information(self, 'All Downloaded', 'All files are already downloaded.')
                return
            
            import random
            selected_file = random.choice(pointers)
            self.status_label.setText(f'Status: Downloading random file {selected_file.name}...')
            self.progress = QProgressDialog(f"Downloading {selected_file.name}...", "Cancel", 0, 0, self)
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.show()
            
            self.fetcher = FileFetcher(selected_file)
            self.fetcher.finished_signal.connect(lambda success, msg: self.on_random_download_finished(success, msg))
            self.fetcher.start()
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
    
    def on_random_download_finished(self, success, msg):
        self.progress.close()
        if success:
            QMessageBox.information(self, 'Success', msg)
            self.refresh_tree()
        else:
            QMessageBox.critical(self, 'Error', msg)
            self.status_label.setText(f'Status: Ready. {self.downloaded}/{self.total_files} files downloaded.')
    
    def on_item_double_clicked(self, item, column):
        if item.text(1) == 'Pointer':
            self.download_selected()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RawDatasetGUI()
    window.show()
    sys.exit(app.exec())
