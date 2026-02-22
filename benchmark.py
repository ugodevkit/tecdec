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
Technocratic/Deliberative Classifier Benchmark
==============================================

Compare the performance of the trained SpaCy model against other models (HF Transformers, Ollama).
Calculates Accuracy, Precision, Recall, and F1 Score.
"""

import sys
import json
import os
import time
import random
import spacy
import requests
import traceback
from spacy.tokens import DocBin
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    accuracy_score = None
    print('Warning: scikit-learn not found. Metrics will be limited.')

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView, QListWidget, 
                             QComboBox, QLineEdit, QMessageBox, QProgressBar, QSplitter, 
                             QCheckBox, QSpinBox, QDialog, QInputDialog, QFileDialog, QApplication)
from PySide6.QtCore import Qt, QThread, Signal

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print('Warning: matplotlib not found. Chart functionality will be disabled.')

# --- Dialogs ---

class HFDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search Hugging Face")
        self.resize(600, 400)
        self.selected_id = None
        
        layout = QVBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Search models (e.g. 'xnli')...")
        self.txt_search.returnPressed.connect(self.search)
        layout.addWidget(self.txt_search)
        
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self.accept_selection)
        layout.addWidget(self.list)
        
        btn = QPushButton("Search")
        btn.clicked.connect(self.search)
        layout.addWidget(btn)
        
        self.chk_gpu = QCheckBox("Use GPU")
        self.chk_gpu.setChecked(True)
        layout.addWidget(self.chk_gpu)
        
        self.setLayout(layout)
        
    def search(self):
        self.list.clear()
        q = self.txt_search.text()
        try:
            url = f"https://huggingface.co/api/models?search={q}&limit=20&sort=downloads&direction=-1&pipeline_tag=zero-shot-classification"
            resp = requests.get(url, timeout=10).json()
            for m in resp:
                self.list.addItem(m['modelId'])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def accept_selection(self, item):
        self.selected_id = {"model": item.text(), "use_gpu": self.chk_gpu.isChecked()}
        self.accept()

class OllamaDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Ollama Model")
        self.resize(400, 300)
        self.selected_id = None
        layout = QVBoxLayout()
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self.accept_selection)
        layout.addWidget(self.list)
        btn = QPushButton("Refresh")
        btn.clicked.connect(self.refresh)
        layout.addWidget(btn)
        self.setLayout(layout)
        self.refresh()
        
    def refresh(self):
        self.list.clear()
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5).json()
            for m in resp['models']:
                self.list.addItem(m['name'])
        except Exception as e:
            QMessageBox.critical(self, "Error", "Is Ollama running?\n" + str(e))

    def accept_selection(self, item):
        self.selected_id = item.text()
        self.accept()

class GroqDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Groq Model")
        self.resize(400, 300)
        self.selected_id = None
        layout = QVBoxLayout()
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self.accept_selection)
        layout.addWidget(self.list)
        btn = QPushButton("Fetch Models")
        btn.clicked.connect(self.refresh)
        layout.addWidget(btn)
        self.setLayout(layout)
        self.refresh()
        
    def refresh(self):
        self.list.clear()
        key = os.environ.get("GROQ_API_KEY")
        if not key:
             key, ok = QInputDialog.getText(self, "API Key", "Enter Groq API Key:")
             if ok and key:
                 os.environ["GROQ_API_KEY"] = key
             else:
                 return
        try:
            headers = {"Authorization": f"Bearer {key}"}
            resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10).json()
            for m in resp['data']:
                self.list.addItem(m['id'])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def accept_selection(self, item):
        self.selected_id = item.text()
        self.accept()

# --- Chart Dialog ---

class ChartDialog(QDialog):
    def __init__(self, parent, accuracy, f1, precision, recall):
        super().__init__(parent)
        self.setWindowTitle("Benchmark Charts")
        self.resize(1200, 800)
        self.setLayout(QVBoxLayout())
        
        if not HAS_MATPLOTLIB:
            self.layout().addWidget(QLabel("Matplotlib not available. Install matplotlib to view charts."))
            return
        
        # Main horizontal layout for charts and legend
        main_layout = QHBoxLayout()
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Benchmark Results')
        
        self.plot_metric(axes[0, 0], accuracy, 'Accuracy Comparison')
        self.plot_metric(axes[0, 1], f1, 'Macro F1 Score')
        self.plot_metric(axes[1, 0], precision, 'Precision')
        self.plot_metric(axes[1, 1], recall, 'Recall')
        
        canvas = FigureCanvas(fig)
        main_layout.addWidget(canvas)
        
        # Legend panel
        legend_panel = self.create_legend_panel(accuracy)
        main_layout.addWidget(legend_panel)
        
        self.layout().addLayout(main_layout)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        self.layout().addWidget(btn_close)
    
    def create_legend_panel(self, data):
        panel = QWidget()
        panel.setLayout(QVBoxLayout())
        panel.setFixedWidth(300)
        
        title = QLabel("Model Key")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        panel.layout().addWidget(title)
        
        colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#808080', '#000000']
        
        for i, d in enumerate(data):
            item_layout = QHBoxLayout()
            
            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background-color: {colors[i % len(colors)]}; border: 1px solid gray;")
            item_layout.addWidget(swatch)
            
            # Label
            label = QLabel(f"M{i+1}. {d['model']}")
            label.setStyleSheet("font-size: 13px;")
            item_layout.addWidget(label)
            item_layout.addStretch()
            
            item_widget = QWidget()
            item_widget.setLayout(item_layout)
            panel.layout().addWidget(item_widget)
        
        panel.layout().addStretch()
        return panel
    
    def plot_metric(self, ax, data, title):
        if not data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(title)
            return
        models = [f"M{i+1}" for i in range(len(data))]
        values = [d['value'] for d in data]
        colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#808080', '#000000']
        for i, (m, v, c) in enumerate(zip(models, values, colors)):
            ax.bar(m, v, color=c)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)

# --- Add Sample Dialog ---

class AddSampleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Manual Sample")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Text:"))
        self.txt_text = QTextEdit()
        layout.addWidget(self.txt_text)
        layout.addWidget(QLabel("Label:"))
        self.cmb_label = QComboBox()
        self.cmb_label.addItems(["TECHNOCRATIC", "DELIBERATIVE"])
        layout.addWidget(self.cmb_label)
        btn_layout = QHBoxLayout()
        btn_generate = QPushButton("Generate")
        btn_generate.clicked.connect(self.generate_text)
        btn_layout.addWidget(btn_generate)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def generate_text(self):
        import google.generativeai as genai
        prompt, ok = QInputDialog.getText(self, "Generate Text", "Enter prompt for Gemini to generate sample text:")
        if not ok or not prompt.strip():
            return
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            QMessageBox.warning(self, "Error", "GOOGLE_API_KEY not set")
            return
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            response = model.generate_content(prompt)
            generated_text = response.text.strip()
            self.txt_text.setText(generated_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate text: {e}")
    
    def accept(self):
        self.text = self.txt_text.toPlainText().strip()
        self.label = self.cmb_label.currentText()
        if not self.text:
            QMessageBox.warning(self, "Error", "Text cannot be empty.")
            return
        super().accept()

# --- Human Label Dialog ---

class HumanLabelDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Text")
        self.resize(600, 400)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Text:"))
        self.txt_display = QTextEdit()
        self.txt_display.setPlainText(text)
        self.txt_display.setReadOnly(True)
        layout.addWidget(self.txt_display)
        layout.addWidget(QLabel("Label:"))
        self.cmb_label = QComboBox()
        self.cmb_label.addItems(["TECHNOCRATIC", "DELIBERATIVE"])
        layout.addWidget(self.cmb_label)
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def accept(self):
        self.label = self.cmb_label.currentText()
        super().accept()

# --- Observer Dialog ---

class ObserverDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Observer Test for Benchmark Samples")
        self.resize(1000, 700)
        self.setLayout(QVBoxLayout())
        
        # Config Panel
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Sample Size:"))
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(1, 50)
        self.spin_samples.setValue(20)
        config_layout.addWidget(self.spin_samples)
        
        config_layout.addWidget(QLabel("Evaluator:"))
        self.cmb_evaluator = QComboBox()
        self.cmb_evaluator.addItems(["Gemini", "Human"])
        config_layout.addWidget(self.cmb_evaluator)
        
        self.btn_run = QPushButton("Evaluate (LLM-jp data)")
        self.btn_run.clicked.connect(self.run_test)
        config_layout.addWidget(self.btn_run)
        
        self.btn_add = QPushButton("Add Sample")
        self.btn_add.clicked.connect(self.add_sample)
        config_layout.addWidget(self.btn_add)
        
        config_layout.addStretch()
        self.layout().addLayout(config_layout)
        
        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Text Snippet", "Label", "Rationale"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout().addWidget(self.table)
        
        # Progress
        self.progress = QProgressBar()
        self.layout().addWidget(self.progress)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_use = QPushButton("Use for Benchmark")
        self.btn_use.clicked.connect(self.use_for_benchmark)
        self.btn_use.setEnabled(False)
        btn_layout.addWidget(self.btn_use)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        self.layout().addLayout(btn_layout)
        
        self.samples = []
        self.results = []
    
    def run_test(self):
        self.btn_run.setEnabled(False)
        self.table.setRowCount(0)
        self.progress.setValue(0)
        
        # Load samples using selector
        try:
            import selector
            self.samples = selector.get_random_sample(self.spin_samples.value())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load selector: {str(e)}")
            self.btn_run.setEnabled(True)
            return
        
        evaluator = self.cmb_evaluator.currentText()
        self.progress.setMaximum(len(self.samples))
        
        if evaluator == "Gemini":
            self.evaluate_with_gemini()
        elif evaluator == "Human":
            self.evaluate_with_human()
    
    def evaluate_with_gemini(self):
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            QMessageBox.critical(self, "Error", "GOOGLE_API_KEY not set")
            self.btn_run.setEnabled(True)
            return
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        self.results = []
        for i, sample in enumerate(self.samples):
            if isinstance(sample, dict):
                text = sample['text']
                id = sample['id']
            else:
                text = sample
                id = i
            
            prompt = f"""Analyze the following text for its Political Discourse Style.\n\nCategories:\n1. TECHNOCRATIC\n2. DELIBERATIVE\n\nText: {text}\n\nRespond with JSON: {{"label": "TECHNOCRATIC" or "DELIBERATIVE", "rationale": "short explanation"}}"""
            try:
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                if not raw_text:
                    label = 'ERROR'
                    rationale = 'Empty response from Gemini'
                else:
                    def extract_json(text):
                        start = text.find('{')
                        end = text.rfind('}')
                        if start != -1 and end != -1 and end > start:
                            return text[start:end+1]
                        return text
                    
                    json_text = extract_json(raw_text)
                    try:
                        import json
                        result = json.loads(json_text)
                        label = result.get('label', 'UNKNOWN')
                        rationale = result.get('rationale', 'No rationale')
                    except Exception as e:
                        label = 'ERROR'
                        rationale = f'Invalid JSON: {raw_text}'
            except Exception as e:
                label = 'ERROR'
                rationale = str(e)
            
            self.results.append({'id': id, 'text': text, 'label': label, 'rationale': rationale})
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(str(id)))
            self.table.setItem(i, 1, QTableWidgetItem(text[:60] + '...' if len(text) > 60 else text))
            self.table.setItem(i, 2, QTableWidgetItem(label))
            self.table.setItem(i, 3, QTableWidgetItem(rationale))
            self.progress.setValue(i + 1)
            QApplication.processEvents()  # Update UI
        
        self.btn_run.setEnabled(True)
        self.btn_use.setEnabled(True)
    
    def evaluate_with_human(self):
        self.results = []
        for i, sample in enumerate(self.samples):
            if isinstance(sample, dict):
                text = sample['text']
                id = sample['id']
            else:
                text = sample
                id = i
            
            dlg = HumanLabelDialog(text, self)
            if dlg.exec():
                label = dlg.label
                rationale = 'Human labeled'
            else:
                label = 'UNKNOWN'
                rationale = 'Skipped'
            
            self.results.append({'id': id, 'text': text, 'label': label, 'rationale': rationale})
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(str(id)))
            self.table.setItem(i, 1, QTableWidgetItem(text[:60] + '...' if len(text) > 60 else text))
            self.table.setItem(i, 2, QTableWidgetItem(label))
            self.table.setItem(i, 3, QTableWidgetItem(rationale))
            self.progress.setValue(i + 1)
        
        self.btn_run.setEnabled(True)
        self.btn_use.setEnabled(True)
    
    def add_sample(self):
        dlg = AddSampleDialog(self)
        if dlg.exec():
            self.results.append({'id': len(self.results), 'text': dlg.text, 'label': dlg.label, 'rationale': 'Manually added'})
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(len(self.results) - 1)))
            self.table.setItem(row, 1, QTableWidgetItem(dlg.text[:60] + '...' if len(dlg.text) > 60 else dlg.text))
            self.table.setItem(row, 2, QTableWidgetItem(dlg.label))
            self.table.setItem(row, 3, QTableWidgetItem('Manually added'))
            self.btn_use.setEnabled(True)
    
    def use_for_benchmark(self):
        # Ask user if they want to add labeled samples to test_data.jsonl
        reply = QMessageBox.question(self, "Add to Test Data", 
            "Do you want to add the labeled samples to test_data.jsonl?\n\n" +
            "This will create the file if it does not exist and append the samples.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            # Append to test_data.jsonl
            import json
            with open('test_data.jsonl', 'a', encoding='utf-8') as f:
                for r in self.results:
                    if r['label'] in ['TECHNOCRATIC', 'DELIBERATIVE']:
                        f.write(json.dumps({'text': r['text'], 'label': r['label']}) + '\n')
            QMessageBox.information(self, "Added", "Samples added to test_data.jsonl.")
        else:
            # Save to temp file as before
            import json
            with open('observer_benchmark_data.jsonl', 'w') as f:
                for r in self.results:
                    if r['label'] in ['TECHNOCRATIC', 'DELIBERATIVE']:
                        f.write(json.dumps({'text': r['text'], 'label': r['label']}) + '\n')
            QMessageBox.information(self, "Saved", "Data saved to observer_benchmark_data.jsonl. Reload the benchmark script to use it.")
        
        # Automatically run the benchmark
        if hasattr(self.parent(), 'load_data') and hasattr(self.parent(), 'start_benchmark'):
            self.parent().load_data()
            self.parent().start_benchmark()
        
        self.accept()

# --- Evaluators ---

class Evaluator:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.labels = ["TECHNOCRATIC", "DELIBERATIVE"] # Default labels

    def set_labels(self, labels):
        if labels: self.labels = list(labels)

    def predict_batch(self, texts):
        return ["UNKNOWN"] * len(texts)

class SpacyEvaluator(Evaluator):
    def predict_batch(self, texts):
        try:
            path = self.config
            if not os.path.exists(path):
                # Try relative to script
                if os.path.exists(os.path.join("model_output", path)):
                    path = os.path.join("model_output", path)
            
            if not os.path.exists(path):
                 print(f"Error: Model path not found: {path}")
                 return ["ERROR"] * len(texts)
            
            nlp = spacy.load(path)
            preds = []
            # Truncate long texts to avoid Sudachi tokenizer limit
            texts = [t[:40000] if len(t) > 40000 else t for t in texts]
            docs = list(nlp.pipe(texts))
            for doc in docs:
                if doc.cats:
                    preds.append(max(doc.cats, key=doc.cats.get))
                else:
                    preds.append("UNKNOWN")
                    print(f"SpaCy prediction for '{doc.text[:50]}...': No cats found, defaulting to UNKNOWN")
            return preds
        except Exception as e:
            traceback.print_exc()
            print(f"Spacy Evaluator Error: {e}")
            return ["ERROR"] * len(texts)

class HFEvaluator(Evaluator):
    def predict_batch(self, texts):
        try:
            from transformers import pipeline
            if isinstance(self.config, dict):
                model = self.config["model"]
                device = 0 if self.config.get("use_gpu", True) else -1
            else:
                model = self.config
                device = 0
            
            print(f"Loading HF pipeline: {model}...")
            # Assume Zero-Shot for simplicity if NLI model, else text-classification
            # For this generic script, we try Zero-Shot with hardcoded labels for TecDec
            labels = ["TECHNOCRATIC", "DELIBERATIVE"] 
            
            classifier = pipeline("zero-shot-classification", model=model, device=device)
            preds = []
            
            # Process in chunks
            results = classifier(texts, candidate_labels=labels, batch_size=8)
            if not isinstance(results, list): results = [results]
            
            for res in results:
                preds.append(res['labels'][0])
            return preds
        except Exception as e:
            traceback.print_exc()
            print(f"HF Evaluator Error: {e}")
            return ["ERROR"] * len(texts)

class OllamaEvaluator(Evaluator):
    def predict_batch(self, texts):
        preds = []
        url = "http://localhost:11434/api/generate"
        model = self.config
        labels_str = ", ".join(self.labels)
        print(f"Querying Ollama ({model}) with labels: {labels_str}...")
        
        for text in texts:
            prompt = f"Classify this text into exactly one of these categories: [{labels_str}]. Reply ONLY with the category name.\n\nText: {text}"
            try:
                resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=30)
                if resp.status_code == 200:
                    ans = resp.json().get("response", "").strip().upper()
                    found = "UNKNOWN"
                    for lb in self.labels:
                        if lb in ans: 
                            found = lb
                            break
                    preds.append(found)
                else:
                    preds.append("ERROR")
            except Exception as e:
                print(f"Ollama error: {e}")
                preds.append("ERROR")
        return preds

class GroqEvaluator(Evaluator):
    def predict_batch(self, texts):
        preds = []
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
             print("Groq Error: GROQ_API_KEY environment variable not set.")
             return ["ERROR"] * len(texts)

        url = "https://api.groq.com/openai/v1/chat/completions"
        model = self.config
        labels_str = ", ".join(self.labels)
        
        for text in texts:
            prompt = f"Classify this text into exactly one of these categories: [{labels_str}]. Reply ONLY with the category name.\n\nText: {text}"
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 200:
                    ans = resp.json()['choices'][0]['message']['content'].strip().upper()
                    found = "UNKNOWN"
                    for lb in self.labels:
                        if lb in ans: 
                            found = lb
                            break
                    preds.append(found)
                else:
                    print(f"Groq API Error: {resp.text}")
                    preds.append("ERROR")
            except Exception as e:
                print(f"Groq error: {e}")
                preds.append("ERROR")
        return preds

# --- Worker ---

class BenchmarkWorker(QThread):
    progress_signal = Signal(int, int)
    result_signal = Signal(int, float, float, float, float, str)
    finished_signal = Signal()
    log_signal = Signal(str)

    def __init__(self, evaluators, test_data, limit):
        super().__init__()
        if limit > 0:
            shuffled = test_data[:]
            random.shuffle(shuffled)
            self.test_data = shuffled[:limit]
        else:
            self.test_data = test_data
        self.evaluators = evaluators
        self.limit = limit

    def run(self):
        texts = [d['text'] for d in self.test_data]
        actuals = [d['label'] for d in self.test_data]
        
        # Detect unique labels from dataset
        unique_labels = sorted(list(set(actuals)))
        self.log_signal.emit(f"Starting benchmark on {len(texts)} samples...")
        self.log_signal.emit(f"Detected labels: {unique_labels}")

        for idx, ev in enumerate(self.evaluators):
            # Only override if we detected a sufficient number of labels to be useful
            if len(unique_labels) >= 2:
                ev.set_labels(unique_labels)
            
            self.log_signal.emit(f"Running {ev.name}...")
            self.progress_signal.emit(idx, len(self.evaluators))
            
            try:
                start = time.time()
                preds = ev.predict_batch(texts)
                duration = time.time() - start
                
                if accuracy_score:
                    acc = accuracy_score(actuals, preds)
                    p, r, f1, _ = precision_recall_fscore_support(actuals, preds, average='macro', zero_division=0)
                else:
                    # Fallback simple accuracy
                    correct = sum(1 for i in range(len(actuals)) if actuals[i] == preds[i])
                    acc = correct / len(actuals)
                    p, r, f1 = 0.0, 0.0, 0.0

                self.result_signal.emit(idx, acc, f1, p, r, "Done")
                self.log_signal.emit(f"Finished {ev.name}: Acc={acc:.4f}, Time={duration:.2f}s")
                
            except Exception as e:
                self.log_signal.emit(f"Error in {ev.name}: {e}")
                self.result_signal.emit(idx, 0, 0, 0, 0, "Error")

        self.finished_signal.emit()

# --- GUI ---

class BenchmarkGUI(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Config Area
        config_layout = QHBoxLayout()
        
        # Model List
        self.model_list = QListWidget()
        # Default models
        # self.add_model("Exported Model (Best)", "spacy", "model-best")
        config_layout.addWidget(self.model_list)
        
        # Add Model Controls
        ctrl_layout = QVBoxLayout()
        
        btn_add_local = QPushButton("Add Local SpaCy")
        btn_add_local.clicked.connect(self.add_local)
        ctrl_layout.addWidget(btn_add_local)
        
        btn_add_hf = QPushButton("Search HF")
        btn_add_hf.clicked.connect(self.add_hf)
        ctrl_layout.addWidget(btn_add_hf)
        
        btn_add_ollama = QPushButton("Scan Ollama")
        btn_add_ollama.clicked.connect(self.add_ollama)
        ctrl_layout.addWidget(btn_add_ollama)
        
        btn_add_groq = QPushButton("Scan Groq")
        btn_add_groq.clicked.connect(self.add_groq)
        ctrl_layout.addWidget(btn_add_groq)
        
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(lambda: self.model_list.takeItem(self.model_list.currentRow()))
        ctrl_layout.addWidget(btn_remove)
        
        ctrl_layout.addStretch()
        config_layout.addLayout(ctrl_layout)
        layout.addLayout(config_layout)
        
        # Run Controls
        run_layout = QHBoxLayout()
        run_layout.addWidget(QLabel("Limit Samples:"))
        self.spin_limit = QSpinBox()
        self.spin_limit.setRange(0, 10000)
        self.spin_limit.setValue(20)
        run_layout.addWidget(self.spin_limit)
        
        self.btn_run = QPushButton("Run Benchmark")
        self.btn_run.clicked.connect(self.start_benchmark)
        run_layout.addWidget(self.btn_run)
        
        self.btn_observer = QPushButton("Get/Generate Samples")
        self.btn_observer.clicked.connect(self.open_observer)
        run_layout.addWidget(self.btn_observer)
        
        self.btn_chart = QPushButton("Show Chart")
        self.btn_chart.clicked.connect(self.show_chart)
        self.btn_chart.setEnabled(False)
        run_layout.addWidget(self.btn_chart)
        
        layout.addLayout(run_layout)
        
        # Results Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Model", "Acc", "F1", "Prec", "Rec", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.setLayout(layout)
        self.worker = None
        self.data = []
        self.load_data()

    def add_model(self, name, type_, config):
        item = f"{name} [{type_}]"
        self.model_list.addItem(item)
        # Store config in item data
        last_item = self.model_list.item(self.model_list.count() - 1)
        last_item.setData(Qt.UserRole, config)

    def add_local(self):
        d = QFileDialog.getExistingDirectory(self, "Select SpaCy Model Directory")
        if d:
            self.add_model("Custom SpaCy", "spacy", d)

    def add_hf(self):
        dlg = HFDialog(self)
        if dlg.exec():
            self.add_model(dlg.selected_id["model"] + " (HF)", "hf", dlg.selected_id)

    def add_ollama(self):
        dlg = OllamaDialog(self)
        if dlg.exec():
            self.add_model(dlg.selected_id + " (Ollama)", "ollama", dlg.selected_id)

    def add_groq(self):
        dlg = GroqDialog(self)
        if dlg.exec():
            self.add_model(dlg.selected_id + " (Groq)", "groq", dlg.selected_id)

    def load_data(self):
        target = "test_data.jsonl"
        if not os.path.exists(target):
            target = "training_data.jsonl" # Fallback
            
        if os.path.exists(target):
            with open(target, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
            print(f"Loaded {len(self.data)} samples from {target}")

    def start_benchmark(self):
        if not self.data:
            QMessageBox.warning(self, "No Data", "No data found (looked for test_data.jsonl or training_data.jsonl).")
            return
            
        self.btn_run.setEnabled(False)
        evaluators = []
        self.table.setRowCount(0)
        
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            txt = item.text()
            config = item.data(Qt.UserRole)
            
            # Parse txt for name and type
            start = txt.rfind("[")
            end = txt.rfind("]")
            if start == -1 or end == -1:
                continue
            type_str = txt[start+1:end]
            name = txt[:start].strip()
            
            if type_str == "spacy": evaluators.append(SpacyEvaluator(name, config))
            elif type_str == "hf": evaluators.append(HFEvaluator(name, config))
            elif type_str == "ollama": evaluators.append(OllamaEvaluator(name, config))
            elif type_str == "groq": evaluators.append(GroqEvaluator(name, config))
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 5, QTableWidgetItem("Pending..."))
            
        self.worker = BenchmarkWorker(evaluators, self.data, self.spin_limit.value())
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.result_signal.connect(self.on_result)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        
    def on_result(self, row, acc, f1, p, r, status):
        self.table.setItem(row, 1, QTableWidgetItem(f"{acc:.4f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{f1:.4f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{p:.4f}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{r:.4f}"))
        self.table.setItem(row, 5, QTableWidgetItem(status))
        
    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_chart.setEnabled(True)
        self.progress.setValue(self.progress.maximum())
        QMessageBox.information(self, "Done", "Benchmark Completed")
    
    def open_observer(self):
        dlg = ObserverDialog(self)
        dlg.exec()
    
    def show_chart(self):
        accuracy = []
        f1 = []
        precision = []
        recall = []
        for row in range(self.table.rowCount()):
            model = self.table.item(row, 0).text()
            acc = float(self.table.item(row, 1).text())
            f1_val = float(self.table.item(row, 2).text())
            prec = float(self.table.item(row, 3).text())
            rec = float(self.table.item(row, 4).text())
            accuracy.append({'model': model, 'value': acc})
            f1.append({'model': model, 'value': f1_val})
            precision.append({'model': model, 'value': prec})
            recall.append({'model': model, 'value': rec})
        dlg = ChartDialog(self, accuracy, f1, precision, recall)
        dlg.exec()

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = BenchmarkGUI()
    window.show()
    sys.exit(app.exec())
