import sys, random, datetime, json, os
import torch
import torch.nn as nn
from PyQt5 import QtWidgets, QtGui, QtCore

# Import matplotlib for the pie chart
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Define the MLP architecture ---
# This architecture is expected to match your saved state dicts:
#   fc1: [64, 14], fc2: [32, 64], fc3: [16, 32], fc4: [1, 16]
class MLP(nn.Module):
    def __init__(self, input_size=14, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# --- PyQt Interface Definition ---
class Lucky28PredictionApp(QtWidgets.QMainWindow):
    def __init__(self, theme='chinese'):
        super().__init__()
        self.setWindowTitle("Lucky 28 Prediction App")
        self.resize(1000, 700)
        self.theme = theme  # 'chinese' or 'indian'
        self.auto_prediction_enabled = False
        
        # JSON file to store all predictions
        self.predictions_file = "latest_predictions.json"
        self.all_predictions = self.load_predictions()

        self.initModels()
        self.initUI()
        self.setupTimer()

    def load_predictions(self):
        """Load stored predictions from JSON and filter out any that don't have the new required keys."""
        required_keys = {"draw_time", "odd_even", "big_small", "final_estimated", "recommended", "odd_even_accuracy", "big_small_accuracy", "total"}
        if os.path.exists(self.predictions_file):
            try:
                with open(self.predictions_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Only use predictions that have all required keys
                        filtered = [pred for pred in data if required_keys.issubset(pred.keys())]
                        return filtered
            except Exception as e:
                print("Error loading predictions file:", e)
        return []

    def save_predictions(self):
        try:
            with open(self.predictions_file, "w") as f:
                json.dump(self.all_predictions, f, indent=4)
        except Exception as e:
            print("Error saving predictions file:", e)

    def initModels(self):
        # Load your trained models using the defined architecture:
        self.odd_even_model = MLP(input_size=14, output_size=1)
        try:
            state_dict = torch.load("models/odd_even_model.pth")
            self.odd_even_model.load_state_dict(state_dict)
            self.odd_even_model.eval()
        except Exception as e:
            print("Error loading odd_even_model:", e)

        self.big_small_model = MLP(input_size=14, output_size=1)
        try:
            state_dict = torch.load("models/big_small_model.pth")
            self.big_small_model.load_state_dict(state_dict)
            self.big_small_model.eval()
        except Exception as e:
            print("Error loading big_small_model:", e)

    def initUI(self):
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)

        # Background styling based on theme
        if self.theme == 'chinese':
            bg_image = "chinese_background.jpg"  # high-quality Chinese themed image
            main_color = "#FF0000"    # Lucky Red
            accent_color = "#FFD700"  # Gold Accent
        elif self.theme == 'indian':
            bg_image = "indian_background.jpg"   # high-quality Indian themed image
            main_color = "#FF9933"    # Saffron
            accent_color = "#138808"  # Deep Green Accent
        else:
            bg_image = ""
            main_color = "#FFFFFF"
            accent_color = "#000000"

        self.central.setStyleSheet(f"""
            QWidget {{
                background-color: {main_color};
            }}
            QPushButton {{
                background-color: {accent_color};
                border: none;
                padding: 8px;
                font-size: 14px;
                color: white;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #FFFFFF;
                color: {accent_color};
            }}
            QLabel {{
                font-size: 16px;
                color: white;
            }}
            QPlainTextEdit {{
                background-color: #333;
                color: #EEE;
            }}
        """)

        if bg_image:
            palette = QtGui.QPalette()
            pixmap = QtGui.QPixmap(bg_image)
            if not pixmap.isNull():
                palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(
                    pixmap.scaled(self.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)))
                self.central.setAutoFillBackground(True)
                self.central.setPalette(palette)
            else:
                print("Warning: Background image not found.")

        self.tabs = QtWidgets.QTabWidget(self.central)
        main_layout = QtWidgets.QVBoxLayout(self.central)
        main_layout.addWidget(self.tabs)

        # Tab 1: Live Predictions
        self.tab_predictions = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_predictions, "Live Predictions")
        self.initPredictionTab()

        # Tab 2: Performance Metrics & Chart
        self.tab_metrics = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_metrics, "Performance Metrics")
        self.initMetricsTab()

        # Tab 3: Settings & Log
        self.tab_settings = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_settings, "Settings & Log")
        self.initSettingsTab()

    def initPredictionTab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_predictions)
        self.table = QtWidgets.QTableWidget(20, 5)
        self.table.setHorizontalHeaderLabels([
            "Draw Time", 
            "Odd/Even (Prediction & Accuracy %)", 
            "Big/Small (Prediction & Accuracy %)",
            "Final Estimated Result", 
            "Recommended Bet"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_predict = QtWidgets.QPushButton("Predict Next Draw")
        self.btn_delete = QtWidgets.QPushButton("Delete Predictions")
        self.btn_refresh = QtWidgets.QPushButton("Refresh Data")
        button_layout.addWidget(self.btn_predict)
        button_layout.addWidget(self.btn_delete)
        button_layout.addWidget(self.btn_refresh)
        layout.addLayout(button_layout)

        self.btn_predict.clicked.connect(self.predictNext)
        self.btn_delete.clicked.connect(self.deletePredictions)
        self.btn_refresh.clicked.connect(self.refreshTable)

    def initMetricsTab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_metrics)
        self.label_breakdown = QtWidgets.QLabel("Group Accuracies: N/A")
        self.label_overall = QtWidgets.QLabel("Overall Accuracy: N/A")
        layout.addWidget(self.label_breakdown)
        layout.addWidget(self.label_overall)

        # Create a matplotlib Figure and embed it in the UI for the pie chart
        self.figure = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def initSettingsTab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_settings)
        settings_box = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QHBoxLayout(settings_box)
        self.btn_change_source = QtWidgets.QPushButton("Change Data Source")
        settings_layout.addWidget(self.btn_change_source)
        self.chk_auto = QtWidgets.QCheckBox("Enable Auto-Prediction (every 3 mins)")
        self.chk_auto.stateChanged.connect(self.toggleAutoPrediction)
        settings_layout.addWidget(self.chk_auto)
        layout.addWidget(settings_box)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Prediction log will appear here...")
        layout.addWidget(self.log_box)

    def setupTimer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.predictNext)

    def toggleAutoPrediction(self, state):
        if state == QtCore.Qt.Checked:
            self.auto_prediction_enabled = True
            self.timer.start(180000)  # 3 minutes
            self.log("Auto-Prediction Enabled")
        else:
            self.auto_prediction_enabled = False
            self.timer.stop()
            self.log("Auto-Prediction Disabled")

    def model_predict(self):
        """
        Uses the loaded models to generate a prediction.
        Generates a 14-dimensional feature vector from three random digits.
        """
        # Draw three random digits (0-9)
        digits = [random.randint(0, 9) for _ in range(3)]
        a, b, c = digits
        s = a + b + c
        p_val = a * b * c
        m_val = s / 3.0
        mx = max(digits)
        mn = min(digits)
        r = mx - mn
        a2 = a * a
        b2 = b * b
        c2 = c * c
        ssq = a2 + b2 + c2
        var = (ssq / 3.0) - (m_val * m_val)
        # Build a 14-dimensional feature vector
        features = [a, b, c, s, p_val, m_val, mx, mn, r, a2, b2, c2, ssq, var]
        tensor_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # shape: [1, 14]

        with torch.no_grad():
            odd_even_logit = self.odd_even_model(tensor_input)
            big_small_logit = self.big_small_model(tensor_input)
            odd_even_prob = torch.sigmoid(odd_even_logit).item() * 100
            big_small_prob = torch.sigmoid(big_small_logit).item() * 100

        # Determine predicted classes using a threshold of 50%
        if odd_even_prob >= 50:
            odd_even_pred = "Odd"
            odd_even_conf = odd_even_prob
        else:
            odd_even_pred = "Even"
            odd_even_conf = 100 - odd_even_prob

        if big_small_prob >= 50:
            big_small_pred = "Big"
            big_small_conf = big_small_prob
        else:
            big_small_pred = "Small"
            big_small_conf = 100 - big_small_prob

        odd_even_str = f"Prediction: {odd_even_pred} ({odd_even_conf:.0f}%)"
        big_small_str = f"Prediction: {big_small_pred} ({big_small_conf:.0f}%)"

        # Choose final prediction based on higher confidence
        if odd_even_conf >= big_small_conf:
            final_estimated = odd_even_str
            chosen_accuracy = odd_even_conf
        else:
            final_estimated = big_small_str
            chosen_accuracy = big_small_conf

        recommended_bet = "Yes" if chosen_accuracy >= 75 else "No"

        return {
            "draw_time": datetime.datetime.now().strftime("%H:%M:%S"),
            "digits": digits,
            "odd_even": odd_even_str,
            "big_small": big_small_str,
            "final_estimated": final_estimated,
            "recommended": recommended_bet,
            "odd_even_accuracy": odd_even_conf,
            "big_small_accuracy": big_small_conf,
            "total": s
        }

    def predictNext(self):
        prediction = self.model_predict()
        self.all_predictions.append(prediction)
        self.save_predictions()  # Save predictions to JSON file
        self.refreshTable()
        self.updateMetrics()
        self.log(f"New prediction at {prediction['draw_time']} (Total: {prediction['total']}, Digits: {prediction['digits']})")

    def deletePredictions(self):
        self.all_predictions.clear()
        self.save_predictions()
        self.log("All predictions have been deleted.")
        self.refreshTable()
        self.updateMetrics()

    def refreshTable(self):
        # Display only the latest 20 predictions in the table
        display_predictions = self.all_predictions[-20:]
        self.table.clearContents()
        for row, prediction in enumerate(reversed(display_predictions)):
            # If a required key is missing, skip that prediction
            if "draw_time" not in prediction:
                continue
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(prediction["draw_time"]))
            item_oe = QtWidgets.QTableWidgetItem(prediction.get("odd_even", "N/A"))
            item_bs = QtWidgets.QTableWidgetItem(prediction.get("big_small", "N/A"))
            self.table.setItem(row, 1, item_oe)
            self.table.setItem(row, 2, item_bs)
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(prediction.get("final_estimated", "N/A")))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(prediction.get("recommended", "N/A")))
            self.applyColorIndicator(item_oe, prediction.get("odd_even_accuracy", 0))
            self.applyColorIndicator(item_bs, prediction.get("big_small_accuracy", 0))

    def applyColorIndicator(self, item, accuracy):
        if accuracy >= 80:
            item.setBackground(QtGui.QColor("green"))
        elif 60 <= accuracy < 80:
            item.setBackground(QtGui.QColor("yellow"))
        else:
            item.setBackground(QtGui.QColor("red"))

    def updateMetrics(self):
        total_predictions = len(self.all_predictions)
        if total_predictions == 0:
            overall_accuracy = 0
        else:
            # Sum only predictions that have the required key
            valid_predictions = [p for p in self.all_predictions if "odd_even_accuracy" in p]
            if valid_predictions:
                overall_accuracy = sum(p["odd_even_accuracy"] for p in valid_predictions) / len(valid_predictions)
            else:
                overall_accuracy = 0

        # Divide predictions into groups of 10 (up to 10 groups)
        groups = [self.all_predictions[i:i+10] for i in range(0, total_predictions, 10)]
        groups = groups[:10]  # Limit to 10 groups
        group_text = " | ".join(
            f"Group {i+1}: {sum(p.get('odd_even_accuracy', 0) for p in grp)/len(grp):.1f}%"
            for i, grp in enumerate(groups) if len(grp) > 0
        )
        self.label_breakdown.setText(f"Group Accuracies: {group_text}")
        self.label_overall.setText(f"Overall Accuracy: {overall_accuracy:.1f}% (Target: 65%+)")
        self.log(f"Metrics updated: Overall accuracy is {overall_accuracy:.1f}%.")

        # --- Update the pie chart ---
        labels = ['Overall Accuracy', 'Remaining']
        sizes = [overall_accuracy, max(0, 100 - overall_accuracy)]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
        self.canvas.draw()

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        self.log_box.appendPlainText(f"{timestamp} {message}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Lucky28PredictionApp(theme='chinese')
    window.show()
    sys.exit(app.exec_())




