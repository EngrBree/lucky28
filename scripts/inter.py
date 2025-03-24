
import subprocess
import sys, requests, json
import time
import torch
import torch.nn as nn
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, pyqtSignal, QRunnable, QThreadPool, QObject

API_URL = "http://127.0.0.1:8000/predict/"  # FastAPI endpoint

class FastAPIServer(QThread):
    started_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.fastapi_process = None

    def run(self):
        self.started_signal.emit("🚀 Starting FastAPI server...")
        if self.is_fastapi_running():
            self.started_signal.emit("✅ FastAPI is already running!")
            return
        try:
            self.fastapi_process = subprocess.Popen(
                ["-m","uvicorn", "lucky28API.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)
            if self.is_fastapi_running():
                self.started_signal.emit("✅ FastAPI started successfully!")
            else:
                self.started_signal.emit("❌ FastAPI failed to start.")
        except Exception as e:
            self.started_signal.emit(f"❌ FastAPI Error: {str(e)}")

    def is_fastapi_running(self):
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=10)
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    def stop_server(self):
        if self.fastapi_process:
            self.fastapi_process.terminate()
            self.fastapi_process = None
            self.started_signal.emit("🔴 FastAPI server stopped.")

class WorkerSignals(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

class PredictionWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    def run(self):
        try:
            response = requests.post(API_URL)
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    self.signals.error.emit(f"❌ API Error: {data['error']}")
                else:
                    self.signals.finished.emit(data)
            else:
                self.signals.error.emit(f"❌ API Request Failed: {response.status_code}")
        except Exception as e:
            self.signals.error.emit(f"❌ Connection Error: {str(e)}")

class Lucky28PredictionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lucky 28 Prediction App")
        self.resize(1000, 700)
        self.threadpool = QThreadPool()
        self.initUI()

        self.fastapi_thread = FastAPIServer()
        self.fastapi_thread.started_signal.connect(self.logMessage)

        self.chart_modes = ['bar100', 'pie', 'stacked_barh', 'doughnut']
        self.chart_index = 0
        self.chart_timer = QtCore.QTimer()
        self.chart_timer.timeout.connect(self.toggleChartMode)
        self.chart_timer.start(10000)

        self.translated = False
        self.translation_map = {
            "Predict": "预测",
            "Delete Predictions": "删除预测",
            "Refresh": "刷新",
            "Predictions": "预测结果",
            "Live Prediction": "实时预测",
            "Performance Metrics": "性能指标",
            "Settings & Log": "设置和日志",
            "Draw ID": "抽奖编号",
            "Prediction": "预测结果",
            "Accuracy (%)": "准确率 (%)",
            "Live prediction updates will appear here.": "实时预测更新将显示在此处。",
            "Big vs Small Distribution": "大与小分布",
            "Even vs Odd Distribution": "奇与偶分布",
            "Big vs Small (bar100)": "大与小 (百分比条形图)",
            "Even vs Odd (bar100)": "奇与偶 (百分比条形图)",
            "Please wait": "请稍候",
            "Processing...": "正在处理...",
            "Translate to Chinese": "切换为中文",
            "Translate to English": "切换为英文",
            "Prediction successful.": "预测成功。",
            "Predictions deleted.": "预测已删除。",
            "Fetching predictions...": "正在获取预测...",
            "Big": "大",
            "Small": "小",
            "Even": "偶",
            "Odd": "奇",
            self.translate("No data"): "暂无数据",
            self.translate("Big/Small"): "大/小",
            self.translate("Even/Odd"): "偶/奇"
        }

    def logMessage(self, message):
        if hasattr(self, "log_text"):
            self.log_text.append(message)
        else:
            print(message)

    def initUI(self):
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)

        self.central.setStyleSheet("""
            QWidget { background-color: #333; }
            QPushButton {
                background-color: #666;
                border: none;
                padding: 10px;
                font-size: 14px;
                color: white;
                border-radius: 6px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QLabel {
                font-size: 16px;
                color: white;
            }
            QTableWidget {
                background-color: #444;
                color: white;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self.central)
        button_layout = QtWidgets.QHBoxLayout()

        self.btn_predict = QtWidgets.QPushButton("Predict")
        self.btn_delete = QtWidgets.QPushButton("Delete Predictions")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_translate = QtWidgets.QPushButton("Translate to Chinese")
        self.btn_translate.clicked.connect(self.toggleLanguage)

        button_layout.addWidget(self.btn_predict)
        button_layout.addWidget(self.btn_delete)
        button_layout.addWidget(self.btn_refresh)
        button_layout.addWidget(self.btn_translate)

        main_layout.addLayout(button_layout)

        self.loader = QtWidgets.QProgressDialog("Processing...", None, 0, 0, self)
        self.loader.setWindowTitle("Please wait")
        self.loader.setCancelButton(None)
        self.loader.setWindowModality(QtCore.Qt.ApplicationModal)
        self.loader.close()

        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_predictions = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_predictions, "Predictions")
        self.initPredictionTab()

        self.tab_live = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_live, "Live Prediction")
        self.initLivePredictionTab()

        self.tab_performance = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_performance, "Performance Metrics")
        self.initPerformanceMetricsTab()

        self.tab_settings = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_settings, "Settings & Log")
        self.initSettingsTab()

        self.btn_predict.clicked.connect(self.predictNext)
        self.btn_delete.clicked.connect(self.deletePredictions)
        self.btn_refresh.clicked.connect(self.refreshTable)

    def translate(self, text):
        return self.translation_map.get(text, text) if self.translated else text

    def toggleLanguage(self):
        self.translated = not self.translated
        self.updateLanguage()

    def updateLanguage(self):
        self.btn_predict.setText(self.translate("Predict"))
        self.btn_delete.setText(self.translate("Delete Predictions"))
        self.btn_refresh.setText(self.translate("Refresh"))
        self.tabs.setTabText(0, self.translate("Predictions"))
        self.tabs.setTabText(1, self.translate("Live Prediction"))
        self.tabs.setTabText(2, self.translate("Performance Metrics"))
        self.tabs.setTabText(3, self.translate("Settings & Log"))
        self.table.setHorizontalHeaderLabels([
            self.translate("Draw ID"),
            self.translate("Prediction"),
            self.translate("Accuracy (%)")
        ])
        self.live_label.setText(self.translate("Live prediction updates will appear here."))
        self.ax_big_small.set_title(self.translate("Big vs Small Distribution"))
        self.ax_even_odd.set_title(self.translate("Even vs Odd Distribution"))
        self.loader.setWindowTitle(self.translate("Please wait"))
        self.loader.setLabelText(self.translate("Processing..."))
        self.btn_translate.setText(self.translate("Translate to English") if self.translated else self.translate("Translate to Chinese"))

    def initPredictionTab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_predictions)
        self.table = QtWidgets.QTableWidget(20, 3)
        self.table.setHorizontalHeaderLabels(["Draw ID", "Prediction", "Accuracy (%)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

    def initLivePredictionTab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_live)
        self.live_label = QtWidgets.QLabel("Live prediction updates will appear here.")
        layout.addWidget(self.live_label)

    def initPerformanceMetricsTab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_performance)
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        layout.addWidget(self.canvas)

        self.ax_big_small = figure.add_subplot(121)
        self.ax_even_odd = figure.add_subplot(122)

        self.ax_big_small.set_title("Big vs Small Distribution")
        self.ax_even_odd.set_title("Even vs Odd Distribution")

    def initSettingsTab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_settings)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def predictNext(self):
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=3)
            if response.status_code != 200:
                raise Exception("FastAPI not running")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "API Error", str(e))
            return

        self.loader.setLabelText("Fetching predictions...")
        self.loader.show()

        worker = PredictionWorker()
        worker.signals.finished.connect(self.onPredictionSuccess)
        worker.signals.error.connect(self.onPredictionError)
        self.threadpool.start(worker)

    def onPredictionSuccess(self, data):
        self.loader.close()
        self.updateTable(data)
        self.updatePerformanceMetrics(data)
        self.log_text.append("✅ Prediction successful.")

    def onPredictionError(self, error_msg):
        self.loader.close()
        self.log_text.append(error_msg)
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)

    def updateTable(self, predictions):
        self.table.setRowCount(len(predictions))
        for row, prediction in enumerate(predictions):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(prediction.get("Draw ID", ""))))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(prediction.get("Prediction", "")))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{prediction.get('Accuracy (%)', '')}%"))


    
    def updatePerformanceMetrics(self, predictions):
        self.predictions = predictions
        big = small = even = odd = 0

        for p in predictions:
            pred = str(p.get("Prediction", "")).strip().lower()
            if pred == "big":
                big += 1
            elif pred == "small":
                small += 1
            elif pred == "even":
                even += 1
            elif pred == "odd":
                odd += 1

        total_bs = big + small
        total_eo = even + odd
        mode = self.chart_modes[self.chart_index] if hasattr(self, 'chart_modes') else 'bar100'

        self.ax_big_small.clear()
        self.ax_even_odd.clear()

        if total_bs == 0 and total_eo == 0:
            self.ax_big_small.text(0.5, 0.5, "No data", ha='center', va='center')
            self.ax_even_odd.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            if mode == 'bar100':
                if total_bs > 0:
                    bs_vals = [big/total_bs*100, small/total_bs*100]
                    self.ax_big_small.bar(["Big vs Small"], [bs_vals[0]], label="Big", color='orange')
                    self.ax_big_small.bar(["Big vs Small"], [bs_vals[1]], bottom=[bs_vals[0]], label="Small", color='blue')
                    self.ax_big_small.set_ylabel("Percentage")
                    self.ax_big_small.legend()
                if total_eo > 0:
                    eo_vals = [even/total_eo*100, odd/total_eo*100]
                    self.ax_even_odd.bar(["Even vs Odd"], [eo_vals[0]], label="Even", color='green')
                    self.ax_even_odd.bar(["Even vs Odd"], [eo_vals[1]], bottom=[eo_vals[0]], label="Odd", color='red')
                    self.ax_even_odd.set_ylabel("Percentage")
                    self.ax_even_odd.legend()
            elif mode == 'pie':
                if total_bs > 0:
                    self.ax_big_small.pie([big, small], labels=["Big", "Small"], autopct="%1.1f%%", startangle=90)
                else:
                    self.ax_big_small.text(0.5, 0.5, "No data", ha='center', va='center')
                if total_eo > 0:
                    self.ax_even_odd.pie([even, odd], labels=["Even", "Odd"], autopct="%1.1f%%", startangle=90)
                else:
                    self.ax_even_odd.text(0.5, 0.5, "No data", ha='center', va='center')
            elif mode == 'stacked_barh':
                if total_bs > 0:
                    self.ax_big_small.barh(["Big vs Small"], [small], label="Small", color='blue')
                    self.ax_big_small.barh(["Big vs Small"], [big], left=[small], label="Big", color='orange')
                    self.ax_big_small.legend()
                if total_eo > 0:
                    self.ax_even_odd.barh(["Even vs Odd"], [odd], label="Odd", color='red')
                    self.ax_even_odd.barh(["Even vs Odd"], [even], left=[odd], label="Even", color='green')
                    self.ax_even_odd.legend()
            elif mode == 'doughnut':
                if total_bs > 0:
                    self.ax_big_small.pie([big, small], labels=["Big", "Small"], autopct="%1.1f%%",
                                          startangle=90, wedgeprops=dict(width=0.4))
                    self.ax_big_small.text(0, 0, "Big/Small", ha='center', va='center')
                if total_eo > 0:
                    self.ax_even_odd.pie([even, odd], labels=["Even", "Odd"], autopct="%1.1f%%",
                                         startangle=90, wedgeprops=dict(width=0.4))
                    self.ax_even_odd.text(0, 0, "Even/Odd", ha='center', va='center')

        self.ax_big_small.set_title(f"Big vs Small ({mode})")
        self.ax_even_odd.set_title(f"Even vs Odd ({mode})")
        self.canvas.draw()

        
   


    def deletePredictions(self):
        self.table.clearContents()
        self.log_text.append("🗑 Predictions deleted.")

    def refreshTable(self):
        self.predictNext()

    def toggleChartMode(self):
        self.chart_index = (self.chart_index + 1) % len(self.chart_modes)
        self.updatePerformanceMetrics(self.predictions if hasattr(self, 'predictions') else [])


    def closeEvent(self, event):
        if hasattr(self, "fastapi_thread"):
            self.fastapi_thread.stop_server()
            self.logMessage("🔴 FastAPI server stopped.")
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Lucky28PredictionApp()
    window.show()
    sys.exit(app.exec_())


    