"""
AgroStake GUI Application - Modern Dark-Themed Interface
Send images to ESP32-S3 and receive real-time inference results
PyInstaller friendly single-file executable
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage

import serial.tools.list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QScrollArea, 
    QTextEdit, QFrame, QSplitter
)
from PyQt6.QtGui import QPixmap, QFont, QIcon, QColor, QPainter, QImage
from PyQt6.QtCore import Qt, QSize, QTimer
from processor import SerialProcessor


class AgroStakeGUI(QMainWindow):
    """Main GUI Application Window"""
    
    def __init__(self):
        super().__init__()
        self.processor = SerialProcessor()
        self.current_image_path = None
        self.init_ui()
        self.apply_dark_theme()
        self.refresh_com_ports()
        
    def init_ui(self):
        """Initialize user interface components"""
        self.setWindowTitle("AgroStake - ESP32 Inference")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # --- LEFT PANEL: Image & Controls ---
        left_panel = QVBoxLayout()
        
        # Image preview section
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        image_frame.setFrameShadow(QFrame.Shadow.Raised)
        image_layout = QVBoxLayout(image_frame)
        
        image_label = QLabel("Image Preview (400×300)")
        image_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        image_layout.addWidget(image_label)
        
        self.image_display = QLabel()
        self.image_display.setFixedSize(400, 300)
        self.image_display.setStyleSheet("border: 2px solid #3a3a3a; background-color: #1a1a1a;")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setText("No image selected\nClick 'Select Image' to load")
        image_layout.addWidget(self.image_display)
        
        left_panel.addWidget(image_frame)
        
        # Control buttons section
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)
        
        # Select Image button
        self.btn_select = QPushButton("📂 Select Image")
        self.btn_select.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_select.setFixedHeight(45)
        self.btn_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select.clicked.connect(self.select_image)
        controls_layout.addWidget(self.btn_select)
        
        # COM Port selection
        com_label = QLabel("Serial Port:")
        com_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        controls_layout.addWidget(com_label)
        
        com_layout = QHBoxLayout()
        self.com_dropdown = QComboBox()
        self.com_dropdown.setFont(QFont("Arial", 10))
        self.com_dropdown.setFixedHeight(35)
        com_layout.addWidget(self.com_dropdown)
        
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setFixedSize(40, 35)
        self.btn_refresh.setFont(QFont("Arial", 12))
        self.btn_refresh.clicked.connect(self.refresh_com_ports)
        com_layout.addWidget(self.btn_refresh)
        controls_layout.addLayout(com_layout)
        
        # Run Inference button
        self.btn_inference = QPushButton("▶ Run Inference")
        self.btn_inference.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_inference.setFixedHeight(50)
        self.btn_inference.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_inference.clicked.connect(self.run_inference)
        self.btn_inference.setEnabled(False)
        controls_layout.addWidget(self.btn_inference)
        
        controls_layout.addStretch()
        left_panel.addWidget(controls_frame)
        
        # --- RIGHT PANEL: Console & Results ---
        right_panel = QVBoxLayout()
        
        # Results section
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.Shape.StyledPanel)
        results_layout = QVBoxLayout(results_frame)
        
        results_title = QLabel("Inference Results")
        results_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        results_layout.addWidget(results_title)
        
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Courier New", 10))
        self.result_display.setFixedHeight(120)
        self.result_display.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f0f;
                color: #00ff00;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
            }
        """)
        results_layout.addWidget(self.result_display)
        right_panel.addWidget(results_frame)
        
        # Console/Log section
        console_frame = QFrame()
        console_frame.setFrameShape(QFrame.Shape.StyledPanel)
        console_layout = QVBoxLayout(console_frame)
        
        console_title = QLabel("Console Output")
        console_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        console_layout.addWidget(console_title)
        
        # Scrollable console
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setFont(QFont("Courier New", 9))
        self.console_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f0f;
                color: #c0c0c0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
            }
        """)
        scroll_area.setWidget(self.console_text)
        console_layout.addWidget(scroll_area)
        
        # Clear console button
        self.btn_clear = QPushButton("🗑️ Clear Console")
        self.btn_clear.setFixedHeight(35)
        self.btn_clear.clicked.connect(self.clear_console)
        console_layout.addWidget(self.btn_clear)
        
        right_panel.addWidget(console_frame)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        right_panel.addWidget(self.status_label)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        central_widget.setLayout(main_layout)
    
    def apply_dark_theme(self):
        """Apply modern dark theme using Fusion style"""
        app = QApplication.instance()
        app.setStyle('Fusion')
        
        palette = app.palette()
        dark_gray = QColor(25, 25, 25)
        lighter_gray = QColor(50, 50, 50)
        light_text = QColor(200, 200, 200)
        
        palette.setColor(palette.ColorRole.Window, dark_gray)
        palette.setColor(palette.ColorRole.WindowText, light_text)
        palette.setColor(palette.ColorRole.Base, QColor(15, 15, 15))
        palette.setColor(palette.ColorRole.AlternateBase, lighter_gray)
        palette.setColor(palette.ColorRole.ToolTipBase, lighter_gray)
        palette.setColor(palette.ColorRole.ToolTipText, light_text)
        palette.setColor(palette.ColorRole.Text, light_text)
        palette.setColor(palette.ColorRole.Button, lighter_gray)
        palette.setColor(palette.ColorRole.ButtonText, light_text)
        palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(palette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        app.setPalette(palette)
        
        # Button styling
        button_style = """
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d3a91;
            }
            QPushButton:disabled {
                background-color: #5a5a5a;
                color: #808080;
            }
        """
        
        # ComboBox styling
        combo_style = """
            QComboBox {
                background-color: #2a2a2a;
                color: #c0c0c0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                width: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #c0c0c0;
                selection-background-color: #0d47a1;
            }
        """
        
        app.setStyleSheet(button_style + combo_style)
    
    def select_image(self):
        """Open file dialog to select image"""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilters(["Image Files (*.png *.jpg *.jpeg *.bmp)", "All Files (*)"])
        file_dialog.setDefaultSuffix("png")
        
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            self.load_image(file_path)
    
    def load_image(self, image_path):
        """Load the image and visualise the detected ROI."""
        try:
            self.current_image_path = image_path
            pil_img = PILImage.open(image_path).convert('RGB')
            orig_w, orig_h = pil_img.size
            
            # 1. Get coordinates from the processor logic
            x, y, w, h = self.processor.get_roi_coordinates(pil_img)
            
            # 2. Prep UI scaling (400x300 box)
            display_img = pil_img.copy()
            display_img.thumbnail((400, 300), PILImage.Resampling.LANCZOS)
            tw, th = display_img.size
            sx, sy = tw / orig_w, th / orig_h 
            
            # 3. Create the pixmap and apply the ROI spotlight
            qimg = QImage(display_img.tobytes("raw", "RGB"), tw, th, tw * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Apply the visual mask
            self.draw_roi_viz(pixmap, int(x*sx), int(y*sy), int(w*sx), int(h*sy))
            
            self.image_display.setPixmap(pixmap)
            self.btn_inference.setEnabled(True)
            self.log(f"✓ ROI Detected: {w}x{h} at ({x},{y})")
        except Exception as e:
            self.log(f"❌ Error: {e}")
    def draw_roi_viz(self, pixmap, x, y, w, h):
        """Draw the high-opacity mask and gradient"""
        painter = QPainter(pixmap)
        mask_color = QColor(0, 0, 0, 180) # Darkened overlay
        
        # Fill areas outside the ROI box
        painter.fillRect(0, 0, pixmap.width(), y, mask_color) # Top
        painter.fillRect(0, y+h, pixmap.width(), pixmap.height()-(y+h), mask_color) # Bottom
        painter.fillRect(0, y, x, h, mask_color) # Left
        painter.fillRect(x+w, y, pixmap.width()-(x+w), h, mask_color) # Right
        
        # Glowing green focus bar
        grad = QLinearGradient(x, y, x, y+10)
        grad.setColorAt(0, QColor(0, 255, 0, 150))
        grad.setColorAt(1, QColor(0, 255, 0, 0))
        painter.fillRect(x, y, w, 10, QBrush(grad))
        painter.end()
        
        painter.end()
    def refresh_com_ports(self):
        """Refresh list of available COM ports"""
        self.com_dropdown.clear()
        ports = serial.tools.list_ports.comports()
        
        if ports:
            for port in ports:
                self.com_dropdown.addItem(f"{port.device} - {port.description}", port.device)
            self.log(f"✓ Found {len(ports)} serial port(s)")
        else:
            self.com_dropdown.addItem("No ports available")
            self.log("⚠️ No serial ports detected")
    
    def run_inference(self):
        """Execute inference sequence"""
        if not self.current_image_path:
            self.log("❌ No image selected")
            return
        
        if self.com_dropdown.count() == 0 or self.com_dropdown.currentText() == "No ports available":
            self.log("❌ No serial port selected")
            return
        
        # Disable buttons during inference
        self.btn_inference.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.com_dropdown.setEnabled(False)
        
        # Get selected port
        port = self.com_dropdown.currentData()
        
        # Create and configure processor
        self.processor = SerialProcessor()
        self.processor.set_parameters(port, self.current_image_path)
        
        # Connect signals
        self.processor.log_signal.connect(self.log)
        self.processor.error_signal.connect(self.show_error)
        self.processor.result_signal.connect(self.display_result)
        self.processor.progress_signal.connect(self.update_status)
        self.processor.finished.connect(self.inference_finished)
        
        self.log("=" * 60)
        self.log(f"🚀 Starting inference at {datetime.now().strftime('%H:%M:%S')}")
        self.log(f"   Port: {port}")
        self.log(f"   Image: {Path(self.current_image_path).name}")
        self.log("=" * 60)
        
        # Start processor thread
        self.processor.start()
    
    def inference_finished(self):
        """Called when inference thread finishes"""
        self.btn_inference.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.com_dropdown.setEnabled(True)
    
    def display_result(self, result):
        """Display inference result"""
        class_name = result.get('class_name', 'Unknown')
        confidence = result.get('confidence', 0)
        
        result_text = f"""
╔════════════════════════════════════════╗
║        INFERENCE RESULT                ║
╠════════════════════════════════════════╣
║ Class:      {class_name:<20} ║
║ Confidence: {confidence:>6.1f}%             ║
╚════════════════════════════════════════╝
        """
        
        self.result_display.setText(result_text)
        self.log("\n" + result_text)
    
    def log(self, message):
        """Add message to console log"""
        self.console_text.append(message)
        # Auto-scroll to bottom
        self.console_text.verticalScrollBar().setValue(
            self.console_text.verticalScrollBar().maximum()
        )
    
    def show_error(self, error_message):
        """Display error message"""
        self.log(f"🚨 {error_message}")
    
    def update_status(self, status):
        """Update status label"""
        status_colors = {
            "Ready": "#00ff00",
            "Loading image...": "#ffff00",
            "Connecting to ESP32...": "#ffaa00",
            "Sending image to ESP32...": "#ffaa00",
            "Waiting for inference result...": "#00aaff",
        }
        
        self.status_label.setText(status)
        color = status_colors.get(status, "#c0c0c0")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def clear_console(self):
        """Clear console output"""
        self.console_text.clear()
        self.log("Console cleared")


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    window = AgroStakeGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
