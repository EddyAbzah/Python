"""
This application allows users to pick colors from anywhere on their screen.
Left-click on any screen to fill half of the GUI with the clicked color.
Right-click to clear the colors.
Middle-click toggles the GUI's "Always on Top" state.

Dependencies:
pip install PyQt5 pynput screeninfo pillow
"""


import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt
from PIL import ImageGrab
from functools import partial
from pynput import mouse
from screeninfo import get_monitors

# Ensure ImageGrab captures all screens by default
ImageGrab.grab = partial(ImageGrab.grab, all_screens=True)


class ColorPickerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color Comparison")
        self.setGeometry(100, 100, 800, 400)
        self.central_widget = ColorWidget()
        self.setCentralWidget(self.central_widget)
        self.always_on_top = False

        # Start the global mouse listener
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()

    def on_click(self, x, y, button, pressed):
        if pressed:
            try:
                monitors = get_monitors()
                for monitor in monitors:
                    if monitor.x <= x < monitor.x + monitor.width and monitor.y <= y < monitor.y + monitor.height:
                        if button == mouse.Button.left:
                            # Use ImageGrab.grab() to capture the full screen and get pixel color
                            screenshot = ImageGrab.grab()
                            rgb = screenshot.getpixel((x, y))
                            color = QColor(*rgb)
                            self.central_widget.update_color(color, rgb)
                        elif button == mouse.Button.right:
                            self.central_widget.clear_colors()
                        elif button == mouse.Button.middle:
                            self.always_on_top = not self.always_on_top
                            self.setWindowFlag(Qt.WindowStaysOnTopHint, self.always_on_top)
                            self.show()
                        return  # Stop checking other monitors if coordinates match
                raise ValueError("Click occurred outside available monitors.")
            except Exception as e:
                print(f"Error: {e}")

    def closeEvent(self, event):
        # Stop the listener when the GUI is closed
        self.listener.stop()
        super().closeEvent(event)


class ColorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.color_left = None
        self.color_right = None
        self.rgb_left = None
        self.rgb_right = None
        self.next_half = "left"  # Determines which half to fill next

    def update_color(self, color, rgb):
        if self.next_half == "left":
            self.color_left = color
            self.rgb_left = rgb
            self.next_half = "right"
        else:
            self.color_right = color
            self.rgb_right = rgb
            self.next_half = "left"
        self.update()

    def clear_colors(self):
        self.color_left = None
        self.color_right = None
        self.rgb_left = None
        self.rgb_right = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        font_size = max(10, min(width, height) // 18)
        painter.setFont(QFont("Arial", font_size))

        # Fill the left half
        if self.color_left:
            painter.fillRect(0, 0, width // 2, height, self.color_left)
            if self.rgb_left:
                painter.setPen(Qt.white if self.color_left.lightness() < 128 else Qt.black)
                painter.drawText(10, 10 + font_size, f"RGB: {self.rgb_left}")
        else:
            painter.fillRect(0, 0, width // 2, height, Qt.white)

        # Fill the right half
        if self.color_right:
            painter.fillRect(width // 2, 0, width // 2, height, self.color_right)
            if self.rgb_right:
                painter.setPen(Qt.white if self.color_right.lightness() < 128 else Qt.black)
                painter.drawText(width // 2 + 10, 10 + font_size, f"RGB: {self.rgb_right}")
        else:
            painter.fillRect(width // 2, 0, width // 2, height, Qt.white)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorPickerGUI()
    window.show()
    sys.exit(app.exec_())
