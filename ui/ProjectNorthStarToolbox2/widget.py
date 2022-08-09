# This Python file uses the following encoding: utf-8
import sys
import numpy as np

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QImage, QPixmap

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_V2Form

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_V2Form()
        self.ui.setupUi(self)
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
