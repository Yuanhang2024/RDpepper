"""Entry point: python -m cycpep_master.gui"""
import sys


def main():
    try:
        from PyQt5 import QtWidgets
    except ImportError:
        sys.stderr.write(
            "PyQt5 is required for the GUI. Install with: pip install PyQt5\n")
        return 1
    from .main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("RDpepper")
    app.setApplicationDisplayName("RDpepper")
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
