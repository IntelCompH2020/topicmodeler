"""
*** IntelComp H2020 project ***
*** Interactive Topic Model Trainer ***

Graphical User Interface for the interactive training of Topic Models

It implements graphical user interface based on PyQT6 for the training and curation of topic models
exploiting the tools available in topicmodeling.py:
   - Different LDA implementations
   - Topic assessment tools
   - Topic curation tools

Contributors: José Antonio Espinosa-Melchor
              Lorena Calvo-Bartolomé
              Jerónimo Arenas-García (jarenas@ing.uc3m.es)
"""
import argparse
import pathlib

from PyQt6.QtWidgets import QApplication
import sys

# Local imports
from src.gui.main_window import MainWindow


def main():
    # ####################
    # Read input arguments
    # ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default=None,
                        help="path to a new or an existing project")
    parser.add_argument('--parquet', type=str, default=None,
                        help="path to downloaded parquet datasets")
    parser.add_argument('--f', action='store_true', default=False,
                        help='Force creation of new project. Overwrite existing.')
    args = parser.parse_args()

    # Read project_path
    project_path = pathlib.Path(args.p) if args.p is not None else None

    # Read parquet_path
    parquet_path = pathlib.Path(args.parquet) if args.parquet is not None else None

    # ####################
    # Create application
    # ####################
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon('images/dc_logo.png'))

    gui = MainWindow(project_path, parquet_path)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()