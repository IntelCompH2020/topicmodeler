# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:19:34 2021
@author: lcalv
"""

# General imports
from PyQt6 import QtCore
import sys
import traceback

# Local imports
from src.gui.utils.worker_signals import WorkerSignals


class Worker(QtCore.QRunnable):
    """
    Module that inherits from QRunnable and is used to handler worker
    thread setup, signals and wrap-up. It has been created based on the analogous
    class provided by:
    https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/.
    """

    def __init__(self, fn, *args, **kwargs):
        """
        Initializes the application's main window based on the parameters received
        from the application's starting window.

        Parameters
        ----------
        callback : UDF
            The function callback to run on this worker thread.
            Supplied args and kwargs will be passed through to the runner.
        callback : UDF
            Function
        args : list
            Arguments to pass to the callback function
        kwargs : dict
            Keywords to pass to the callback function
        """
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        """Initialises the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.signals.started.emit()
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()  # Done
