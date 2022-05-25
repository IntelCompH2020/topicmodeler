# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:19:34 2021
@author: lcalv
"""

from PyQt6 import QtCore


class WorkerSignals(QtCore.QObject):
    """ Module that defines the signals that are available from a running
        worker thread, the supported signals being “finished” (there is no more data to process),
        “error”, “result” (object data returned from processing) and “progress” (a
        numerical indicator of the progress that has been achieved at a particular moment).
        It has been created based on the analogous class provided by:
        https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
    """
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
