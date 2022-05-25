# -*- coding: utf-8 -*-
"""
@author: lcalv
"""

# General imports
import sys
from PyQt6 import QtCore


class OutputWrapper(QtCore.QObject):
    """
    Module that overrides the "sys.stderr" and "sys.stdout" with a wrapper object
    that emits a signal whenever output is written. In order to account for other
    modules that need "sys.stdout" / "sys.stderr" (such as the logging module) use
    the wrapped versions wherever necessary, the instance of the OutputWrapper are
    created before the TaskManager object.

    It has been created based on the analogous
    class provided by:
    https://stackoverflow.com/questions/19855288/duplicate-stdout-stderr-in-qtextedit-widget
    """

    outputWritten = QtCore.pyqtSignal(str)

    def __init__(self, parent, stdout=True):
        super().__init__(parent)
        if stdout:
            self._stream = sys.stdout
            sys.stdout = self
        else:
            self._stream = sys.stderr
            sys.stderr = self
        self._stdout = stdout

    def write(self, text):
        self._stream.write(text)
        self.outputWritten.emit(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            if self._stdout:
                sys.stdout = self._stream
            else:
                sys.stderr = self._stream
        except AttributeError:
            pass
