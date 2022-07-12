"""
* *IntelComp H2020 project*

Python module with a set of auxiliary methods for the Interactive Topic Model Trainer's GUI
Among others, it implements the functions needed to

    - Execute certain high computational load tasks in a secondary thread so the GUI does not freeze
    - Configure a set of QWidget objects with some specific criteria
    - Keep track of recent projects and parquet folders
"""

import os
import pathlib
import pickle
import shutil
import xml.etree.ElementTree as ET

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (QProgressBar, QPushButton, QTableWidget,
                             QTableWidgetItem, QTextEdit)
from src.gui.utils.worker import Worker


def execute_in_thread(gui, function, function_output, progress_bar):
    """
    Method to execute a certain function in a secondary thread while keeping the GUI's execution in the main thread.
    A progress bar is shown at the time the function is being executed if a progress bar object is provided. When finished, it forces the execution of the method to be executed after the function executing in a thread is completed. Based on the functions provided in the manual available at: https://www.pythonguis.com/tutorials multithreading-pyqt-applications-qthreadpool/

    Parameters
    ----------
    gui: MainWindow
        Window associated to the execution in the main thread
    function: UDF
        Function to be executed in thread
    function_output: UDF
        Function to be executed at the end of the thread
    progress_bar: QProgressBar
        If a QProgressBar object is provided, it shows a progress bar in the
        main thread while the main task is being carried out in a secondary thread
    """

    # Pass the function to execute
    gui.worker = Worker(function)

    # Show progress if a QProgressBar object has been passed as argument to the function
    if progress_bar is not None:
        signal_accept(progress_bar)

    # Connect function that is going to be executed when the task being
    # carrying out in the secondary thread has been completed
    gui.worker.signals.finished.connect(function_output)

    # Execute
    gui.thread_pool.start(gui.worker)

    return


def signal_accept(progress_bar):
    """
    Makes the progress bar passed as an argument visible and configures it for
    an event whose duration is unknown by setting both its minimum and maximum
    both to 0, thus the bar shows a busy indicator instead of a percentage of steps.

    Parameters
    ----------
    progress_bar: QProgressBar
         Progress bar object in which the progress is going to be displayed.
    """

    progress_bar.setVisible(True)
    progress_bar.setMaximum(0)
    progress_bar.setMinimum(0)

    return


def configure_table_header(list_tables, window):
    """
    Configures all the tables defined in the list given by "list_tables" in the sense that it makes their horizontal headers visible disables the functionality of highlighting sections and resizes their rows to fit their content.

    Parameters
    ----------
    list_tables: List[str]
         List with the names of the QTableWidget objects to configure
    window: MainWindow
        Window to which the tables belong to
    """

    for table_name in list_tables:
        table_widget = window.findChild(QTableWidget, table_name)
        table_widget.horizontalHeader().setVisible(True)
        table_widget.horizontalHeader().setHighlightSections(False)
        table_widget.resizeRowsToContents()

    return


def initialize_progress_bar(list_progress_bars, window):
    """
    Initializes all the progress bars defined in the list given by "list_progress_bars" in the sense that it makes them invisible to the user and establishes their progress at 0.

    Parameters
    ----------
    list_progress_bars: List[str]
         List with the names of the QProgressBar objects to configure
    window: MainWindow
        Window to which the progress bars belong to
    """
    for progress_bar in list_progress_bars:
        progress_bar_widget = window.findChild(QProgressBar, progress_bar)
        progress_bar_widget.setVisible(False)
        progress_bar_widget.setValue(0)

    return


def add_checkboxes_to_table(table, column=None):
    """
    Adds a checkbox at the last column of every row of the table specified by "table".

    Parameters
    ----------
    table: QTableWidget
        Table to which the checkboxes will be added to
    """
    if column is None:
        column = table.columnCount() - 1
    for row in range(table.rowCount()):
        chkBoxItem = QTableWidgetItem()
        chkBoxItem.setFlags(
            QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        chkBoxItem.setCheckState(QtCore.Qt.CheckState.Unchecked)
        table.setItem(row, column, chkBoxItem)

    return


def save_recent(current_project, current_parquet, current_wordlist):
    """
    Saves in a pickle file a dictionary structure with a list of the last used projects and parquet folders. If the file exists, the current project and parquet folders are added to the corresponding lists in the dictionary with the condition that they are not equal to their corresponding pair in the previous execution. If the file does not exist, a new dictionary is instantiated with each of its lists being conformed by just the current project and parquet folder, respectively,

    Parameters
    ----------
    current_project: pathlib.Path
        Route to the project folder of the current execution
    current_parquet: pathlib.Path
        Route to the parquet folder of the current execution
    """

    dtSets = pathlib.Path("src/gui/utils").iterdir()
    dtSets = sorted([d for d in dtSets if d.name.endswith(".pickle")])
    if dtSets:
        with open(dtSets[0], "rb") as f:
            dict_recent = pickle.load(f)
            if current_project != dict_recent["recent_projects"][-1]:
                dict_recent["recent_projects"].append(current_project)
            if current_parquet != dict_recent["recent_parquets"][-1]:
                dict_recent["recent_parquets"].append(current_parquet)
            if current_wordlist != dict_recent["recent_wordlists"][-1]:
                dict_recent["recent_wordlists"].append(current_wordlist)
        with open(dtSets[0], 'wb') as f:
            pickle.dump(dict_recent, f)
    else:
        dict_recent = {"recent_projects": [current_project],
                       "recent_parquets": [current_parquet],
                       "recent_wordlists": [current_wordlist]}
        with open(pathlib.Path("src/gui/utils/recent.pickle"), 'wb') as f:
            pickle.dump(dict_recent, f)

    return


def set_recent_buttons(window):
    """
    Loads the dictionary of recent projects and parquet folders into execution time. If the dictionary is not empty, it looks for the last two project folders and parquet folders. After checking if the routes still exist in the user's OS, it adds the route as text in the corresponding buttons. Otherwise, in case the dictionary is empty and thus there are no recent projects, the frame that contains the buttons for the recent folders is hidden from the window.

    Parameters
    ----------
    window: MainWindow
        Window in which the buttons are going to be configured
    """

    dtSets = pathlib.Path("src/gui/utils").iterdir()
    dtSets = sorted([d for d in dtSets if d.name.endswith(".pickle")])
    dict_recent = {}
    if dtSets:
        with open(dtSets[0], "rb") as f:
            dict_recent = pickle.load(f)
    if len(dict_recent) != 0:
        # Fill buttons
        for rp in reversed(range(len(dict_recent['recent_projects']))):
            if os.path.exists(dict_recent['recent_projects'][rp]):
                button_name = "pushButton_recent_project_folder_" + str(rp + 1)
                button_widget = window.findChild(QPushButton, button_name)
                button_widget.setText(
                    dict_recent['recent_projects'][rp].as_posix())
            else:
                continue
        for rpa in reversed(range(len(dict_recent['recent_parquets']))):
            if os.path.exists(dict_recent['recent_parquets'][rpa]):
                button_name = "pushButton_recent_parquet_folder_" + \
                    str(rpa + 1)
                button_widget = window.findChild(QPushButton, button_name)
                button_widget.setText(
                    dict_recent['recent_parquets'][rpa].as_posix())
            else:
                continue
        for rwl in reversed(range(len(dict_recent['recent_wordlists']))):
            if os.path.exists(dict_recent['recent_wordlists'][rpa]):
                button_name = "pushButton_recent_wordlists_folder_" + \
                    str(rpa + 1)
                button_widget = window.findChild(QPushButton, button_name)
                button_widget.setText(
                    dict_recent['recent_wordlists'][rwl].as_posix())
            else:
                continue
    return


def clearQTreeWidget(tree):
    """
    Removes all the elements of a QTreeWidget.

    Parameters
    ----------
    window: QTreeWidget
        QTreeWidget whose elements are desired to be removed.
    """

    iterator = QtWidgets.QTreeWidgetItemIterator(
        tree, QtWidgets.QTreeWidgetItemIterator.IteratorFlag.All)
    while iterator.value():
        iterator.value().takeChildren()
        iterator += 1
    i = tree.topLevelItemCount()
    while i > -1:
        tree.takeTopLevelItem(i)
        i -= 1


def printTree(xml_ret, treeWidget):
    """
    Displays the elements of a XML ET.Element into a QTreeWidget.

    Parameters
    ----------
    xml_ret: ET.Element
        ET.Element whose elements are going to be displayed in the QTreeWidget.
    treeWidget: QTreeWidget
        QTreeWidget object to visualize the items on.
    """

    treeWidget.setColumnCount(1)
    treeWidget.setHeaderHidden(True)
    a = QtWidgets.QTreeWidgetItem([xml_ret.tag])
    treeWidget.addTopLevelItem(a)
    a.setExpanded(True)

    def displayTree(a, s):
        for child in s:
            branch = QtWidgets.QTreeWidgetItem([child.tag])
            a.addChild(branch)
            displayTree(branch, child)
        if s.text is not None:
            content = s.text
            a.addChild(QtWidgets.QTreeWidgetItem([content]))

    displayTree(a, xml_ret)


def get_model_xml(path):
    """
    Gets a XML ET.Element in which the hierarchical structure of all HTMs contained in the project folder.

    Parameters
    ----------
    path: str
        String referring to the path to the "models" directory in the project folder.

    Returns
    -------
    ret : ET.Element
        Pretty printed XML ElementTree with the project folder's HTMs structure.
    """

    ret = xml_dir(pathlib.Path(path))
    indent(ret)

    return ret


def xml_dir(pth, et_element=None):
    """
    Recursively creates a xml file which lists all the contents of a directory. Based on:
    https://stackoverflow.com/questions/44435618/python-programatically-create-an-xml-file-which-lists-all-the-contents-of-a-dir

    Parameters
    ----------
    pth: Pathlib.path
        Directory whose contents are going to be listed.
    path: et_element, optional
        Root of the XML file, defaults to None

    Returns
    -------
    et_element : ET.Element
        ET structure of the directory
    """

    if et_element is None:
        et_element = ET.Element(pth.name)
    else:
        et_element = ET.SubElement(et_element, pth.name)

    for directory in (fle for fle in pth.iterdir() if fle.is_dir()):
        if not directory.as_posix().endswith("modelFiles") and not directory.as_posix().endswith("corpus.parquet"):
            xml_dir(directory, et_element)

    return et_element


def indent(elem, level=0):
    """
    Pretty prints a XML ElementTree.
    Taken from: http://effbot.org/zone/element-lib.htm#prettyprint

    Parameters
    ----------
    elem: ET.Element
        XML ElementTree to pretty print.
    level: int, optional
        Level to start the indentation a, defaults to 0

    Returns
    -------
    et_element : ET.Element
        ET structure of the directory
    """

    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

    return


def modify_pyldavis_html(model_dir):
    """
    Modifies the PyLDAvis HTML file returned by the Gensim library to include the direct paths of the 'd3.js' and 'ldavis.v3.0.0.js', which are copied into the model/submodel directory.

    Parameters
    ----------
    model_dir: str
        String representation of the path wwhere the model/submodel is located
    """

    # Copy necessary files in model / submodel folder for PyLDAvis visualization
    d3 = pathlib.Path("src/gui/resources/d3.js")
    v3 = pathlib.Path("src/gui/resources/ldavis.v3.0.0.js")
    shutil.copyfile(d3, pathlib.Path(model_dir, "d3.js"))
    shutil.copyfile(v3, pathlib.Path(model_dir, "ldavis.v3.0.0.js"))

    # Update d3 and v3 paths in pyldavis.html
    fin = open(pathlib.Path(model_dir, "pyLDAvis.html").as_posix(),
               "rt")  # read input file
    data = fin.read()  # read file contents to string
    # Replace all occurrences of the required string
    data = data.replace(
        "https://d3js.org/d3.v5.js", "d3.js")
    data = data.replace(
        "https://d3js.org/d3.v5", "d3.js")
    data = data.replace(
        "https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", "ldavis.v3.0.0.js")
    fin.close()  # close the input file
    fin = open(pathlib.Path(model_dir, "pyLDAvis.html").as_posix(),
               "wt")  # open the input file in write mode
    fin.write(data)  # overrite the input file with the resulting data
    fin.close()  # close the file

    return
