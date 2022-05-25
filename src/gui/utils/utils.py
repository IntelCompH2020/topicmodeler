from PyQt6.QtWidgets import QTableWidget, QProgressBar

from src.gui.utils.worker import Worker


def configure_table_header(list_tables, window):
    for table_name in list_tables:
        table_widget = window.findChild(QTableWidget, table_name)
        table_widget.horizontalHeader().setVisible(True)
        table_widget.horizontalHeader().setHighlightSections(False)
        table_widget.resizeColumnsToContents()


def initialize_progress_bar(list_progress_bars, window):
    for progress_bar in list_progress_bars:
        progress_bar_widget = window.findChild(QProgressBar, progress_bar)
        progress_bar_widget.setVisible(False)
        progress_bar_widget.setValue(0)


def execute_in_thread(gui, function, function_output, progress_bar):
    """
    Method to execute a function in the secondary thread while showing
    a progress bar at the time the function is being executed if a progress bar object is provided.
    When finished, it forces the execution of the method to be
    executed after the function executing in a thread is completed.
    Based on the functions provided in the manual available at:
    https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/

    Parameters
    ----------
    gui: MainWindow
        ...
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
