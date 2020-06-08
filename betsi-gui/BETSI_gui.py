# basics
import sys

from PyQt5 import QtCore
from PyQt5 import QtGui
# qt gui framework
from PyQt5.QtWidgets import *
# matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
import sys

import traceback

# core lib
plt.ion()
from BETSI_lib import *


class BETSI_gui(QMainWindow):
    """Defines the main window of the BETSI gui and adds all the toolbars.
    The main widget is the BETSI_widget"""

    def __init__(self):
        super().__init__()

        self.title = 'BETSI'
        self.setWindowTitle(self.title)

        # define the menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        toolsMenu = menubar.addMenu('&Tools')

        # Option for importing a single file.
        importfile_menu = fileMenu.addAction('&Import File')
        importfile_menu.setShortcut("Ctrl+I")
        importfile_menu.triggered.connect(self.import_file)

        # Option for changing the root output directory.
        outputdir_menu = fileMenu.addAction('&Set Output Directory')
        outputdir_menu.setShortcut("Ctrl+O")
        outputdir_menu.triggered.connect(self.set_output_dir)

        # Option for running on a chosen directory.
        directoryrun_menu = fileMenu.addAction('&Analyse Directory')
        directoryrun_menu.setShortcut("Ctrl+D")
        directoryrun_menu.triggered.connect(self.analyse_dir)

        # Option for clearing the memory and resetting to defaults
        clear_menu = toolsMenu.addAction('&Clear')
        clear_menu.setShortcut("Ctrl+C")
        clear_menu.triggered.connect(self.clear)
        # Option for replotting
        replot_menu = toolsMenu.addAction('&Replot')
        replot_menu.triggered.connect(self.replot_betsi)

        self.betsimagic_menu_action = QAction(
            'BETSI Magic', self, checkable=True)
        self.betsimagic_menu_action.setStatusTip(
            "Resets the settings to default values and performs BETSI calculation")
        self.betsimagic_menu_action.setChecked(False)
        self.betsimagic_menu_action.triggered.connect(self.trigger_betsimagic)

        betsimagic_menu = toolsMenu.addAction(self.betsimagic_menu_action)

        # define the widget
        self.betsi_widget = BETSI_widget(self)
        self.setCentralWidget(self.betsi_widget)

        # enable drag and drop
        self.setAcceptDrops(True)

    def set_output_dir(self):
        self.betsi_widget.set_output_dir()

    def analyse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 'Select Directory', os.getcwd())
        print(f'Imported directory: {Path(dir_path).name}')
        self.betsi_widget.analyse_directory(dir_path)

    def import_file(self):
        file_path = QFileDialog.getOpenFileName(
            self, 'Select File', os.getcwd(), '*.csv')[0]
        self.betsi_widget.target_filepath = file_path
        self.betsi_widget.populate_table(csv_path=file_path)
        print(f'Imported file: {Path(file_path).name}')

    def dragEnterEvent(self, e):
        data = e.mimeData()
        urls = data.urls()
        drag_type = [u.scheme() for u in urls]
        paths = [u.toLocalFile() for u in urls]
        extensions = [os.path.splitext(p)[-1] for p in paths]
        # accept files only for now.
        if all(dt == 'file' for dt in drag_type) and all(ext == '.csv' for ext in extensions):
            e.accept()
        elif len(drag_type) == 1 and os.path.isdir(paths[0]):
            e.ignore()
        else:
            e.ignore()

    def dropEvent(self, e):
        data = e.mimeData()
        urls = data.urls()
        paths = [u.toLocalFile() for u in urls]

        # Single path to csv file
        if len(paths) == 1 and Path(paths[0]).suffix == '.csv':
            self.betsi_widget.target_filepath = paths[0]
            self.betsi_widget.populate_table(csv_path=paths[0])
            print(f'Imported file: {Path(paths[0]).name}')
            self.betsi_widget.run_calculation()

    def clear(self):
        self.betsi_widget.clear()

    def replot_betsi(self):
        self.betsi_widget.run_calculation()
        #self.bet_widget.plot_RD()

    def trigger_betsimagic(self, state):
        if state:
            self.betsi_widget.set_defaults()
        self.betsi_widget.set_editable(not state)

    def closeEvent(self, evt):
        print('Closing')
        try:
            plt.close(self.bet_widget.current_fig)
            plt.close(self.bet_widget.current_fig_2)
        except:
            pass


class BETSI_widget(QWidget):
    """Widget containing all the options for running the BETSI analysis and display of the data.

    Args:
        parent: QMainWindow the widget belongs to

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # basic properties
        self.output_dir = os.getcwd()
        self.current_fig = None
        self.current_fig_2 = None

        self.target_filepath = None
        self.bet_object = None
        self.bet_filter_result = None

        # Create label boxes to display info
        self.current_output_label = QLabel()
        self.current_targetpath_label = QLabel()

        self.current_output_label.setText(
            f"Output Directory: {self.output_dir}")
        self.current_targetpath_label.setText(
            f"Loaded File: {self.target_filepath}")
        self.current_output_label.setFont(QtGui.QFont("Times", 10))
        self.current_targetpath_label.setFont(QtGui.QFont("Times", 10))

        # add a group box containing controls
        self.criteria_box = QGroupBox("BET area selection criteria")
        self.criteria_box.setMaximumWidth(500)
        self.criteria_box.setMaximumHeight(800)
        self.min_points_label = QLabel(self.criteria_box)
        self.min_points_label.setText('Minimum number of points in the linear region:')
        self.min_points_edit = QLineEdit()
        self.min_points_edit.setMaximumWidth(75)
        self.min_points_slider = QSlider(QtCore.Qt.Horizontal)
        self.minr2_label = QLabel('Minimum R<sup>2</sup>:')
        self.minr2_edit = QLineEdit()
        self.minr2_edit.setMaximumWidth(75)
        self.minr2_slider = QSlider(QtCore.Qt.Horizontal)
        self.rouq1_tick = QCheckBox("Rouquerol criteria 1: Monotonic")
        self.rouq2_tick = QCheckBox("Rouquerol criteria 2: Positive C")
        self.rouq3_tick = QCheckBox("Rouquerol criteria 3: Pressure in linear range")
        self.rouq4_tick = QCheckBox("Rouquerol criteria 4: Small error")
        self.rouq5_tick = QCheckBox("Rouquerol criteria 5: End at the knee")
        self.rouq4_edit = QLineEdit()
        self.rouq4_edit.setMaximumWidth(75)
        self.rouq4_slider = QSlider(QtCore.Qt.Horizontal)
        self.export_button = QPushButton('Export Results')
        self.export_button.pressed.connect(self.export)

        # Any change in states updates the display.
        self.rouq1_tick.stateChanged.connect(self.maybe_run_calculation)
        self.rouq2_tick.stateChanged.connect(self.maybe_run_calculation)
        self.rouq3_tick.stateChanged.connect(self.maybe_run_calculation)
        self.rouq4_tick.stateChanged.connect(self.maybe_run_calculation)
        self.rouq5_tick.stateChanged.connect(self.maybe_run_calculation)

        # define widget parameters
        self.rouq4_slider.setRange(5, 75)
        self.minr2_slider.setRange(800, 999)
        self.min_points_slider.setRange(3, 10)

        # set the defaults
        self.set_defaults()

        # connect the actions
        self.minr2_edit.editingFinished.connect(self.minr2_edit_changed)
        self.minr2_edit.returnPressed.connect(self.minr2_edit_changed)
        self.rouq4_edit.editingFinished.connect(self.rouq4_edit_changed)
        self.rouq4_edit.returnPressed.connect(self.rouq4_edit_changed)
        self.min_points_edit.editingFinished.connect(
            self.min_points_edit_changed)
        self.min_points_edit.returnPressed.connect(
            self.min_points_edit_changed)
        self.rouq4_slider.valueChanged.connect(self.rouq4_slider_changed)
        self.minr2_slider.valueChanged.connect(self.minr2_slider_changed)
        self.min_points_slider.valueChanged.connect(
            self.min_points_slider_changed)

        # add the table for results
        self.results_table = QTableWidget(self)
        self.results_table.setFixedWidth(520)
        self.clean_table()

        # create layout
        criteria_layout = QGridLayout()
        criteria_layout.addWidget(
            self.min_points_label, criteria_layout.rowCount(), 1, 1, 1)
        criteria_layout.addWidget(
            self.min_points_edit, criteria_layout.rowCount() - 1, 2)
        criteria_layout.addWidget(
            self.min_points_slider, criteria_layout.rowCount(), 1, 1, 2)
        criteria_layout.addWidget(
            self.minr2_label, criteria_layout.rowCount(), 1)
        criteria_layout.addWidget(
            self.minr2_edit, criteria_layout.rowCount() - 1, 2)
        criteria_layout.addWidget(
            self.minr2_slider, criteria_layout.rowCount(), 1, 1, 2)
        criteria_layout.addWidget(
            self.rouq1_tick, criteria_layout.rowCount(), 1)
        criteria_layout.addWidget(
            self.rouq2_tick, criteria_layout.rowCount(), 1)
        criteria_layout.addWidget(
            self.rouq3_tick, criteria_layout.rowCount(), 1)
        criteria_layout.addWidget(
            self.rouq4_tick, criteria_layout.rowCount(), 1)
        criteria_layout.addWidget(
            self.rouq4_edit, criteria_layout.rowCount() - 1, 2)
        criteria_layout.addWidget(
            self.rouq4_slider, criteria_layout.rowCount(), 1, 1, 2)
        criteria_layout.addWidget(
            self.rouq5_tick, criteria_layout.rowCount(), 1)
        criteria_layout.addWidget(
            self.export_button, criteria_layout.rowCount(), 1, 1, 2)

        criteria_layout.addWidget(
            self.current_output_label, criteria_layout.rowCount(), 1, 1, 1)
        criteria_layout.addWidget(
            self.current_targetpath_label, criteria_layout.rowCount(), 1, 1, 1)

        criteria_layout.addItem(QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), criteria_layout.rowCount() + 1, 1)
        self.criteria_box.setLayout(criteria_layout)

        main_layout = QGridLayout()
        main_layout.addWidget(self.criteria_box, 1, 1)
        main_layout.addWidget(self.results_table, 1, 2)

        self.setLayout(main_layout)

    def set_output_dir(self):
        """Defines the output directory of the BETSI analysis"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 'Select Output Directory', os.getcwd())
        print(f"Output directory set to {dir_path}")
        self.output_dir = dir_path
        self.current_output_label.setText(
            f"Output Directory: {self.output_dir}")

    def maybe_run_calculation(self):
        self.check_rouq_compatibility()
        if self.target_filepath is not None:
            self.run_calculation()

    def plot_bet(self):
        if self.current_fig is not None:
            self.current_fig.clear()
            self.current_fig_2.clear()
        try:
            # check if the figure has been closed, if it doesn't reset it to none and replot
            if self.current_fig is not None and not plt.fignum_exists(self.current_fig.number):
                self.current_fig = None
                self.current_fig_2=None
            fig = create_matrix_plot(self.bet_filter_result, self.rouq3_tick.isChecked(), self.rouq4_tick.isChecked(),  name=Path(
                self.target_filepath).stem,  fig=self.current_fig)
            fig_2 = regression_diagnostics_plots(self.bet_filter_result, name=Path(
                self.target_filepath).stem, fig_2=self.current_fig_2)
            # connect the picker event to the figure
            if self.current_fig is None:
                fig.canvas.mpl_connect('pick_event', self.point_picker)
            self.current_fig = fig
            self.current_fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.95])
            self.current_fig_2 = fig_2
            #self.current_fig_2.tight_layout(pad=.3, rect=[0,0,1,.95])
            plt.figure(num=1)
            plt.draw()
            plt.figure(num=2)
            plt.draw()
        except TypeError:
            # traceback.print_exc()
            pass
        
#    def plot_RD(self):
#        if self.current_fig_2 is not None:
#            self.current_fig_2.clear()
#        try:
#            if self.current_fig_2 is not None and not plt.fignum_exists(self.current_fig_2.number):
#                self.current_fig_2=None
#                fig_2 = regression_diagnostics_plots(self.bet_filter_result, name=Path(
#                self.target_filepath).stem, fig_2=self.current_fig_2)
#            # connect the picker event to the figure
#            if self.current_fig_2 is None:
#                fig_2 = regression_diagnostics_plots(self.bet_filter_result, name=Path(
#                self.target_filepath).stem, fig_2=self.current_fig_2)
#                fig.canvas.mpl_connect('pick_event', self.point_picker)
#            self.current_fig_2 = fig_2
#            self.current_fig_2.tight_layout()
#            plt.draw()
#
#        except TypeError:
#            pass
        
    def run_calculation(self):
        """ Applies the currently specified filters to the currently specified target csv file. """

        assert self.target_filepath, "You must provide a csv file before calling run."

        use_rouq1 = self.rouq1_tick.isChecked()
        use_rouq2 = self.rouq2_tick.isChecked()
        use_rouq3 = self.rouq3_tick.isChecked()
        use_rouq4 = self.rouq4_tick.isChecked()
        use_rouq5 = self.rouq5_tick.isChecked()
        min_num_pts = int(self.min_points_edit.text())
        min_r2 = float(self.minr2_edit.text())
        max_perc_error = float(self.rouq4_edit.text())

        # Retrieve the Betsi Results Object if non-existent
        if self.bet_object is None:
            pressure, q_adsorbed = get_data(input_file=self.target_filepath)
            self.bet_object = BETResult(pressure, q_adsorbed)

        # Apply the currently selected filters.
        self.bet_filter_result = BETFilterAppliedResults(self.bet_object,
                                                         min_num_pts=min_num_pts,
                                                         min_r2=min_r2,
                                                         use_rouq1=use_rouq1,
                                                         use_rouq2=use_rouq2,
                                                         use_rouq3=use_rouq3,
                                                         use_rouq4=use_rouq4,
                                                         use_rouq5=use_rouq5,
                                                         max_perc_error=max_perc_error)
        self.plot_bet()
        #self.plot_RD()

    def point_picker(self, event):
        line = event.artist
        picked_coords = line.get_offsets()[event.ind][0]
        # redefine min_i and min_j
        self.bet_filter_result.find_nearest_idx(picked_coords)
        #self.compute_BET_curve
        # replot based on the new min_i and min_j
        self.plot_bet()
        #self.plot_RD()

    def export(self):
        """ Print out the plot, filter config and results to the output directory. """
        if self.bet_filter_result is not None:

            # Create a local sub-directory for export.
            target_path = Path(self.target_filepath)
            output_subdir = Path(self.output_dir) / target_path.name
            output_subdir.mkdir(exist_ok=True)

            self.bet_filter_result.export(output_subdir)

            # self.current_fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.95])
            self.current_fig.savefig(
                str(output_subdir / f'{target_path.stem}_plot.pdf'), bbox_inches='tight')
            plt.show()
            self.current_fig_2.savefig(
                    str(output_subdir / f'{target_path.stem}_RD_plots.pdf'))
            plt.show()

    def analyse_directory(self, dir_path):
        """ Run Betsi on all csv files within dir_path. Use current filter config."""
        use_rouq1 = self.rouq1_tick.isChecked()
        use_rouq2 = self.rouq2_tick.isChecked()
        use_rouq3 = self.rouq3_tick.isChecked()
        use_rouq4 = self.rouq4_tick.isChecked()
        use_rouq5 = self.rouq5_tick.isChecked()
        min_num_points = int(self.min_points_edit.text())
        min_r2 = float(self.minr2_edit.text())
        max_perc_error = float(self.rouq4_edit.text())

        csv_paths = Path(dir_path).glob('*.csv')

        for file_path in csv_paths:
            # Update the table with current file
            self.populate_table(csv_path=str(file_path))

            # Run the analysis
            analyse_file(input_file=str(file_path),
                         output_dir=self.output_dir,
                         min_num_pts=min_num_points,
                         min_r2=min_r2,
                         use_rouq1=use_rouq1,
                         use_rouq2=use_rouq2,
                         use_rouq3=use_rouq3,
                         use_rouq4=use_rouq4,
                         use_rouq5=use_rouq5,
                         max_perc_error=max_perc_error)

    def set_defaults(self):
        """Sets the widget to defualt state
        """
        # set default values for the tick marks
        self.rouq1_tick.setCheckState(True)
        self.rouq2_tick.setCheckState(True)
        self.rouq3_tick.setCheckState(True)
        self.rouq4_tick.setCheckState(True)
        self.rouq5_tick.setCheckState(False)

        # the ticks can only be on or off - Not sure why I need to do this every time, but doesn matter
        self.rouq1_tick.setTristate(False)
        self.rouq2_tick.setTristate(False)
        self.rouq3_tick.setTristate(False)
        self.rouq4_tick.setTristate(False)
        self.rouq5_tick.setTristate(False)
        # set defaults for text fields
        self.minr2_edit.setText('0.995')
        self.rouq4_edit.setText('20')
        self.min_points_edit.setText('10')
        # trigger the corresponding sliders
        self.rouq4_edit_changed()
        self.minr2_edit_changed()
        self.min_points_edit_changed()
        # check the compatibility
        self.check_rouq_compatibility()
        # if there is a figure, replot
        self.plot_bet()

    def clear(self):
        """Closes all plots and removes all data from memory"""
        # remove the bet filter result from memory
        self.bet_filter_result = None
        self.bet_object = None
        # clear the table
        self.clean_table()
        # close the plot
        if self.current_fig is not None:
            plt.close(fig=self.current_fig)
            plt.close(fig=self.current_fig_2)

    def set_editable(self, state):
        if state:
            self.rouq1_tick.setEnabled(True)
            self.rouq2_tick.setEnabled(True)
            self.rouq3_tick.setEnabled(True)
            self.rouq4_tick.setEnabled(True)
            self.rouq5_tick.setEnabled(True)
            self.minr2_edit.setEnabled(True)
            self.rouq4_edit.setEnabled(True)
            self.min_points_edit.setEnabled(True)
            self.rouq4_slider.setEnabled(True)
            self.minr2_slider.setEnabled(True)
            self.min_points_slider.setEnabled(True)
        else:
            self.rouq1_tick.setEnabled(False)
            self.rouq2_tick.setEnabled(False)
            self.rouq3_tick.setEnabled(False)
            self.rouq4_tick.setEnabled(False)
            self.rouq5_tick.setEnabled(False)
            self.minr2_edit.setEnabled(False)
            self.rouq4_edit.setEnabled(False)
            self.min_points_edit.setEnabled(False)
            self.rouq4_slider.setEnabled(False)
            self.minr2_slider.setEnabled(False)
            self.min_points_slider.setEnabled(False)

    def check_rouq_compatibility(self):
        use_rouq1 = self.rouq1_tick.isChecked()
        use_rouq2 = self.rouq2_tick.isChecked()
        use_rouq3 = self.rouq3_tick.isChecked()
        use_rouq4 = self.rouq4_tick.isChecked()
        use_rouq5 = self.rouq5_tick.isChecked()
        if not (use_rouq3 or use_rouq4):
            self.rouq2_tick.setEnabled(True)
        else:
            self.rouq2_tick.setChecked(True)
            self.rouq2_tick.setEnabled(False)

            
    def populate_table(self, csv_path):
        self.results_table.setColumnCount(2)
        self.results_table.setRowCount(1)
        self.results_table.setHorizontalHeaderLabels(
            ['Relative pressure (p/p\u2080)', 'Quantity adsorbed (cm\u00B3/g)'])
        self.results_table.setColumnWidth(0, 250)
        self.results_table.setColumnWidth(1, 250)

        # Change the box title
        self.current_targetpath_label.setText(
            f"Loaded File: {self.target_filepath}")

        if csv_path is not None and Path(csv_path).exists():
            pressure, q_adsorbed = get_data(input_file=csv_path)
            self.results_table.setRowCount(len(pressure))
            for i in range(len(pressure)):
                self.results_table.setItem(
                    i, 0, QTableWidgetItem(str(pressure[i])))
                self.results_table.setItem(
                    i, 1, QTableWidgetItem(str(q_adsorbed[i])))

    def clean_table(self):
        """Cleans the table"""
        self.results_table.setColumnCount(2)
        self.results_table.setRowCount(0)
        self.results_table.setHorizontalHeaderLabels(
            ['Relative pressure (p/p\u2080)', 'Quantity adsorbed (cm\u00B3/g)'])
        self.results_table.setColumnWidth(0, 250)
        self.results_table.setColumnWidth(1, 250)


    def minr2_edit_changed(self):
        value = self.minr2_edit.text()
        mn = self.minr2_slider.minimum()
        mx = self.minr2_slider.maximum()
        try:
            value = int(round(float(value) * 1000))
            if value < mn:
                value = mn
                self.minr2_edit.setText(str(value / 1000))
            elif value > mx:
                value = mx
                self.minr2_edit.setText(str(value / 1000))
            self.minr2_slider.setValue(value)
        except (ValueError, TypeError):
            self.minr2_edit.setText('0.995')
        self.maybe_run_calculation()

    def rouq4_edit_changed(self):
        value = self.rouq4_edit.text()
        mn = self.rouq4_slider.minimum()
        mx = self.rouq4_slider.maximum()
        try:
            value = int(float(value))
            if value < mn:
                value = mn
                self.rouq4_edit.setText(str(value))
            if value > mx:
                value = mx
                self.rouq4_edit.setText(str(value))
            self.rouq4_slider.setValue(value)
        except (ValueError, TypeError):
            self.rouq4_edit.setText('20')
        self.maybe_run_calculation()

    def min_points_edit_changed(self):
        value = self.min_points_edit.text()
        mn = self.min_points_slider.minimum()
        mx = self.min_points_slider.maximum()
        try:
            value = int(round(float(value)))
            if value < mn:
                value = mn
                self.min_points_edit.setText(str(value))
            if value > mx:
                value = mx
                self.min_points_edit.setText(str(value))
            self.min_points_slider.setValue(value)
        except (ValueError, TypeError):
            self.min_points_edit.setText('10')
        self.maybe_run_calculation()

    def rouq4_slider_changed(self):
        value = self.rouq4_slider.value()
        self.rouq4_edit.setText(str(value))
        self.maybe_run_calculation()

    def minr2_slider_changed(self):
        value = self.minr2_slider.value()
        self.minr2_edit.setText(str(value / 1000))
        self.maybe_run_calculation()

    def min_points_slider_changed(self):
        value = self.min_points_slider.value()
        self.min_points_edit.setText(str(value))
        self.maybe_run_calculation()

    def __del__(self):
        try:
            plt.close(self.current_fig_2)
            plt.close(self.current_fig)
        except:
            pass
        # sys.stdout = sys.__stdout__


class OutputCanvas(FigureCanvas):
    def __init__(self, parent, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.fig.subplots_adjust(left=0.05, right=0.95)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class OutputCanvas_2(FigureCanvas):
    def __init__(self,parent,dpi=100):
        self.fig_2 = Figure(dpi=dpi)
        self.fig_2.subplots_adjust(left=.05, right=.95)
        FigureCanvas.__init__(self,self.fig_2)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class OutLog(logging.Handler):
    def __init__(self, out_widget):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        """
        logging.Handler.__init__(self)
        self.out_widget = out_widget

    def emit(self, message):
        self.write(message.getMessage() + '\n')

    def write(self, m):
        self.out_widget.moveCursor(QtGui.QTextCursor.End)
        self.out_widget.insertPlainText(m)


if __name__ == '__main__':
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    ex = BETSI_gui()
    ex.show()
    sys.exit(app.exec_())
