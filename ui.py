# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from switch import Switch
import torch

class Ui_extras(object):
    def drawGpuSwitch(self, MainWindow):
        MainWindow.GpuSwitch = Switch(thumb_radius=8, track_radius=10, show_text = False)
        MainWindow.horizontalLayout.addWidget(MainWindow.GpuSwitch)
        MainWindow.GpuSwitch.setEnabled(torch.cuda.is_available())
        MainWindow.use_cuda = False
        MainWindow.GpuSwitch.setToolTip("<h4>CUDA installed: {}</h4>".format(torch.cuda.is_available()))

    def initWidgets(self, MainWindow):
        MainWindow.TTSStopButton.setDisabled(True)
        MainWindow.progressBar2Label.setText('')
        MainWindow.progressBarLabel.setText('')
        MainWindow.ClientStopBtn.setDisabled(True)
        MainWindow.ClientSkipBtn.setDisabled(True)
        MainWindow.TTModelCombo.setDisabled(True)
        MainWindow.WGModelCombo.setDisabled(True)
        MainWindow.TTSDialogButton.setDisabled(True)
        MainWindow.tab_2.setDisabled(True)
        MainWindow.log_window2.ensureCursorVisible()
        MainWindow.label_10.setDisabled(True)
        MainWindow.OptLimitCpuCombo.setDisabled(True)

        MainWindow.OptLimitCpuCombo.addItems(
            [str(i) for i in range(1,torch.get_num_threads()+1)])
        MainWindow.OptLimitCpuCombo.setCurrentIndex(torch.get_num_threads()-1)

    def setUpconnections(self,MainWindow):
        # Static widget signals
        MainWindow.TTModelCombo.currentIndexChanged.connect(MainWindow.set_reload_model_flag)
        MainWindow.WGModelCombo.currentIndexChanged.connect(MainWindow.set_reload_model_flag)
        MainWindow.TTSDialogButton.clicked.connect(MainWindow.start_synthesis)
        MainWindow.TTSStopButton.clicked.connect(MainWindow.skip_infer_playback)
        MainWindow.LoadTTButton.clicked.connect(MainWindow.add_TTmodel_path)
        MainWindow.LoadWGButton.clicked.connect(MainWindow.add_WGmodel_path)
        MainWindow.ClientSkipBtn.clicked.connect(MainWindow.skip_eventloop)
        MainWindow.ClientStartBtn.clicked.connect(MainWindow.start_eventloop)
        MainWindow.ClientStopBtn.clicked.connect(MainWindow.stop_eventloop)
        MainWindow.OptLimitCpuBtn.stateChanged.connect(MainWindow.toggle_cpu_limit)
        MainWindow.OptLimitCpuCombo.currentIndexChanged.connect(MainWindow.change_cpu_limit)
        MainWindow.OptApproveDonoBtn.stateChanged.connect(MainWindow.toggle_approve_dono)
        MainWindow.OptBlockNumberBtn.stateChanged.connect(MainWindow.toggle_block_number)
        MainWindow.OptDonoNameAmountBtn.stateChanged.connect(MainWindow.toggle_dono_amount)
        # Instantiated widget signals
        MainWindow.GpuSwitch.toggled.connect(MainWindow.set_cuda)
        # Instantiated signals
        MainWindow.signals.progress.connect(MainWindow.update_log_bar)
        MainWindow.signals.elapsed.connect(MainWindow.on_elapsed)





