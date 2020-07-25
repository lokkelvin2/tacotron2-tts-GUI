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
        MainWindow.TTSSkipButton.setDisabled(True)
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



    
