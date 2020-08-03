import sys
from PyQt5 import Qt
from PyQt5 import QtCore,QtGui
from PyQt5.QtCore import QMutex, QObject, QRunnable, pyqtSignal, pyqtSlot, QThreadPool, QTimer, QThread
from PyQt5.QtWidgets import QWidget,QMainWindow,QHeaderView, QMessageBox, QFileDialog
from nvidia_tacotron_TTS_Layout import Ui_MainWindow
from ui import Ui_extras
from timerthread import timerThread
from preprocess import preprocess_text

import time
import requests
import json
import datetime
import numpy as np
import os
import pygame

import sys
sys.path.append(os.path.join(sys.path[0],'waveglow/'))

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence, cleaners
#from denoiser import Denoiser

#from secrets import TOKEN # for debugging

_mutex1 = QMutex()
_running1 = False # tab 0 synthesis QThread : Start/stop
_mutex2 = QMutex()
_running2 = False # tab 1 eventloop QRunnable: Start/stop
_mutex3 = QMutex()
_running3 = False # tab 1 eventloop QRunnable: Skip current item

#https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''

    textready = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    elapsed = pyqtSignal(int)
    fncallback = pyqtSignal(tuple)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['elapsed_callback'] = self.signals.elapsed
        self.kwargs['text_ready'] = self.signals.textready
        self.kwargs['fn_callback'] = self.signals.fncallback

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            pass
            # traceback.print_exc()
            # exctype, value = sys.exc_info()[:2]
            # self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class GUISignals(QObject):
    progress = pyqtSignal(int)
    elapsed = pyqtSignal(int)

class GUI(QMainWindow, Ui_MainWindow, Ui_extras):
    def __init__(self,app):
        super(GUI, self).__init__()
        self.app = app

        ### Setup UI and signals
        self.setupUi(self)
        self.drawGpuSwitch(self)
        self.initWidgets(self)
        self.signals = GUISignals()
        self.setUpconnections(self)

        ### Init vars
        self.model = None
        self.waveglow = None
        self.hparams = None
        self.current_thread = None
        self.t_1 = None # timing
        self.logs = [] # message logs
        self.logs2 = []
        self.max_log_lines = 3
        self.max_log2_lines = 100
        self.TTmodel_dir = [] # list of model paths
        self.WGmodel_dir = []
        self.reload_model_flag = True
        self.channel_id = '' # stream elements channel ID
        # Because of bug in streamelements timestamp filter, need 2 variables for previous time
        self.startup_time = datetime.datetime.utcnow().isoformat()
        #self.startup_time = '0' # For debugging
        self.prev_time = datetime.datetime.utcnow().isoformat()
        #self.prev_time = '0' # for debugging
        self.msg_offset = 0
        self.se_opts = {'approve only': 2, # Stream element options
                        'block large numbers': 0,
                        'read dono amount': 2,
                        }
        self.fns = {'GUI: start of polling loop': self.fns_gui_startpolling, # Callback functions
                    'GUI: end of polling loop': self.fns_gui_endpolling ,
                    'Wav: playback' : self.fns_wav_playback,
                    'Var: offset': self.fns_var_offset,
                    'Var: prev_time': self.fns_var_prevtime,
                    'GUI: progress bar 2 text' : self.fns_gui_pbtext,
                    'GUI: reenable skip btn' : self.fns_gui_enableclientskipbtn}
        self.pyt_opts = {'cpu limit': None, # pytorch options
                        'denoiser':None}
        
        ### Init pygame mixer
        pygame.mixer.quit()
        pygame.mixer.init(frequency=22050,size=-16, channels=1)
        self.channel = pygame.mixer.Channel(0)
        
        ### Init qthreadpool
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        ### Setup Complete
        self.update_log_window("Begin by loading a model")

    @pyqtSlot(int)
    def toggle_cpu_limit(self, state):
        self.label_10.setEnabled(state)
        self.OptLimitCpuCombo.setEnabled(state)

    @pyqtSlot(int)
    def change_cpu_limit(self, indx):
        num_thread = indx + 1
        self.pyt_opts['cpu limit'] = num_thread

    @pyqtSlot(int)
    def toggle_approve_dono(self, state):
        self.se_opts['approve only'] = state

    @pyqtSlot(int)
    def toggle_block_number(self, state):
        self.se_opts['block large numbers'] = state

    @pyqtSlot(int)
    def toggle_dono_amount(self, state):
        self.se_opts['read dono amount'] = state

    @pyqtSlot(tuple)
    def on_fncallback(self,tup):
        option,arg = tup
        self.fns[option](arg)

    @pyqtSlot(str)
    def on_textready(self,text):
        # Function to send text from client thread to GUI thread
        # Format of text: <Obj>:<Message>
        obj = text[0:4]
        msg = text[5:]
        if obj=='Log1':
            if len(self.logs) > self.max_log_lines:
                self.logs.pop(0)
            self.logs.append(msg)
            log_text = '\n'.join(self.logs)
            self.log_window1.setText(log_text)
        if obj=='Log2':
            if len(self.logs2) > self.max_log2_lines:
                self.logs2.pop(0)
            self.logs2.append(msg)
            log_text = '\n'.join(self.logs2)
            self.log_window2.setPlainText(log_text)
            self.log_window2.verticalScrollBar().setValue(
                self.log_window2.verticalScrollBar().maximum())
        if obj=='Sta2':
            self.statusbar.setText(msg)

    @pyqtSlot(int)
    def update_log_bar(self,val):
        self.progressBar.setValue(val)
        #self.progressBar.setTextVisible(val != 0)

    @pyqtSlot(int)
    def update_log_bar2(self,val):
        self.progressBar2.setValue(val)
        #self.progressBar2.setTextVisible(val != 0)

    @pyqtSlot(int)
    def on_elapsed(self,val):
        if self.tabWidget.currentIndex()==0:
            self.update_log_window('Elapsed: '+str(val)+'s',mode='overwrite')
        else:
            pass # No elapsed time for tab2

    @pyqtSlot(np.ndarray)
    def on_inferThread_complete(self,wav):
        global _running1
        _mutex1.lock()
        _running1 = False
        _mutex1.unlock()
        self.playback_wav(wav)
        self.TTSDialogButton.setEnabled(True)
        self.TTModelCombo.setEnabled(True)
        self.WGModelCombo.setEnabled(True)
        self.TTSTextEdit.setEnabled(True)
        self.LoadTTButton.setEnabled(True)
        self.LoadWGButton.setEnabled(True)
        self.tab_2.setEnabled(True)
        elapsed = (time.time() - self.t_1)
        wav_length = (len(wav) / self.hparams.sampling_rate)
        rtf = elapsed / wav_length
        line = 'Generated {:.1f}s of audio in {:.1f}s ({:.2f} real-time factor)'.format(wav_length,elapsed,rtf)
        self.update_log_window(line,'overwrite')
        tps = elapsed / len(wav)
        print(" > Run-time: {}".format(elapsed))
        print(" > Real-time factor: {}".format(rtf))
        print(" > Time per step: {}".format(tps))
        self.update_status_bar("Ready")
        # TODO get pygame mixer callback on end or use sounddevice

    @pyqtSlot(tuple)
    def on_itersignal(self,tup):
        # Displays current iteration on progress bar
        current,total = tup
        self.progressBarLabel.setText('{}/{}'.format(current,total))

    @pyqtSlot()
    def on_interrupt(self):
        # Reenable buttons
        self.TTSDialogButton.setEnabled(True)
        self.TTModelCombo.setEnabled(True)
        self.WGModelCombo.setEnabled(True)
        self.TTSTextEdit.setEnabled(True)
        self.LoadTTButton.setEnabled(True)
        self.LoadWGButton.setEnabled(True)
        self.tab_2.setEnabled(True)
        # Refresh progress bar
        self.update_log_bar(0)
        self.progressBarLabel.setText('')
        # Write to log window
        self.update_log_window('Interrupted','overwrite')
        # Write to status bar
        self.update_status_bar("Ready")

    def fns_gui_startpolling(self,arg=None):
        self.ClientStartBtn.setDisabled(True)
        self.ClientStopBtn.setEnabled(True)
        self.ClientSkipBtn.setEnabled(True)
        self.tab.setDisabled(True)
        self.tab_3.setDisabled(True)
        self.ClientAmountLine.setDisabled(True)

    def fns_gui_endpolling(self,arg=None):
        self.update_log_bar2(0)
        self.progressBar2Label.setText('')
        self.ClientStartBtn.setEnabled(True)
        self.ClientStopBtn.setDisabled(True)
        self.ClientSkipBtn.setDisabled(True)
        self.tab.setEnabled(True)
        self.tab_3.setEnabled(True)
        self.ClientAmountLine.setEnabled(True)

    def fns_wav_playback(self,wav):
        if self.tabWidget.currentIndex()==0:
            self.TTSStopButton.setEnabled(True)
        else:
            self.ClientSkipBtn.setEnabled(True)
        if wav.dtype != np.int16 :
            # Convert from float32 or float16 to signed int16 for pygame
            wav = (wav/np.amax(wav) * 32767).astype(np.int16)
        sound = pygame.mixer.Sound(wav)
        self.channel.queue(sound)

    def fns_var_offset(self,arg):
        self.msg_offset = arg

    def fns_var_prevtime(self,arg):
        self.prev_time = arg

    def fns_gui_pbtext(self,tup):
        current,total = tup
        self.progressBar2Label.setText('{}/{}'.format(current,total))

    def fns_gui_enableclientskipbtn(self,arg=None):
        self.ClientSkipBtn.setEnabled(True)

    def on_finished(self):
        #print("THREAD COMPLETE!")
        pass

    def on_result(self, s):
        #print(s)
        pass

    def start_eventloop(self):
        # Pass the function to execute
        global _running2,_running3
        if not self.validate_se():
            return
        if self.reload_model_flag:
            self.reload_model()
            self.reload_model_flag = False
        min_donation = self.get_min_donation()
        TOKEN = self.get_token()
        _mutex2.lock()
        _running2 = True
        _mutex2.unlock()
        _mutex3.lock()
        _running3 = True
        _mutex3.unlock()
        worker = Worker(self.eventloop, TOKEN, min_donation, self.channel,
                    self.se_opts, self.use_cuda, self.model, self.waveglow, self.pyt_opts['cpu limit'],
                    self.msg_offset, self.prev_time, self.startup_time)
                    # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.on_result)
        worker.signals.finished.connect(self.on_finished)
        worker.signals.progress.connect(self.update_log_bar2)
        worker.signals.textready.connect(self.on_textready)
        worker.signals.elapsed.connect(self.on_elapsed)
        worker.signals.fncallback.connect(self.on_fncallback)
        # Execute
        self.threadpool.start(worker)

    def stop_eventloop(self):
        global _running2, _running3
        _mutex2.lock()
        _running2 = False
        _mutex2.unlock()
        _mutex3.lock()
        _running3 = False
        _mutex3.unlock()
        self.skip_wav()

    def skip_eventloop(self):
        global _running3
        _mutex3.lock()
        _running3 = False
        _mutex3.unlock()
        self.skip_wav()

    def eventloop(self, TOKEN, min_donation, channel, se_opts,
                    use_cuda, model, waveglow, num_thread,
                    offset, prev_time, startup_time,
                    progress_callback, elapsed_callback, text_ready, fn_callback):
        # TODO: refactor this messy block
        global _running3
        if num_thread:
            torch.set_num_threads(num_thread)
            os.environ['OMP_NUM_THREADS'] = str(num_thread)
            os.environ['MKL_NUM_THREADS'] = str(num_thread)
        fn_callback.emit(('GUI: start of polling loop',None))
        text_ready.emit("Sta2:Connecting to StreamElements")
        url = "https://api.streamelements.com/kappa/v2/tips/"+self.channel_id
        headers = {'accept': 'application/json',"Authorization": "Bearer "+TOKEN}
        text_ready.emit('Log2:Initializing')
        text_ready.emit('Log2:Minimum amount for TTS: '+str(min_donation))
        while True:
            _mutex2.lock()
            if _running2 == False:
                _mutex2.unlock()
                break
            else:
                _mutex2.unlock()
            if not channel.get_busy():
                #print('Polling', datetime.datetime.utcnow().isoformat())
                text_ready.emit("Sta2:Waiting for incoming donations . . .")
                current_time = datetime.datetime.utcnow().isoformat()
                # TODO: possible bug: missed donations once time pasts midnight
                querystring = {"offset":offset,
                                "limit":"1",
                                "sort":"createdAt",
                                "after":startup_time,
                                "before":current_time}
                response = requests.request("GET", url, headers=headers, params=querystring)
                data = json.loads(response.text)
                for dono in data['docs']:
                    text_ready.emit("Sta2:Processing donations")
                    dono_time = dono['createdAt']
                    offset += 1
                    if dono_time > prev_time: # Str comparison
                        amount = dono['donation']['amount'] # Int
                        if float(amount) >= min_donation and dono['approved']=='allowed':
                            _mutex3.lock()
                            if not _running3: 
                                _running3 = True
                            _mutex3.unlock()
                            fn_callback.emit(('GUI: reenable skip btn',None))
                            name = dono['donation']['user']['username']
                            msg = dono['donation']['message']
                            if msg.isspace(): break # Check for empty line
                            ## TODO Allow multiple speaker in msg
                            currency = dono['donation']['currency']
                            dono_id = dono['_id']
                            text_ready.emit("Log2:\n###########################")
                            text_ready.emit("Log2:"+name+' donated '+currency+str(amount))
                            text_ready.emit("Log2:"+msg)
                            lines = preprocess_text(msg)
                            if se_opts['read dono amount'] == 1: # reads dono name and amount
                                msg = '{} donated {} {}.'.format(name,
                                                    str(amount),
                                                    cleaners.expand_currency(currency))
                                lines.insert(0,msg) # Add to head to list
                            output = []
                            for count, line in enumerate(lines):
                                fn_callback.emit(('GUI: progress bar 2 text', (count,len(lines))))
                                sequence = np.array(text_to_sequence(line, ['english_cleaners']))[None, :]
                                # Inference
                                device = torch.device('cuda' if use_cuda else 'cpu')
                                sequence = torch.autograd.Variable(
                                    torch.from_numpy(sequence)).to(device).long()
                                # Decode text input
                                mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
                                with torch.no_grad():
                                    audio = waveglow.infer(mel_outputs_postnet,
                                                            sigma=0.666,
                                                            progress_callback = progress_callback,
                                                            elapsed_callback = None,
                                                            get_interruptflag = self.get_interruptflag2)
                                    if type(audio) != torch.Tensor:
                                        # Catches when waveglow is interrupted and returns none
                                        break
                                    fn_callback.emit(('GUI: progress bar 2 text', (count+1,len(lines))))
                                    wav = audio[0].data.cpu().numpy()
                                output.append(wav)
                            _mutex3.lock()
                            if _running3 == True:
                                _mutex3.unlock()
                                outwav = np.concatenate(output)
                                # Playback
                                fn_callback.emit(('Wav: playback',outwav))
                            else: _mutex3.unlock()
                            prev_time = dono_time # Increment time
            time.sleep(0.5)
        fn_callback.emit(('GUI: end of polling loop',None))
        text_ready.emit('Log2:\nDisconnected')
        text_ready.emit('Sta2:Ready')
        fn_callback.emit(('Var: offset', offset))
        fn_callback.emit(('Var: prev_time', prev_time))
        return #'Return value of execute_this_fn'

    def startup_update(self):
        if not self.tab_2.isEnabled():
            self.tab_2.setEnabled(True)
        if not self.TTSDialogButton.isEnabled():
            self.TTSDialogButton.setEnabled(True)

    def playback_wav(self,wav):
        if self.tabWidget.currentIndex()==1:
            self.ClientSkipBtn.setEnabled(True)
        if wav.dtype != np.int16 :
            # Convert from float32 or float16 to signed int16 for pygame
            wav = (wav/np.amax(wav) * 32767).astype(np.int16)
        sound = pygame.mixer.Sound(wav)
        self.channel.queue(sound)
        # TODO Disable skip btn on playback end

    def skip_wav(self):
        if self.channel.get_busy():
            self.channel.stop()
        self.ClientSkipBtn.setDisabled(True)

    def skip_infer_playback(self):
        global _running1
        if self.channel.get_busy():
            self.channel.stop()
        _mutex1.lock()      # We could also use a signal/slot mechanism
        if _running1:
            self.progressBarLabel.setText('Interrupting...')
            _running1 = False   # instead of mutex since inference is on QThread
        _mutex1.unlock()
        self.TTSStopButton.setDisabled(True)

    def reload_model(self):
        TTmodel_fpath = self.get_current_TTmodel_dir()
        WGmodel_fpath = self.get_current_WGmodel_dir()
        # Setup hparams
        self.hparams = create_hparams()
        self.hparams.sampling_rate = 22050
        # Load Tacotron 2 from checkpoint
        self.model = load_model(self.hparams,self.use_cuda)
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model.load_state_dict(torch.load(TTmodel_fpath, map_location = device)['state_dict'])
        if self.use_cuda:
            _ = self.model.cuda().eval().half()
        else:
            _ = self.model.eval()
        #  Load WaveGlow for mel2audio synthesis and denoiser
        self.waveglow = torch.load(WGmodel_fpath, map_location = device)['model']
        self.waveglow.use_cuda = self.use_cuda
        if self.use_cuda:
            self.waveglow.cuda().eval().half()
        else:
            self.waveglow.eval()
        for k in self.waveglow.convinv:
            k.float()
        #denoiser = Denoiser(waveglow,use_cuda=self.use_cuda)

    def start_synthesis(self):
        # Runs in main gui thread. Synthesize blocks gui.
        # Can update gui directly in this function.
        text = self.TTSTextEdit.toPlainText()
        if text.isspace():return
        global _running1
        self.t_1 = time.time()
        self.TTSDialogButton.setDisabled(True)
        self.TTModelCombo.setDisabled(True)
        self.WGModelCombo.setDisabled(True)
        self.TTSTextEdit.setDisabled(True)
        self.LoadTTButton.setDisabled(True)
        self.LoadWGButton.setDisabled(True)
        self.TTSStopButton.setEnabled(True)
        self.tab_2.setDisabled(True)
        self.update_log_bar(0)
        self.update_log_window('Initializing','clear')
        self.update_status_bar("Creating voice")
        # We use a signal callback here to stick to the same params type in synthesize.py
        if self.reload_model_flag:
            self.reload_model()
            self.reload_model_flag = False
        # Prepare text input
        _mutex1.lock()
        _running1 = True
        _mutex1.unlock()
        self.current_thread = inferThread(text,
                                        self.use_cuda,
                                        self.model,
                                        self.waveglow,
                                        self.signals.progress,
                                        None,
                                        self.t_1,
                                        self.pyt_opts['cpu limit'],
                                        parent = self)
        self.current_thread.audioSignal.connect(self.on_inferThread_complete)
        self.current_thread.timeElapsed.connect(self.on_elapsed)
        self.current_thread.iterSignal.connect(self.on_itersignal)
        self.current_thread.interruptSignal.connect(self.on_interrupt)

    def validate_se(self):
        # Connect to streamelement and saves channel id
        # return true if chn id and token returns valid
        # Test Channel ID
        self.update_status_bar("Validating StreamElements")
        CHANNEL_NAME = ''.join(self.ChannelName.text().split())
        url = "https://api.streamelements.com/kappa/v2/channels/"+CHANNEL_NAME
        response = requests.request("GET", url, headers={'accept': 'application/json'})
        if response.status_code == 200:
            # Test JWT Token
            self.channel_id = json.loads(response.text)['_id']
            url = "https://api.streamelements.com/kappa/v2/tips/"+self.channel_id
            querystring = {"offset":"0","limit":"10","sort":"createdAt","after":"0","before":"0"}
            TOKEN = self.get_token()
            headers = {'accept': 'application/json',"Authorization": "Bearer "+TOKEN}
            response2 = requests.request("GET", url, headers=headers, params=querystring)
            if response2.status_code == 200:
                self.update_log_window_2("\nConnected to "+CHANNEL_NAME)
                return True
            else:
                self.update_log_window_2("\nError: Double check your token")
                self.update_status_bar("Invalid StreamElements")
                print(response2.text)
        else:
            self.update_log_window_2("\nError: Double check your channel name")
            self.update_status_bar("Invalid StreamElements")
            print(response.text)

        return False

    def get_min_donation(self):
        return float(self.ClientAmountLine.value())

    def get_token(self):
        TOKEN = ''.join(self.APIKeyLine.text().split())
        return TOKEN
        #tokenobj = TOKEN() # for debugging
        #return tokenobj.token # for debugging

    def get_current_TTmodel_dir(self):
        return self.TTmodel_dir[self.TTModelCombo.currentIndex()]

    def get_current_WGmodel_dir(self):
        return self.WGmodel_dir[self.WGModelCombo.currentIndex()]

    def get_current_TTmodel_fname(self):
        return self.TTModelCombo.currentText()

    def get_current_WGmodel_fname(self):
        return self.WGModelCombo.currentText()

    def get_interruptflag2(self):
        _mutex3.lock()
        val = _running3
        _mutex3.unlock()
        return val

    def set_reload_model_flag(self):
        self.reload_model_flag = True

    def set_cuda(self):
        self.use_cuda = self.GpuSwitch.isChecked()
        self.reload_model_flag = True

    def add_TTmodel_path(self):
        fpath = str(QFileDialog.getOpenFileName(self,
                                            'Select Tacotron2 model',
                                            filter='*.pt')[0])
        if not fpath: # If no folder selected
            return
        if fpath not in self.TTmodel_dir:
            head,tail = os.path.split(fpath) # Split into parent and child dir
            self.TTmodel_dir.append(fpath) # Save full path
            self.populate_modelcombo(tail, self.TTModelCombo)
            self.update_log_window("Added Tacotron 2 model: "+tail)
            if self.WGModelCombo.count() > 0:
                self.startup_update()

    def add_WGmodel_path(self):
        fpath = str(QFileDialog.getOpenFileName(self,
                                            'Select Waveglow model',
                                            filter='*.pt')[0])
        if not fpath: # If no folder selected
            return
        if fpath not in self.WGmodel_dir:
            head,tail = os.path.split(fpath) # Split into parent and child dir
            self.WGmodel_dir.append(fpath) # Save full path
            self.populate_modelcombo(tail, self.WGModelCombo)
            self.update_log_window("Added Waveglow model: "+tail)
            if self.TTModelCombo.count() > 0:
                self.startup_update()

    def populate_modelcombo(self, item, combobox):
        combobox.addItem(item)
        combobox.setCurrentIndex(combobox.count()-1)
        if not combobox.isEnabled():
            combobox.setEnabled(True)

    def update_log_window(self, line, mode="newline"):
        if mode == "newline" or not self.logs:
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        elif mode == "clear":
            self.logs = [line]
        log_text = '\n'.join(self.logs)
        self.log_window1.setText(log_text)

    def update_log_window_2(self, line, mode="newline"):
        if mode == "newline" or not self.logs2:
            self.logs2.append(line)
        elif mode == "append":
            self.logs2[-1] += line
        elif mode == "overwrite":
            self.logs2[-1] = line
        log_text = '\n'.join(self.logs2)
        self.log_window2.setPlainText(log_text)
        self.log_window2.verticalScrollBar().setValue(
            self.log_window2.verticalScrollBar().maximum())

    def update_status_bar(self, line):
        self.statusbar.setText(line)

class inferThread(QThread):
    timeElapsed = pyqtSignal(int)
    audioSignal = pyqtSignal(np.ndarray)
    iterSignal = pyqtSignal(tuple)
    interruptSignal = pyqtSignal()

    def __init__(self, text, use_cuda, model, waveglow,
                progress, elapsed, timestart, num_thread, parent=None):
        super(inferThread, self).__init__(parent)
        self.text = text
        self.use_cuda = use_cuda
        self.model = model
        self.waveglow = waveglow
        self.progress = progress
        self.elapsed = elapsed
        self.num_thread = num_thread
        self.timeoffset = time.time()-timestart
        self.timerThread = timerThread(self.timeoffset, parent = self)
        self.timerThread.timeElapsed.connect(self.timeElapsed.emit)
        self.start()

    def run(self):
        self.timerThread.start(time.time())
        if self.num_thread:
            torch.set_num_threads(self.num_thread)
            os.environ['OMP_NUM_THREADS'] = str(self.num_thread)
            os.environ['MKL_NUM_THREADS'] = str(self.num_thread)
        lines = preprocess_text(self.text)
        output  = []
        for count,line in enumerate(lines):
            _mutex1.lock()
            if _running1 == False:
                _mutex1.unlock()
                self.interruptSignal.emit()
                return
            else:
                _mutex1.unlock()
            self.iterSignal.emit((count,len(lines)))
            sequence = np.array(text_to_sequence(line, ['english_cleaners']))[None, :]
            device = torch.device('cuda' if self.use_cuda else 'cpu')
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).to(device).long()
            # Decode text input
            mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
            with torch.no_grad():
                audio = self.waveglow.infer(mel_outputs_postnet,
                                        sigma=0.666,
                                        progress_callback = self.progress,
                                        elapsed_callback = self.elapsed,
                                        get_interruptflag = self.get_interruptflag)
                if type(audio) != torch.Tensor:
                    # Catches when waveglow is interrupted and returns none
                    self.interruptSignal.emit()
                    return
                self.iterSignal.emit((count+1,len(lines)))
                wav = audio[0].data.cpu().numpy()
            output.append(wav)
        outwav = np.concatenate(output)
        self.audioSignal.emit(outwav)

    def get_interruptflag(self):
        _mutex1.lock()
        val = _running1
        _mutex1.unlock()
        return val


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = GUI(app)
    window.show()
    sys.exit(app.exec_())