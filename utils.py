import win32con
import win32api
import win32gui
import time
import random
import numpy as np
import torch
import psutil
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Credits: https://gist.github.com/chriskiehl/2906125
VK_CODE = {'backspace':0x08,
           'tab':0x09,
           'clear':0x0C,
           'enter':0x0D,
           'shift':0x10,
           'ctrl':0x11,
           'alt':0x12,
           'pause':0x13,
           'caps_lock':0x14,
           'esc':0x1B,
           'spacebar':0x20,
           'page_up':0x21,
           'page_down':0x22,
           'end':0x23,
           'home':0x24,
           'left_arrow':0x25,
           'up_arrow':0x26,
           'right_arrow':0x27,
           'down_arrow':0x28,
           'select':0x29,
           'print':0x2A,
           'execute':0x2B,
           'print_screen':0x2C,
           'ins':0x2D,
           'del':0x2E,
           'help':0x2F,
           '0':0x30,
           '1':0x31,
           '2':0x32,
           '3':0x33,
           '4':0x34,
           '5':0x35,
           '6':0x36,
           '7':0x37,
           '8':0x38,
           '9':0x39,
           'a':0x41,
           'b':0x42,
           'c':0x43,
           'd':0x44,
           'e':0x45,
           'f':0x46,
           'g':0x47,
           'h':0x48,
           'i':0x49,
           'j':0x4A,
           'k':0x4B,
           'l':0x4C,
           'm':0x4D,
           'n':0x4E,
           'o':0x4F,
           'p':0x50,
           'q':0x51,
           'r':0x52,
           's':0x53,
           't':0x54,
           'u':0x55,
           'v':0x56,
           'w':0x57,
           'x':0x58,
           'y':0x59,
           'z':0x5A,
           'numpad_0':0x60,
           'numpad_1':0x61,
           'numpad_2':0x62,
           'numpad_3':0x63,
           'numpad_4':0x64,
           'numpad_5':0x65,
           'numpad_6':0x66,
           'numpad_7':0x67,
           'numpad_8':0x68,
           'numpad_9':0x69,
           'multiply_key':0x6A,
           'add_key':0x6B,
           'separator_key':0x6C,
           'subtract_key':0x6D,
           'decimal_key':0x6E,
           'divide_key':0x6F,
           'F1':0x70,
           'F2':0x71,
           'F3':0x72,
           'F4':0x73,
           'F5':0x74,
           'F6':0x75,
           'F7':0x76,
           'F8':0x77,
           'F9':0x78,
           'F10':0x79,
           'F11':0x7A,
           'F12':0x7B,
           'F13':0x7C,
           'F14':0x7D,
           'F15':0x7E,
           'F16':0x7F,
           'F17':0x80,
           'F18':0x81,
           'F19':0x82,
           'F20':0x83,
           'F21':0x84,
           'F22':0x85,
           'F23':0x86,
           'F24':0x87,
           'num_lock':0x90,
           'scroll_lock':0x91,
           'left_shift':0xA0,
           'right_shift ':0xA1,
           'left_control':0xA2,
           'right_control':0xA3,
           'left_menu':0xA4,
           'right_menu':0xA5,
           'browser_back':0xA6,
           'browser_forward':0xA7,
           'browser_refresh':0xA8,
           'browser_stop':0xA9,
           'browser_search':0xAA,
           'browser_favorites':0xAB,
           'browser_start_and_home':0xAC,
           'volume_mute':0xAD,
           'volume_Down':0xAE,
           'volume_up':0xAF,
           'next_track':0xB0,
           'previous_track':0xB1,
           'stop_media':0xB2,
           'play/pause_media':0xB3,
           'start_mail':0xB4,
           'select_media':0xB5,
           'start_application_1':0xB6,
           'start_application_2':0xB7,
           'attn_key':0xF6,
           'crsel_key':0xF7,
           'exsel_key':0xF8,
           'play_key':0xFA,
           'zoom_key':0xFB,
           'clear_key':0xFE,
           '+':0xBB,
           ',':0xBC,
           '-':0xBD,
           '.':0xBE,
           '/':0xBF,
           '`':0xC0,
           ';':0xBA,
           '[':0xDB,
           '\\':0xDC,
           ']':0xDD,
           "'":0xDE,
           '`':0xC0}

def press(*args):
    '''
    one press, one release.
    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.005)
        win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)


def pressAndHold(*args):
    '''
    press and hold. Do NOT release.
    accepts as many arguments as you want.
    e.g. pressAndHold('left_arrow', 'a','b').
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.001)


def release(*args):
    '''
    release depressed keys
    accepts as many arguments as you want.
    e.g. release('left_arrow', 'a','b').
    '''
    for i in args:
           win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)


class ReplayMemory:  # ToDo: implement save functions with pickle library
    def __init__(self, size):
        self.state_memory = np.empty((size, 2, 20, 10))
        self.action_reward_memory = np.empty((size, 2))
        self.curr_size = 0
        self.idx = 0

    def sample(self, batch_size=128):
        batch_idx = random.sample(range(self.curr_size), k=batch_size)
        return torch.from_numpy(self.state_memory[batch_idx, 0]).to(device).to(torch.float), torch.from_numpy(self.state_memory[batch_idx, 1]).to(device).to(torch.float), torch.from_numpy(self.action_reward_memory[batch_idx, 0]).to(device).to(torch.int), torch.from_numpy(self.action_reward_memory[batch_idx, 1]).to(device).to(torch.float)

    def push(self, state, new_state, action, reward):
        self.state_memory[self.idx, 0] = state.cpu()
        self.state_memory[self.idx, 1] = new_state.cpu()
        self.action_reward_memory[self.idx] = np.array([action, reward], dtype=float)
        self.idx = (self.idx + 1) % len(self)
        if self.curr_size < len(self):
            self.curr_size += 1

    def __len__(self):  # Maximum Capacity
        return len(self.state_memory)


def relaunch_routine():
    pressAndHold('ctrl')
    time.sleep(0.1)
    press('p')
    time.sleep(0.1)
    release('ctrl')
    time.sleep(0.2)
    press('F1')
    time.sleep(1)
    press('enter')
    time.sleep(2.5)
    pressAndHold('ctrl')
    time.sleep(0.1)
    press('p')
    time.sleep(0.1)
    release('ctrl')


def launch_environment_routine():
    try:
        for proc in psutil.process_iter():
            if proc.name() == 'VisualBoyAdvance.exe':
                proc.kill()
        process = subprocess.Popen('E:/Programmation/Python/tetrist_rl/emulator/VisualBoyAdvance.exe')
        p = psutil.Process(process.pid)
        time.sleep(1)
        window = win32gui.FindWindow(None, "VisualBoyAdvance")
        width, height = win32gui.GetWindowRect(window)[2] - win32gui.GetWindowRect(window)[0], \
                        win32gui.GetWindowRect(window)[3] - win32gui.GetWindowRect(window)[1]
        win32gui.MoveWindow(window, -7, 0, width, height, False)
        pressAndHold('ctrl')
        time.sleep(0.1)
        press('o')
        time.sleep(0.1)
        release('ctrl')
        time.sleep(0.1)
        press('t')
        press('down_arrow')
        press('enter')
        time.sleep(0.1)
        pressAndHold('ctrl')
        time.sleep(0.2)
        press('1')
        time.sleep(0.1)
        press('2')
        time.sleep(0.1)
        press('3')
        time.sleep(0.2)
        release('ctrl')
        time.sleep(0.1)
        press('F1')
        time.sleep(1)
        press('enter')
        time.sleep(2)
        pressAndHold('ctrl')
        time.sleep(0.1)
        press('p')
        time.sleep(0.1)
        release('ctrl')
        time.sleep(0.1)
        return p


    except Exception as e:
        print(e)
        p = None
