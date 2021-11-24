import win32con
import win32api
import win32gui
import time
import random
import numpy as np
import torch
import psutil
import subprocess
import pyautogui


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
        time.sleep(.001)
        win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)
        #time.sleep(.001)

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
        self.state_memory[self.idx, 0] = state.cpu() #ToDO check if opti
        self.state_memory[self.idx, 1] = new_state.cpu()
        self.action_reward_memory[self.idx] = np.array([action, reward], dtype=float)
        self.idx = (self.idx + 1) % len(self)
        if self.curr_size < len(self):
            self.curr_size += 1

    def __len__(self):  # Maximum Capacity
        return len(self.state_memory)


Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (1, 20, 10)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((1, 20, 10), dtype=np.uint8), 0, 0.0, False)

class SegmentTree:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)  # Build structured array
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class PrioritizedMemory:
    def __init__(self, capacity, history_length=1, discount=0.95, multi_step=1, priority_weight=0.4, priority_exponent=0.5):
        self.device = device
        self.capacity = capacity
        self.history = history_length
        self.discount = discount
        self.n = multi_step
        self.priority_weight = priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
        self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def push(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns the transitions with blank states where appropriate
    def _get_transitions(self, idxs):
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1]) # True if future frame has timestep 0
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0
        transitions[blank_mask] = blank_trans
        return transitions

    # Returns a valid sample from each segment
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
            if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxs)
        # Create un-discretised states and nth next states
        all_states = transitions['state']
        states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
        next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255)
        # Discrete actions to be used as index
        actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        # Mask for non-terminal nth next states
        nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
        return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size, p_total)  # Get batch of valid samples
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # If future frame has timestep 0
        transitions[blank_mask] = blank_trans
        state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state
def relaunch_routine():
    pressAndHold('ctrl')
    time.sleep(0.1)
    press('p')
    time.sleep(0.1)
    release('ctrl')
    time.sleep(0.5)
    press('F1')
    time.sleep(0.5)
    pressAndHold('enter')
    time.sleep(0.1)
    release('enter')
    time.sleep(2)
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
        time.sleep(0.01)
        press('down_arrow')
        time.sleep(0.01)
        press('enter')
        time.sleep(0.1)
        pressAndHold('ctrl')
        press('1')
        press('2')
        press('3')
        time.sleep(0.1)
        release('ctrl')
        time.sleep(0.1)
        pyautogui.click(200, 150)
        time.sleep(0.1)
        press('F1')
        time.sleep(1)
        press('enter')
        time.sleep(3)
        pressAndHold('ctrl')
        time.sleep(0.1)
        press('p')
        time.sleep(0.1)
        release('ctrl')
        time.sleep(0.3)
        return p

    except Exception as e:
        print(e)
        p = None
        return p