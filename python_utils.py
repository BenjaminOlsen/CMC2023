import numpy as np
from timeit import default_timer as timer

try:
    import pyext
except:
    print "ERROR: This script must be loaded through the PD external!"


class RingBuffer:
    size = 0
    arr = np.array([])
    time_arr = np.array([])
    cur_idx = 0

    def __init__(self, size):
        self.size = size
        self.arr = np.zeros(size)
        self.time_arr = np.zeros(size)
    
    def append(self, val):
        #print("appending value", val, "to idx", self.cur_idx, "size", self.size, "t:", timer())
        self.arr[self.cur_idx] = val
        self.time_arr[self.cur_idx] = timer()
        self.cur_idx = (self.cur_idx + 1) % self.size
        #print("jk lol: t:", timer())

    def get_last_val(self, idx):
        return self.arr[(self.cur_idx - idx - 1) % self.size]

    def get_rms_energy(self):
        return np.sqrt(np.sum(self.arr*self.arr)/self.arr.size)
    
    def get_gradient(self):
        # calculates dy/dt using finite differences
        return np.gradient(self.arr, self.time_arr)

    def get_last_gradient(self, len=1):
        gradient = self.get_gradient()
        print("GRADIENT", gradient)
        return np.average(gradient.take(range(self.cur_idx-len, self.cur_idx), mode='wrap'))

    def get_last_energy_gradient(self, len=1):
        energy = self.arr**2
        gradient = np.gradient(energy, self.time_arr)
        return np.average(gradient.take(range(self.cur_idx-len, self.cur_idx), mode='wrap'))

    def get_energy_gradient(self):
        return np.gradient(self.get_rms_energy())

    def resize(self, new_size):
        #TODO: subtle detail, to trim at cur_idx-new_size:cur_idx-1
        if new_size == self.size:
            return
        if new_size < self.size:
            self.arr = self.arr[:new_size]
            self.time_arr = self.time_arr[:new_size]
        elif new_size > self.size:
            self.arr = np.append(self.arr, np.zeros(new_size-self.size))
            self.time_arr = np.append(self.time_arr, np.zeros(new_size-self.size))
            

        self.size = new_size
# object to turn accelerometer data from mobmuplat into groove events
class acc_to_groove(pyext._class):
    _inlets = 7
    _outlets = 5
    buf_size = 64
    pitch_buf = RingBuffer(buf_size)
    roll_buf =  RingBuffer(buf_size)
    yaw_buf =   RingBuffer(buf_size)
    threshold = 1
    total_energy_buf = RingBuffer(buf_size)
    energy_window = 24

    # in seconds:
    last_event_time = 0
    debounce_time = 0.1
    
    def resize(self, new_buf_size):
        print("acc_to_groove resizing buffers from", self.buf_size, "to", new_buf_size)
        self.buf_size = new_buf_size
        self.pitch_buf.resize(new_buf_size)
        self.roll_buf.resize(new_buf_size)
        self.yaw_buf.resize(new_buf_size)

    def set_energy_window(self, n):
        n = max(1, min(self.buf_size, n))
        print("setting energy window to", n, "samples")
        self.energy_window = n
        
    def update_total_energy(self):
        p = self.pitch_buf.get_rms_energy()
        r = self.roll_buf.get_rms_energy()
        y = self.yaw_buf.get_rms_energy()
        total_energy = p + r + y 
        
        print("acc_to_groove: total energy:", total_energy, "roll/yaw/pitch", r, y, p)
        self.total_energy_buf.append(total_energy)
    
    def get_last_total_energy(self, idx = 0):
        return self.total_energy_buf.get_last_val(idx)

    def get_total_energy_change(self, len = 1):
        v1 = self.get_last_total_energy(0)
        v2 = self.get_last_total_energy(len)

        print("total energy change:", v1-v2, "v1:", v1, "v2:", v2)
        return v1 - v2

    def get_energy_change(self):
        roll_energy_avg  = self.roll_buf.get_last_energy_gradient(self.energy_window)
        pitch_energy_avg = self.pitch_buf.get_last_energy_gradient(self.energy_window)
        yaw_energy_avg   = self.yaw_buf.get_last_energy_gradient(self.energy_window)

        total = roll_energy_avg + pitch_energy_avg + yaw_energy_avg
        return total

    def bang_1(self):
        print("banG")
        self.output_dE() 

    def output_dE(self):
        #self.update_total_energy()
        #TODO: make "derivative" length customizable
        dE = self.get_energy_change() 
        #print("dE: {:.2f}, threshold: {:.2f}, event?: {}".format(dE, self.threshold, np.abs(dE) > self.threshold))

        outval = 0
        if np.abs(dE) > self.threshold:
            outval = np.abs(dE)
            if timer() - self.last_event_time > self.debounce_time:
                self._outlet(4, outval)
                self._outlet(5, "bang")
                self.last_event_time = timer()

        #the value should come before the bang     
        #self._outlet(4, outval)


    def float_2(self, f):
        self.pitch_buf.append(f)
        self.output_dE()
        #print("got pitch", f)

    def float_3(self, f):
        self.roll_buf.append(f)
        self.output_dE() 
        #print("got roll", f)

    def float_4(self, f):
        self.yaw_buf.append(f)
        self.output_dE()
        #print("got yaw", f)


    def float_5(self, f):
        print("acc_to_groove setting threshold to", f)
        self.threshold = f

    def int_6(self, f):
        self.set_energy_window(f)

    def float_7(self, f):
        f = max(0, min(f, 2))
        self.debounce_time = f
        print("set debounce time to {}".format(f))

################################################################################3
################################################################################3

GROOVE_STEP_SIZE = 32
TICKS_PER_STEP = 96

def idx_to_v_ut(idx):
    # assumes groove of step width 192, with ut from -0.5 to 0.5 
    # centered
    ticks = TICKS_PER_STEP
    half = ticks/2
    v_idx = idx / ticks 
    ut = ((idx % ticks) - half) / (2.0 * half)
    return v_idx, ut

# input: index and value -> update class arrays of v's and ut's
# output on BANG one LIST of v1..v32; ut1...ut32
class groove_to_vars(pyext._class):
    _inlets = 4
    _outlets = 2
    
    v = np.zeros(GROOVE_STEP_SIZE)
    ut = np.zeros(GROOVE_STEP_SIZE)

    # output v, ut
    def bang_1(self):
        print("bang 1")
        self._outlet(1, tuple(self.v))
        self._outlet(2, tuple(self.ut))
    
    def bang_4(self):
        print("py clearing v,ut")
        self.v = np.zeros(GROOVE_STEP_SIZE)
        self.ut = np.zeros(GROOVE_STEP_SIZE)
        print("v: {}, ut:".format(self.v, self.ut))
        
    # whole groove dump
    def list_3(self, *a):
        if len(a) == GROOVE_STEP_SIZE * TICKS_PER_STEP:
            print("got groove_dump of size {}".format(len(a)))
        else:
            print("groove_dump bad size!")
            return

        idx = 0
        for groove_val in a:
            if groove_val != 0:
                self.set_v_ut(idx, groove_val)
            idx += 1

    def set_v_ut(self, tick_idx, val):
        v_idx, ut = idx_to_v_ut(tick_idx)

        if v_idx < 0 or v_idx > GROOVE_STEP_SIZE:
            print("groove got index out of range! {}".format(v_idx))
            return

        self.v[v_idx] = val
        self.ut[v_idx] = ut

        #print("set v{} to {}; ut{} to {}".format(v_idx+1, val, v_idx+1, ut))

    def groove_val_2(self, *a):
        if len(a) == 2:
            print("groove got (tick idx {:.3f}, val {:.3f})".format(a[0], a[1]))
        else:
            print("groove got bad argument")
            return

        tick_idx = a[0]
        val      = a[1]

        self.set_v_ut(tick_idx, val)

