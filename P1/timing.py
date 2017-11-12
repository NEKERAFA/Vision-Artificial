import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('> Hecho en {:0.2f} ms'.format((time2-time1)*1000.0))
        return ret
    return wrap
