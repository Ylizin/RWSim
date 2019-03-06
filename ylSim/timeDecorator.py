import time 
from functools import wraps

def time_counter(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.process_time()
        ret = func(*args,**kwargs)
        end = time.process_time()
        print('{} used time: {}'.format(func.__name__,end-start))
        return ret
    return wrapper
