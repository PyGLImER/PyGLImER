import os

def chdir(func, rpath=".."):
    def wrapper_chdir(*args, **kwargs):
        
        # Get dir
        cw = os.path.abspath(os.getcwd())
        
        # Change dir
        os.chdir(os.path.join(cw, rpath))
        
        # Exectute function
        ret = func(*args, **kwargs)
        
        # Change back
        os.chdir(cw)
        
        return ret
        
    return wrapper_chdir