import os

def get_proj_dir():
    # start where we currently are
    fpath = os.getcwd()
    while fpath:
        # check if fpath is directory with .root.txt in it
        if os.path.exists(os.path.join(fpath, ".root.txt")):
            return os.path.join(fpath, '')  # adds slash if missing
        fpath = os.path.dirname(fpath.rstrip("/"))  # go up to parent dir
    
    raise FileNotFoundError("Could not determine project directory path from Python script.")
