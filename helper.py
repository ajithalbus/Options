import pickle

def load_data(name='bin.dat'):
    try:
        with open(name) as f:
            data = pickle.load(f)
    except:
        data = None
    return data

def save_data(data,name='bin.dat'):
    with open(name, "wb") as f:
        pickle.dump(data, f)