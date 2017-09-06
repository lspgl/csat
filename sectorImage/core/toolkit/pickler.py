import pickle as pickle


class Pickler:

    def __init__(self):
        pass

    def load(self, fn):
        print('Loading', fn)
        with open(fn, 'rb') as inp:
            obj = pickle.load(inp)
        return obj

    def save(self, obj, fn):
        with open(fn, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        return
