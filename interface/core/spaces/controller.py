import importlib


class Controller:

    def __init__(self, parent, geometry):
        self.geometry = geometry
        self.parent = parent
        # Initialize cache for spaces
        self.spaces = {}
        self.activeSpace = None

    def loadSpace(self, name, forceReload=False):

        if self.activeSpace is not None:
            self.activeSpace.hide()

        if name in self.spaces and not forceReload:
            print('Loading', name, 'from cache')
            self.activeSpace = self.spaces[name]
        else:
            print('Loading', name, 'from module')
            mod = importlib.import_module(name)
            self.spaces[name] = mod.Space(self)
            self.activeSpace = self.spaces[name]

        self.activeSpace.show()
