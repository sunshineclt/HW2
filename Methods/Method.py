class Method:
    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError

    def find(self, stop_fun=None):
        raise NotImplementedError
