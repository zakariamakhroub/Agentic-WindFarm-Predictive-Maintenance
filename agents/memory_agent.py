class MemoryAgent:

    def __init__(self):
        self.history = []

    def store(self, decision):

        self.history.append(decision)

    def get_history(self):

        return self.history