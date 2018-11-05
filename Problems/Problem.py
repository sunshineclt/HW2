class Problem:
    def evaluate(self):
        raise NotImplementedError

    def evaluate_on_all_datasets(self, method):
        raise NotImplementedError
