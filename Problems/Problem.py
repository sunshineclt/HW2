class Problem:
    def evaluate(self, param):
        raise NotImplementedError

    def evaluate_on_all_datasets(self, method):
        raise NotImplementedError
