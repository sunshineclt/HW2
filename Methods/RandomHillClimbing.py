from Methods.Method import Method


class RandomHillClimbing(Method):
    def __init__(self, params, neighbor_op, problem, max_try_per_step=100, max_iter=1000, print_freq=1000, verbose=False):
        super().__init__()
        self.params = params
        self.hyper_params = {"max_try_per_step": max_try_per_step,
                             "max_iter": max_iter}
        self.neighbor_op = neighbor_op
        self.problem = problem
        self.previous_score = 0
        self.print_freq = print_freq
        self.verbose = verbose

    def step(self):
        time = 0
        while time < self.hyper_params["max_try_per_step"]:
            neighbor = self.neighbor_op(self.params)
            neighbor_score = self.problem.evaluate(neighbor)
            if self.previous_score < neighbor_score:
                self.params = neighbor
                if self.verbose:
                    print("Param Updated! Score %f -> %f" % (self.previous_score, neighbor_score))
                self.previous_score = neighbor_score
                break
            time += 1
        is_no_updating = time == self.hyper_params["max_try_per_step"]
        if is_no_updating and self.verbose:
            print("No param update! ")
        return time != self.hyper_params["max_try_per_step"]

    def find(self, stop_fun=None):
        iter_time = 0
        is_updating = True
        self.previous_score = self.problem.evaluate(self.params)
        max_iter = self.hyper_params["max_iter"]
        result_record = []
        while max_iter == -1 or iter_time < max_iter:
            iter_time += 1
            if self.verbose:
                print("iter: %d" % iter_time, end=" ")
            is_updating = self.step()
            if iter_time % self.print_freq == 0:
                result = self.problem.evaluate(self.params)
                result_record.append(result)
                if self.verbose:
                    print("Iter time: %d, result: %f" % (iter_time, result))
                if stop_fun is not None:
                    if stop_fun(result):
                        if self.verbose:
                            print("Optimal Reached! ")
                        break
        return result_record
