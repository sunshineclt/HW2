import pickle
import matplotlib.pyplot as plt
import numpy as np


def load(filename):
    with open("result/" + filename + ".pkl", "rb") as f:
        return pickle.load(f)


# fp_rhc = load("fp_rhc_multi")
# fp_sa = load("fp_sa")
# fp_ga = load("fp_ga_crosspoint")
# plt.plot(range(100, 20001, 100), np.mean(fp_rhc, axis=0), label="RHC")
# plt.plot(range(100, 20001, 100), np.mean(fp_sa, axis=0), label="SA")
# plt.plot(range(100, 20001, 100), np.mean(fp_ga, axis=0), label="GA")
# plt.legend()
# plt.xlabel("iteration")
# plt.ylabel("fitness")
# plt.savefig("fig/fp_comparison")
# plt.show()

fp_rhc = load("fo_rhc")
fp_sa = load("fo_sa")
fp_ga = load("fo_ga_pointwise")
plt.plot(range(100, 4001, 100), np.mean(fp_rhc, axis=0), label="RHC")
plt.plot(range(100, 4001, 100), np.mean(fp_sa, axis=0), label="SA")
plt.plot(range(100, 4001, 100), np.mean(fp_ga, axis=0), label="GA")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("fitness")
plt.savefig("fig/fo_comparison")
plt.show()
