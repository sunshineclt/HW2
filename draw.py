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

# fo_rhc = load("fo_rhc")
# fo_sa = load("fo_sa")
# fo_ga = load("fo_ga_pointwise")
# plt.plot(range(100, 4001, 100), np.mean(fo_rhc, axis=0), label="RHC")
# plt.plot(range(100, 4001, 100), np.mean(fo_sa, axis=0), label="SA")
# plt.plot(range(100, 4001, 100), np.mean(fo_ga, axis=0), label="GA")
# plt.legend()
# plt.xlabel("iteration")
# plt.ylabel("fitness")
# plt.savefig("fig/fo_comparison")
# plt.show()

nn_rhc = load("nn_rhc")
nn_sa = load("nn_sa")
nn_ga = load("nn_ga")
plt.plot(range(10, 5001, 10), np.mean(nn_rhc, axis=0), label="RHC")
plt.plot(range(10, 50001, 10), np.mean(nn_sa, axis=0), label="SA")
plt.plot(range(10, 20001, 10), np.mean(nn_ga, axis=0), label="GA")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("fitness")
plt.savefig("fig/nn_comparison")
plt.show()
