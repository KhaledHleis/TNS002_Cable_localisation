import sys
import pickle
import matplotlib.pyplot as plt

with open(sys.argv[1], "rb") as f:
    fig = pickle.load(f)

fig.show()
plt.show()
