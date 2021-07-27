import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fp = open('train.log', 'rt')
    assert fp
    lines = fp.read()
    lines = lines.split('\n')
    lines = list(filter(lambda x: len(x)==37, lines))
    loss_ = [float(lines[i][-8:]) for i in range(len(lines))]
    loss_ = np.array(loss_)
    plt.figure()
    plt.plot(np.arange(len(loss_)), loss_)
    plt.ylim(0, 255)
    plt.show()