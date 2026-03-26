import matplotlib.pyplot as plt
import numpy as np

# 指令: 0=停, 1=前进, 2=左转, 3=右转
cmd_map = {
    0: (0.0, 0.0),
    1: (0.3, 0.0),
    2: (0.0, 0.5),
    3: (0.0, -0.5)
}

if __name__ == "__main__":
    actions = [1,1,1,2,1,3,1,0,0,1,2,3,0]
    v_list, w_list = [], []
    for a in actions:
        v, w = cmd_map[a]
        v_list.append(v)
        w_list.append(w)
    plt.subplot(2,1,1)
    plt.plot(v_list, label='v')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(w_list, label='w')
    plt.legend()
    plt.show()
