import random
import numpy as np

from baseline.sae.src.awa_demo import AWA


def kill_semantic_attributes(data, rate, new_value):
    """
    Randomly sets to new_value a specific rate of the semantic attributes

    @param data: 2D numpy array with semantic data
    @param rate: float number from 0 to 1 specifying the rate of values to be replaced
    @param new_value: float value that will replace the degraded ones
    @return: 2D numpy array with new data set
    """
    num_sem_attrs = data.shape[1]

    new_data = np.copy(data)
    for ex in range(new_data.shape[0]):
        mask = [False] * data.shape[1]
        for idx in random.sample(range(data.shape[1]), round(num_sem_attrs * rate)):
            mask[idx] = True

        new_data[ex, mask] = new_data[ex, mask] * 0 + new_value
    return new_data


if __name__ == '__main__':
    awa = AWA('../../../../Datasets/SAE/awa_demo_data.mat')
    awa.set_semantic_data(kill_semantic_attributes(awa.data['S_tr'], 0.9, 0))
    acc = awa.v2s_projection()
    print(acc)
