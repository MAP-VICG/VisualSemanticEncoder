import random
import numpy as np


class SemanticDegradation:
    @staticmethod
    def kill_semantic_attributes(data, rate, new_value=None):
        max_value = np.max(data)
        min_value = np.min(data)
        new_data = np.copy(data)

        if rate == 0:
            return new_data

        if new_value:
            for ex in range(new_data.shape[0]):
                for idx in random.sample(range(data.shape[1]), round(data.shape[1] * rate)):
                    new_data[ex, idx] = new_value
        else:
            for ex in range(new_data.shape[0]):
                for idx in random.sample(range(data.shape[1]), round(data.shape[1] * rate)):
                    new_data[ex, idx] = random.uniform(min_value, max_value)

        return new_data
