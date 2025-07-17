import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


class fuzzylabel():
        
        def __init__(self, name: str, label: str, range: np.ndarray, degree:float=0.0, weight:float=0.0):
            self.name = name
            self.label = label
            self.fuzzy_range = range
            self.degree = degree
            self.weight = weight

class fuzzyitem():

    def __init__(self, name: str, range: np.ndarray):
        self.name = name
        self.range = range
        self.labels = []
    
    def define_fuzzy_ranges(self, labels, fuzzy_ranges):
        # print(self.name)
        for i, label in enumerate(labels):
            # 使用三角和梯度
            if (i == 0) or (i == len(labels)-1):
                fuzzy_ranges[i].sort()
                assert len(fuzzy_ranges[i]) == 4
                # print(f'fuzz.trapmf(np.linspace(0, 1, 1000), {fuzzy_ranges[i]})')
                self.labels.append(
                    fuzzylabel(self.name, label, fuzz.trapmf(self.range, fuzzy_ranges[i]))
                    )
            else:
                fuzzy_ranges[i].sort()
                assert len(fuzzy_ranges[i]) == 3
                # print(f'fuzz.trimf(np.linspace(0, 1, 1000), {fuzzy_ranges[i]})')
                self.labels.append(
                     fuzzylabel(self.name, label, fuzz.trimf(self.range, fuzzy_ranges[i]))
                )
        return

    def show_membership_func(self):
        colors = ['g', 'b', 'r', 'c', 'y']
        plt.figure()
        plt.title(f'{self.name}')
        for i, label in enumerate(self.labels):
            plt.plot(self.range, label.fuzzy_range, colors[i], label=f'{label.label}')
        plt.legend()
        plt.show()

    def get_matching_label(self, x):
        result = {}
        for label in self.labels:
            result[label.label] = fuzz.interp_membership(self.range, label.fuzzy_range, x)
        match_label = max(result.items(), key=lambda x: x[1])[0]
        return match_label
    
    def get_matching_label_degree(self, x):
        result = {}
        for label in self.labels:
            result[label.label] = fuzz.interp_membership(self.range, label.fuzzy_range, x)
        match_label = max(result.items(), key=lambda x: x[1])[0]
        match_degree = result[match_label]
        return match_degree
    

class antecedent(fuzzyitem):

    def __init__(self, name: str, range: np.ndarray):
        super().__init__(name, range)

    def get_matching_degree(self, ante_label, x):
        degree = fuzz.interp_membership(self.range, ante_label.fuzzy_range, x)
        return degree
    

class consequent(fuzzyitem):

    def __init__(self, name: str, range: np.ndarray):
        super().__init__(name, range)

    def get_belief_degree(self, cons_label, x):
        degree = fuzz.interp_membership(self.range, cons_label.fuzzy_range, x)
        return degree
