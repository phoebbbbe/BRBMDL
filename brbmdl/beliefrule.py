import numpy as np
from .antecedent_consequent import antecedent, consequent, fuzzylabel

class beliefrule():

    def __init__(
            self, 
            antecedent: list[fuzzylabel], 
            consequent: list[fuzzylabel], 
            rule_weight: float=0.1, 
            and_func=np.fmin, 
            or_func=np.fmax):
        self.antecedent = antecedent
        self.consequent = consequent
        self.rule_weight = rule_weight
        self.and_func = and_func
        self.or_func = or_func
    
    # def __repr__(self):
    #     antes_str = []
    #     for ante in self.antecedent:
    #         ante_str = f"('{ante.name}'is'{ante.label}':{ante.degree:.2f} with {ante.weight:.2f})"
    #         antes_str.append(ante_str)
    #     antes_str = ' and '.join(antes_str)
    #     conss_str = []

    #     for cons in self.consequent:
    #         cons_str = f"('{cons.label}':{cons.degree:.2f})"
    #         conss_str.append(cons_str)
    #     conss_str = ' and '.join(conss_str)
    #     return f'{self.rule_weight:.2f} If {antes_str} Then {conss_str}'

    def __repr__(self):
        antes_str = []
        for ante in self.antecedent:
            ante_str = f"({ante.name} is {ante.label})"
            antes_str.append(ante_str)
        antes_str = ' ∧ '.join(antes_str)
        conss_str = []

        for cons in self.consequent:
            cons_str = f"({cons.label},{cons.degree:.3f})"
            conss_str.append(cons_str)
        conss_str = ' , '.join(conss_str)
        return f'{self.rule_weight:.3f} If {antes_str} Then {conss_str}'
    

    def get_new_rule(self, n):
            self.antecedent = self.antecedent[:n]



    




