import pandas as pd
from .antecedent_consequent import fuzzylabel, antecedent, consequent
from .beliefrule import beliefrule
from .fuzzylogic import *
from .training import *
from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time
import pickle
# import dtreeviz


class brbm():
    '''belief rule-based model'''
    def __init__(self, antes: list, cons: str):
        r1 = np.linspace(0, 1, 1000)
        r2 = np.linspace(-1, 1, 1000)
        self.antecedents = []
        for ante_name in antes:
            self.antecedents.append(antecedent(ante_name, r1))
        self.consequent = consequent(cons, r2)
        self.ante_labels = []
        self.cons_labels = []
        self.database = pd.DataFrame()
        self.rulebase = {}
        self.labels = []
        self.antes_labels = []
        self.labels_features = []
        self.dtree = None


    def add_antecedent(self, ante: antecedent):
        self.antecedents.append(ante)
        return


    def add_consequent(self, cons: consequent):
        self.consequent = cons
        return


    def set_ante_labels(self, labels):
        self.ante_labels = labels
        for ante in self.antecedents:
            self.labels.append(f'{ante.name}_label')
            self.antes_labels.append(f'{ante.name}_label')
            for ante_label in ante.labels:
                self.labels_features.append(f'{ante.name}_{ante_label.label}')
        return


    def set_cons_labels(self, labels):
        self.cons_labels = labels
        cons = self.consequent
        self.labels.append(f'{cons.name}_label')
        for cons_label in cons.labels:
            self.labels_features.append(cons_label.label)
        return



    # def set_all_ante_fuzzy_ranges(self, X_train, y_train_label, features):
    def set_all_ante_fuzzy_ranges(self, dtree_name, features):

        ante_fuzzy_range = {}
        # le = LabelEncoder()
        # y_train_code = le.fit_transform(y_train_label)
        # labels = list(le.classes_)

        with open(f'{dtree_name}', 'rb') as f:
            self.dtree = pickle.load(f)

        tree_rules = export_treerules(self.dtree, features)
        ifthen_values = get_fuzzy_values_from_treerules(tree_rules)
        ante_fuzzy_range = get_fuzzy_range(ifthen_values, self.antecedents, len(self.ante_labels))

        # while True :
        #     dt_model = DecisionTreeClassifier(criterion='gini', max_depth=7, max_features=None)
        #     dt_model.fit(X_train, y_train_label)
        #     # dt_model.fit(X_train, y_train_code)
        #     tree_rules = export_treerules(dt_model, features)
        #     ifthen_values = get_fuzzy_values_from_treerules(tree_rules)
        #     ante_fuzzy_range = get_fuzzy_range(ifthen_values, self.antecedents, len(self.ante_labels))

        #     if len(ante_fuzzy_range) == len(features):
        #         # viz_model = dtreeviz.model(dt_model, X_train, y_train_code,
        #         #                 feature_names=features,
        #         #                 target_name='Stock Price',
        #         #                 class_names=labels
        #         #                 )

        #         # v = viz_model.view()
        #         # v.show()
        #         # v.save('./data/dtree.svg')
        #         self.dtree = dt_model
        #         break
        #     else:
        #         print("not match")

        for ante in self.antecedents:
            ante.define_fuzzy_ranges(self.ante_labels, ante_fuzzy_range[f'{ante.name}'])
            # ante.show_membership_func()
        return


    def set_cons_fuzzy_range(self, fuzzy_range: list):
        self.consequent.define_fuzzy_ranges(self.cons_labels, fuzzy_range)
        # self.consequent.show_membership_func()
        return
    

    def add_ante_label_cols(self, dataset: pd.DataFrame, type: int):
        if type > 0:
            for ante in self.antecedents:
                dataset[f'{ante.name}_label'] = dataset[ante.name].apply(lambda x: ante.get_matching_label(x))
                if type > 1:
                    for ante_label in ante.labels:
                        dataset[f'{ante.name}_{ante_label.label}'] = dataset[ante.name].apply(
                            lambda x: ante.get_matching_degree(ante_label, x))
        return dataset
    

    def add_ante_degree_cols(self, dataset: pd.DataFrame):
        input_degrees = []
        for ante in self.antecedents:
            dataset[f'{ante.name}_degree'] = dataset[ante.name].apply(lambda x: ante.get_matching_label_degree(x))
            input_degrees.append(f'{ante.name}_degree')
        return dataset, input_degrees

    
    def add_cons_label_cols(self, dataset: pd.DataFrame, type: int):
        cons = self.consequent
        if type > 0:
            dataset[f'{cons.name}_label'] = dataset[cons.name].apply(lambda x: cons.get_matching_label(x))
            if type > 1:
                for cons_label in cons.labels:
                    dataset[f'{cons_label.label}'] = dataset[cons.name].apply(
                        lambda x: cons.get_belief_degree(cons_label, x))
        return dataset


    def get_database(self):
        print(self.database)
        return self.database


    def get_rulebase(self):
        print(self.rulebase)
        return self.rulebase
    

    def inital_database(self, X_train, y_train):
        self.database = pd.concat([X_train, y_train], axis=1)
        self.add_ante_label_cols(self.database, type=2)
        self.add_cons_label_cols(self.database, type=2)
        self.database.reset_index(drop=True, inplace=True)
        return
    

    def inital_rulebase(self):
        fuzzylabelbase = []
        max_rule_weight = 1

        for i in range(len(self.database)):
            rule_antecedents, rule_consequents = [], []

            for ante in self.antecedents:
                degrees = {
                    label.label: self.database[f'{ante.name}_{label.label}'][i] 
                    for label in ante.labels
                    }
                max_key, max_value = max(degrees.items(), key=lambda x: x[1])
                rule_ante = fuzzylabel(
                    ante.name, max_key, ante.range, 
                    degree=max_value, weight=np.random.rand()
                    )
                rule_antecedents.append(rule_ante)

            for cons_label in self.consequent.labels:
                degree = self.database[cons_label.label][i]
                rule_cons = fuzzylabel(
                    self.consequent.name, cons_label.label, self.consequent.range, 
                    degree=degree, weight=1
                    )
                rule_consequents.append(rule_cons)

            fuzzylabel_i = self.database[self.labels].iloc[i].values.tolist()
            if fuzzylabel_i not in fuzzylabelbase:
                fuzzylabelbase.append(fuzzylabel_i)
                rule = beliefrule(rule_antecedents, rule_consequents, rule_weight=0.1)
                index = len(fuzzylabelbase)-1
                self.rulebase[index] = rule
            else:
                index = fuzzylabelbase.index(fuzzylabel_i)
                self.rulebase[index].rule_weight += 0.1
                if self.rulebase[index].rule_weight > max_rule_weight:
                    max_rule_weight = self.rulebase[index].rule_weight

        ''' 標準化規則權重 '''
        for i in range(len(self.rulebase)):
            self.rulebase[i].rule_weight = self.rulebase[i].rule_weight / max_rule_weight
            
        return
    

    def preprocess_database(self):
        self.database.drop_duplicates(subset=self.labels, keep='first', inplace=True)
        self.database.reset_index(drop=True, inplace=True)
        for ante in self.antecedents:
            self.database[f'{ante.name}_degree'] = self.database[ante.name].apply(lambda x: ante.get_matching_label_degree(x))
        return


    def get_match_rule_index(self, row, input_labels):
        rule_df = self.database[input_labels]
        row_dict = dict(map(lambda item: (item[0], [item[1]]), row.items()))
        match_rule_index = rule_df[rule_df.isin(row_dict).all(axis=1)].index.to_list()
        return match_rule_index

    
    def optimize(self, X_train, y_train, features, target):
        train_df = pd.concat([X_train, y_train], axis=1)
        train_df = self.add_ante_label_cols(train_df, 1)
        train_df, input_degrees = self.add_ante_degree_cols(train_df)
        train_df = self.add_cons_label_cols(train_df, 1)
        new_rule = {}

        for i in tqdm(range(len(train_df)), desc='Optimize'):

            '''找到符合的規則'''
            match_rule_index, matched_columns = [], []
            for n in range(len(self.labels), -1, -1):
                train_item_i = train_df[self.labels[:n]].iloc[i].to_dict()
                match_rule_index = self.get_match_rule_index(train_item_i, self.labels[:n])
                if len(match_rule_index) != 0:
                    matched_columns = self.antes_labels[:n]
                    break
            
            alpha_value = train_df[input_degrees].iloc[i]            
            
            '''取得符合的規則'''
            train_row_X = np.array([train_df[features].iloc[i]])
            train_row_y = np.array([[train_df[target].iloc[i]]])
            for k in match_rule_index:

                '''反向傳播-優化屬性權重'''
                att_weights = []
                for ante in self.rulebase[k].antecedent:
                    att_weights.append(ante.weight)
                att_weights = np.array([att_weights]).T

                # print('train_row_X:', train_row_X)
                # print('att_weights:', att_weights)

                new_att_weights = backpropagation(train_row_X, train_row_y, att_weights)
                # print('new_att_weights:', new_att_weights)
                for weight, ante in zip(new_att_weights.flat, self.rulebase[k].antecedent):
                    ante.weight = weight

                '''反向傳播-優化信念度'''
                # beta = []
                # for cons_k in self.rulebase[k].consequent:
                #     beta.append(cons_k.degree)
                # result = np.array([[0.1, -0.1]])
                # beliefdegree = np.array([beta]).T
                # # print('result:', result)
                # # print('beliefdegree:', beliefdegree)
                # new_beliefdegree = backpropagation(result, train_row_y, beliefdegree)
                # # print('new_beliefdegree:', new_beliefdegree)
                # for new_degree, cons_k in zip(new_beliefdegree.flat, self.rulebase[k].consequent):
                #         # print(f'rule {k} {cons_k.label} : {new_degree}')
                #         cons_k.degree = new_degree

                '''更新信念度'''
                beta_k = []
                for cons_k in self.rulebase[k].consequent:
                    beta_k.append(cons_k.degree)
                if len(matched_columns) != 0:
                    all_tau = 0
                    match_alpha = 0
                    for label in self.labels:
                        if label in matched_columns:
                            all_tau += 1
                            temp = label.split('_')
                            temp = temp[0] + '_degree'
                            match_alpha += alpha_value[temp]

                    beta_k = np.array(beta_k)
                    if np.any(beta_k == 0.0):
                        beta_k += 1e-3
                    beta_new = beta_k * (match_alpha / all_tau)
                    beta_new = beta_new / beta_new.sum()
                    for new_degree, cons_k in zip(beta_new.flat, self.rulebase[k].consequent):
                        cons_k.degree = new_degree

            time.sleep(0.05)
        return
    

    def get_parameters(self):
        theta, delta, beta = [], [], []

        for rule_k in self.rulebase.values():
            delta_k = []
            beta_k = []

            theta_k = rule_k.rule_weight
            theta.append(theta_k)

            for ante_k in rule_k.antecedent:
                delta_k.append(ante_k.weight)
            delta.append(delta_k)
            
            for cons_k in rule_k.consequent:
                beta_k.append(cons_k.degree)
            beta.append(beta_k)
        return theta, delta, beta


    def update_parameter(self, alpha_value: pd.Series, match_rule_index: list, matched_columns: list,):
        alpha_ki = alpha_value.to_list()
        rules = []
        theta, delta, beta = self.get_parameters()
        betas, omegas = [], []
        alpha_k_list = {}
        alpha_sum = 0

        for k in match_rule_index:
            '''式 3.12'''
            alpha_ki = np.array(alpha_ki)
            delta_k = np.array(delta[k]/np.sum(delta))
            terms = np.power(alpha_ki, delta_k)
            pro_a_k = np.prod(terms)
            alpha_k_list[k] = pro_a_k
            alpha_sum += pro_a_k

        for k in match_rule_index:
            '''式 3.14'''
            pro_a_k = alpha_k_list[k]
            theta_k = theta[k]
            omega_k = (theta_k * pro_a_k) / alpha_sum
            omegas.append(omega_k)

            '''式 3.13 更新權重（或說重新分配權重）'''
            if len(matched_columns) != 0:
                all_tau = 0
                match_alpha = 0
                for label in self.labels:
                    if label in matched_columns:
                        all_tau += 1
                        temp = label.split('_')
                        temp = temp[0] + '_degree'
                        match_alpha += alpha_value[temp]

                beta_k = np.array(beta[k])
                if np.any(beta_k == 0.0):
                    beta_k += 1e-3
                beta_new = beta_k * (match_alpha / all_tau)
                beta_new = beta_new / beta_new.sum()
                for new_degree, cons_k in zip(beta_new.flat, self.rulebase[k].consequent):
                        cons_k.degree = new_degree
            
            betas.append(beta_new)
            rules.append(self.rulebase[k])
            # print(self.rulebase[k])
        return rules, betas, omegas


    def predict(self, X_test: pd.DataFrame):
        y_pred = []
        input_degrees = []
        test_df = X_test.copy()
        test_df = self.add_ante_label_cols(test_df, 1)
        test_df, input_degrees = self.add_ante_degree_cols(test_df)
        
        for i in range(len(test_df)):

            '''取得觸發的規則'''
            match_rule_index, matched_columns = [], []
            for n in range(len(self.antes_labels), -1, -1):
                test_item_i = test_df[self.antes_labels[:n]].iloc[i].to_dict()
                match_rule_index = self.get_match_rule_index(test_item_i, self.antes_labels[:n])
                if len(match_rule_index) != 0:
                    matched_columns = self.antes_labels[:n]
                    break

            '''計算觸發權重和更新信念度'''
            alpha_value = test_df[input_degrees].iloc[i]
            rules, betas, omegas = self.update_parameter(alpha_value, match_rule_index, matched_columns)
            # print('betas:', betas)
            # print('omegas:', omegas)

            '''式 3.15 聚合規則'''
            left_parts = []
            right_part = 0
            base_right_part = 0
            for l in range(len(self.cons_labels)):
                sum_beta_kj = 0
                for k in range(len(betas)):
                    sum_beta_kj = sum(betas[k])

                left_values = []
                right_values = []
                base_right_values = []
                for k in range(len(betas)):
                    beta_kl = betas[k][l]
                    # print('beta_kl:', beta_kl)
                    omega_k = omegas[k]
                    # print('omega_k:', omega_k)
                    left_values.append(omega_k * beta_kl + 1 - omega_k * sum_beta_kj)
                    right_values.append(1 - omega_k * sum_beta_kj)
                    base_right_values.append(1 - omega_k)
                left_part = np.prod(left_values)
                # print('left_part:', left_part)
                left_parts.append(left_part)
            # print('left_parts:', left_parts)
            right_part = np.prod(right_values)
            # print('right_part:', right_part)
            base_right_part = np.prod(base_right_values)
            # print('base_right_part:', base_right_part)

            mu_left = 0
            for l in range(len(self.cons_labels)):
                mu_left += left_parts[l]
            mu_base = (mu_left - (len(self.consequent.labels)-1) * right_part)
            # print('mu_base:', mu_base)
            mu = 1 / mu_base
            # print('mu:', mu)

            aggragate_belief_degree = [
                (mu * (left_part - right_part)) /(1 - mu * base_right_part)
                for left_part in left_parts
            ]
            # print('aggragate_belief_degree:', aggragate_belief_degree)

            y_index = aggragate_belief_degree.index(max(aggragate_belief_degree))
            brb_prediction = self.cons_labels[y_index]

            y_pred.append(brb_prediction)
            # print('pred:', brb_prediction)

        return y_pred