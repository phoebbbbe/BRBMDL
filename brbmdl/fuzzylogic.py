import numpy as np
import skfuzzy as fuzz

def get_fuzzy_values_from_treerules(tree_rules):
        lines = tree_rules.split('\n')
        rules = []
        stack = []
        values = {}

        for line in lines:
            depth = line.count('|')
            content = line.split('|')[-1].replace('---', '').strip()

            if 'class:' in content:
                decision = content.split(':')[-1].strip()
                rule = ' and '.join(stack) + f' then {decision}'
                rules.append(rule)
            elif 'truncated branch' in content:
                continue
            elif content == '':
                continue
            else:
                operators = content.split(' ')
                feature = operators[0]
                value = operators[-1]
                if f'{feature}' not in values:
                    values[f'{feature}'] = set()
                    values[f'{feature}'].add(value)
                else:
                    values[f'{feature}'].add(value)
                if len(stack) >= depth:
                    stack = stack[:depth-1]
                stack.append(content)
        return values


def get_average_group(arr, n):
    arr = [float(x) for x in arr]
    arr.sort()
    while len(arr) < n:
        if len(arr) < 2:
            arr.append(0)
        max_gap = 0
        insert_index = 0
        for i in range(len(arr)-1):
            gap = arr[i+1] - arr[i]
            if gap > max_gap:
                max_gap = gap
                insert_index = i

        new_val = (arr[insert_index] + arr[insert_index+1]) / 2
        arr.append(new_val)
    
    groups = np.array_split(arr, n)
    averaged_groups = [round(np.mean(group), 2) for group in groups]
    averaged_groups.sort()
    return averaged_groups


def set_fuzzy_range(arr, min_val, max_val):
    result = []
    for i in range(len(arr)):
        if i < len(arr)-2:
            result.append([arr[i], arr[i+1], arr[i+2]])
    # result.insert(0, [min_val, arr[0], arr[1]])
    result.insert(0, [min_val, min_val, arr[0], arr[1]])
    # result.append([arr[-2], arr[-1], max_val])
    result.append([arr[-2], arr[-1], max_val, max_val])
    return result


def get_fuzzy_range(ifthen_values, antecedents, group_num):
    fuzzy_range = {}
    for key, value in ifthen_values.items():
        for ante in antecedents:
            if key == ante.name:
                min_val = min(ante.range)
                max_val = max(ante.range)
                break
            else:
                min_val = -1.0
                max_val = 1.0
        value = list(value)
        value = get_average_group(value, group_num)
        value = set_fuzzy_range(value, min_val, max_val)

        key = str(key).lower()
        fuzzy_range[f'{key}'] = value
        # print (key, value, '\n')
    return fuzzy_range


