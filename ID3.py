import pandas as pd
import math
import pprint

def entropy(target_col):
    elements, counts = pd.Series(target_col).value_counts().index, pd.Series(target_col).value_counts()
    entropy = 0
    for i in range(len(elements)):
        probability = counts[i] / sum(counts)
        entropy -= probability * math.log2(probability)
    return entropy

def info_gain(data, split_attribute_name, target_name="PlayTennis"):
    total_entropy = entropy(data[target_name])
    vals, counts = data[split_attribute_name].value_counts().index, data[split_attribute_name].value_counts()
    
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute_name] == vals[i]]
        weight = counts[i] / sum(counts)
        weighted_entropy += weight * entropy(subset[target_name])
    
    gain = total_entropy - weighted_entropy
    return gain

def ID3(data, original_data, features, target_attribute_name="PlayTennis", parent_node_class=None):
    # If all target values have the same class, return it
    if len(data[target_attribute_name].unique()) == 1:
        return data[target_attribute_name].iloc[0]

    # If dataset is empty, return the most common class from original data
    elif len(data) == 0:
        return original_data[target_attribute_name].mode()[0]

    # If no more features, return the most common class
    elif len(features) == 0:
        return data[target_attribute_name].mode()[0]

    # Default case
    else:
        # Most common target value of current node
        parent_node_class = data[target_attribute_name].mode()[0]
        
        # Select best feature
        gains = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = gains.index(max(gains))
        best_feature = features[best_feature_index]

        # Tree construction
        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]

        for value in data[best_feature].unique():
            sub_data = data[data[best_feature] == value]
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree

def predict(query, tree):
    for key in query.keys():
        if key in tree:
            try:
                result = tree[key][query[key]]
            except:
                return "Unknown"
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result
    return "Unknown"

# Load dataset
data = pd.DataFrame([
    ['Sunny','Hot','High','Weak','No'],
    ['Sunny','Hot','High','Strong','No'],
    ['Overcast','Hot','High','Weak','Yes'],
    ['Rain','Mild','High','Weak','Yes'],
    ['Rain','Cool','Normal','Weak','Yes'],
    ['Rain','Cool','Normal','Strong','No'],
    ['Overcast','Cool','Normal','Strong','Yes'],
    ['Sunny','Mild','High','Weak','No'],
    ['Sunny','Cool','Normal','Weak','Yes'],
    ['Rain','Mild','Normal','Weak','Yes'],
    ['Sunny','Mild','Normal','Strong','Yes'],
    ['Overcast','Mild','High','Strong','Yes'],
    ['Overcast','Hot','Normal','Weak','Yes'],
    ['Rain','Mild','High','Strong','No'],
], columns=['Outlook','Temperature','Humidity','Wind','PlayTennis'])

# Feature list
features = list(data.columns)
features.remove('PlayTennis')

# Build tree
tree = ID3(data, data, features)
print("Decision Tree:")
pprint.pprint(tree)

# Classify new sample
sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
print("\nNew sample classification:")
print("Sample:", sample)
print("Predicted class:", predict(sample, tree))
