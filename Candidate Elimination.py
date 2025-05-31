import csv

def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        all_rows = list(reader)
        header = all_rows[0]
        data = all_rows[1:]
    return header, data

def more_general(h1, h2):
    for x, y in zip(h1, h2):
        if x != '?' and (x != y and x != '0'):
            return False
    return True

def generalize_S(example, S):
    for i, val in enumerate(example):
        if S[i] != val:
            S[i] = '?'
    return S

def specialize_G(example, G, domains):
    new_G = []
    for g in G:
        if not more_general(g, example):
            continue
        for i in range(len(g)):
            if g[i] == '?':
                for value in domains[i]:
                    if value != example[i]:
                        new_hypothesis = g[:]
                        new_hypothesis[i] = value
                        if new_hypothesis not in new_G:
                            new_G.append(new_hypothesis)
            elif g[i] != example[i]:
                continue
    return new_G

def candidate_elimination(training_data):
    num_attributes = len(training_data[0]) - 1
    domains = [set() for _ in range(num_attributes)]

    for row in training_data:
        for i in range(num_attributes):
            domains[i].add(row[i])

    S = ['0'] * num_attributes  # Most specific hypothesis
    G = [['?'] * num_attributes]  # Most general hypothesis

    for row in training_data:
        example, label = row[:-1], row[-1]
        if label == 'Yes':
            # Remove inconsistent hypotheses from G
            G = [g for g in G if more_general(g, example)]
            # Update S
            if S == ['0'] * num_attributes:
                S = example[:]
            else:
                S = generalize_S(example, S)
        elif label == 'No':
            # Specialize G
            G = specialize_G(example, G, domains)
            # Remove generalizations that are more general than S
            G = [g for g in G if not more_general(S, g)]

    return S, G

# Load and run
if __name__ == "__main__":
    filename = 'data.csv'  # Make sure this file exists in the same directory
    header, training_data = load_csv(filename)
    S, G = candidate_elimination(training_data)

    print("Final Specific Hypothesis S:")
    print(S)
    print("\nFinal General Hypotheses G:")
    for g in G:
        print(g)
