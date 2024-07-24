import matplotlib.pyplot as plt

def plot_rules(rules, algorithm_name):
    if isinstance(rules, list) and len(rules) > 0 and isinstance(rules[0], dict):
        support = [rule['support'] for rule in rules]
        confidence = [rule['confidence'] for rule in rules]
        lift = [rule['lift'] for rule in rules]

        plt.figure(figsize=(10, 6))
        plt.scatter(support, confidence, alpha=0.5, c='blue', label='Confidence')
        plt.scatter(support, lift, alpha=0.5, c='red', label='Lift')
        plt.title(f'Association Rules ({algorithm_name})')
        plt.xlabel('Support')
        plt.ylabel('Metric Value')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(f'static/{algorithm_name}_rules.png')
        plt.close()
    else:
        raise ValueError("The rules parameter must be a list of dictionaries containing 'support', 'confidence', and 'lift' keys.")

def plot_apriori_rules(rules, filename='apriori_rules.png'):
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, marker="o", edgecolors="w", s=100)
    plt.title('Apriori: Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_fp_growth_rules(rules, filename='fp_growth_rules.png'):
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, marker="o", edgecolors="w", s=100)
    plt.title('FP-Growth: Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_frequency_domain_features(frequency_features, filename='frequency_domain_features.png'):
    frequency_features.plot(kind='bar', figsize=(10, 6))
    plt.title('Frequency Domain Features')
    plt.ylabel('Value')
    plt.xlabel('Features')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_time_domain_features(time_features, filename='time_domain_features.png'):
    time_features.plot(kind='bar', figsize=(10, 6))
    plt.title('Time Domain Features')
    plt.ylabel('Value')
    plt.xlabel('Features')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
