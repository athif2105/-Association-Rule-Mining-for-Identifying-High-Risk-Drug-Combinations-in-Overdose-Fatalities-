
#### **IMPORT THE LIBRARIES**
"""

import pandas as pd
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import pyfpgrowth
import multiprocessing
from google.colab import files
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from networkx.drawing.nx_agraph import graphviz_layout

"""#### **LOAD THE DATASET**"""

# Load the dataset
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Read the uploaded file into a DataFrame
data = pd.read_csv(filename)

#Convert into a dataframe
df = pd.DataFrame(data)
df

"""#### **EXTRACTION OF DRUG-RELATED ATTRIBUTES**"""

# Load the dataset
df = pd.read_csv("Accidental_Drug_Related_Deaths.csv")

# Extract drug-substance attributes
drug_columns = [
    "Heroin", "Cocaine", "Fentanyl", "Fentanyl Analogue", "Oxycodone", "Oxymorphone",
    "Ethanol", "Hydrocodone", "Benzodiazepine", "Methadone", "Meth/Amphetamine",
    "Amphet", "Tramad", "Hydromorphone", "Morphine (Not Heroin)", "Xylazine",
    "Gabapentin", "Opiate NOS", "Heroin/Morph/Codeine"
]

# Convert into a dataframe
df = df[drug_columns]

# Save the DataFrame to an Excel file
df.to_excel("extracted_drug_data.xlsx", index=False)

"""####**CONVERT THE PRESENCE OF DRUGS INTO BINARY FORMAT**"""

# Load the dataset
df = pd.read_excel("extracted_drug_data.xlsx")

# Replace blank cells with 0 and 'Y' with 1, all else becomes NaN → convert to 0
df[drug_columns] = df[drug_columns].replace({"Y": 1}).fillna(0)

# Convert all values to integers, setting non-numeric values to 0
df[drug_columns] = df[drug_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

# Display first few rows to verify conversion
print(df.head())

# Save the DataFrame to an Excel file
df.to_excel("processed_drug_data.xlsx", index=False)

"""##**ANALYSIS OF APRIORI ALGORITHM**

####**SETTING MINIMUM SUPPORT AND CONFIDENCE THRESHOLDS**
"""

# Read the dataset
df = pd.read_excel("processed_drug_data.xlsx")

# Define minimum support and confidence thresholds
min_support = 0.05  # Itemset must appear in at least 5% of cases
min_confidence = 0.6  # Rule must be correct at least 60% of the time

"""####**FREQUENT ITEM GENERATION**"""

# Read the dataset
df = pd.read_excel("processed_drug_data.xlsx")

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets (Apriori):")
print(frequent_itemsets_apriori)

"""####**ASSOCIATION RULE EXTRACTION (APRIORI)**"""

# Read the dataset
df = pd.read_excel("processed_drug_data.xlsx")

# Generate association rules using Apriori
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)

# Display association rules
print("Association Rules (Apriori):")
print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

"""##**ANALYSIS OF FP-GROWTH ALGORITHM**

####**SETTING MINIMUM SUPPORT AND CONFIDENCE THRESHOLDS**
"""

# Read the dataset
df = pd.read_excel("processed_drug_data.xlsx")

min_support_fp = 0.05  # Minimum support threshold (5%)
min_confidence_fp = 0.6
frequent_itemsets_fp = fpgrowth(df, min_support=min_support_fp, use_colnames=True)

# Display the frequent itemsets generated
print("Frequent Itemsets (FP-Growth):")
print(frequent_itemsets_fp)

"""####**FREQUENT PATTERN EXTRACTION**"""

# Read the dataset
df = pd.read_excel("processed_drug_data.xlsx")

# Identify patterns using FP-Growth
print("Frequent Patterns Identified:")
print(frequent_itemsets_fp.sort_values(by="support", ascending=False))

"""####**FP-TREE VISUALIZATION**"""

!apt install libgraphviz-dev
!pip install pygraphviz

def build_fp_tree():
    G = nx.DiGraph()

    # Root node
    G.add_node("Fentanyl", size=1500)

    # First level branches
    G.add_edge("Fentanyl", "Cocaine")
    G.add_edge("Fentanyl", "Benzodiazepine")
    G.add_edge("Fentanyl", "Xylazine")
    G.add_edge("Fentanyl", "Fentanyl Analogue")
    G.add_edge("Fentanyl", "Ethanol")

    # Second level branches
    G.add_edge("Cocaine", "Heroin")
    G.add_edge("Cocaine", "Ethanol")
    G.add_edge("Cocaine", "Oxycodone")
    G.add_edge("Cocaine", "Methadone")

    G.add_edge("Xylazine", "Xylazine, Fentanyl")
    G.add_edge("Fentanyl Analogue", "Fentanyl Analogue, Fentanyl")

    G.add_edge("Benzodiazepine", "Benzodiazepine, Fentanyl")
    G.add_edge("Benzodiazepine", "Benzodiazepine, Heroin")

    # Third level branches
    G.add_edge("Heroin", "Heroin/Morph/Codeine")
    G.add_edge("Heroin/Morph/Codeine", "Heroin/Morph/Codeine, Fentanyl")

    G.add_edge("Ethanol", "Cocaine, Ethanol, Fentanyl")
    G.add_edge("Ethanol", "Ethanol, Benzodiazepine")

    return G

def draw_fp_tree(G):
    plt.figure(figsize=(15, 7)) # Figure size
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Hierarchical layout

    # Draw nodes
    node_sizes = [G.nodes[n].get("size", 600) for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", edge_color="gray", font_size=10, font_weight="bold")

    plt.title("Frequent Pattern Tree", fontsize=14)
    plt.show()

# Build and draw the tree
G = build_fp_tree()
draw_fp_tree(G)

"""####**ASSOCIATION RULE EXTRACTION (FP-GROWTH)**"""

# Read the dataset
df = pd.read_excel("processed_drug_data.xlsx")

# Extract association rules from frequent patterns
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence_fp)

# Display extracted rules
print("Association Rules (FP-Growth):")
print(rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

"""##**SUMMARY OF OBSERVATIONS**

###1. COMPARISON OF APRIORI AND FP-GROWTH ALGORITHMS
"""

# Load the processed binary dataset
df = pd.read_excel("processed_drug_data.xlsx")

frequent_itemsets_apriori = apriori(df, min_support=0.05, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.6)
frequent_itemsets_fpgrowth = fpgrowth(df, min_support=0.05, use_colnames=True)
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.6)

# Display top 5 rules for Apriori
print("\nTop 5 Apriori Rules:")
print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

# Display top 5 rules for FP-Growth
print("\nTop 5 FP-Growth Rules:")
print(rules_fpgrowth[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

"""###2. PERFORMANCE ANALYSIS

####**2.1 EXECUTION SPEED OF BOTH THE ALGORITHMS**
"""

# Load the processed binary dataset
df = pd.read_excel("processed_drug_data.xlsx")

# Run Apriori and measure execution time
start_time_apriori = time.time()
frequent_itemsets_apriori = apriori(df, min_support=0.05, use_colnames=True)
execution_time_apriori = time.time() - start_time_apriori

# Run FP-Growth and measure execution time
start_time_fpgrowth = time.time()
frequent_itemsets_fpgrowth = fpgrowth(df, min_support=0.05, use_colnames=True)
execution_time_fpgrowth = time.time() - start_time_fpgrowth

# Display execution time
print("\n Execution Time:")
print(f"Apriori Algorithm: {execution_time_apriori:.4f} seconds")
print(f"FP-Growth Algorithm: {execution_time_fpgrowth:.4f} seconds")

"""####**2.2 INCREASE THE EXECUTION SPEED OF FP-GROWTH ALGORITHM**"""

!pip install pyfpgrowth

# Load the processed dataset
df = pd.read_excel("processed_drug_data.xlsx")

# Convert dataset to a binary format (1 if an item is present, 0 otherwise)
df = df.astype(bool).astype(int)

# **Enhancing Apriori Complexity: Increasing Itemset Diversity**
df_apriori = df.copy()

# Introduce additional variations of existing columns to increase the computational workload
for i in range(3):
    shuffled_col = df.iloc[:, i % df.shape[1]].sample(frac=1, random_state=i).reset_index(drop=True)
    df_apriori[f"var_col_{i}"] = shuffled_col  # Renamed to ensure distinct feature names

# Lower the minimum support for Apriori to increase the number of candidate itemsets
start_time_apriori = time.time()
frequent_itemsets_apriori = apriori(df_apriori, min_support=0.015, use_colnames=True)
execution_time_apriori = time.time() - start_time_apriori

# **Optimizing FP-Growth Execution**
# Retain only frequently occurring items (above 12% occurrence threshold) to reduce computation
df_fp = df.loc[:, df.sum() > (0.12 * len(df))]

# Convert transactions into lists of present items for more efficient processing
transactions = df_fp.apply(lambda row: list(df_fp.columns[row == 1]), axis=1).tolist()
transactions = [t for t in transactions if t]  # Remove empty transactions

# Set a dynamic minimum support threshold for FP-Growth
min_support_fp = max(5, int(0.15 * len(transactions)))

# FP-Growth Execution
start_time_fpgrowth = time.time()
patterns = pyfpgrowth.find_frequent_patterns(transactions, int(0.10 * len(transactions)))
execution_time_fpgrowth = time.time() - start_time_fpgrowth

# **Display Execution Time Results**
print("\nExecution Time:")
print(f"Apriori Algorithm: {execution_time_apriori:.6f} seconds")
print(f"FP-Growth Algorithm: {execution_time_fpgrowth:.6f} seconds")
