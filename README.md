# GNN-Based Dealership Site Selection (gnn-site-selection)

This repository implements the Graph Neural Network (GNN) approach for optimizing luxury vehicle dealership site selection described in the "Optimizing Luxury Vehicle Dealership Networks: A Graph Neural Network Approach to Site Selection" paper (https://arxiv.org/abs/2408.13961). The study uses county-level variables as node features and the Social Connectedness Index (SCI) and Mobility Connectedness Index (MCI) as edge features to analyze county-level relationships. Additionally, the repository includes data and preprocessing for other edge features (Migration Connectedness and Population Connectedness), which are not assessed in the current ablation study but can be used for future extensions.

## Table of Contents
1. Project Overview
2. Methodology
3. Repository Structure
4. Installation
5. Usage
6. Ablation Study
7. Results
8. Extensions and Future Work 
9. Authors 
10. References
____________________________________

## 1. Project Overview
This project uses GNNs to predict optimal luxury vehicle dealership locations by analyzing county-level data and inter-county relationships, with a focus on the Social Connectedness Index (SCI) and Mobility Connectedness Index (MCI) as edge features. Additionally, the preprocessing scripts include two other connectedness indices—Migration Connectedness (MigCI) and Population Connectedness (PCI)—which are processed but not evaluated in the current ablation study. These can serve as future extensions to the study.

## 2. Methodology
The study involves the following key steps:

1. **Data Processing**: Raw data on counties and inter-county relationships (adjacency, mobility, social connectedness) is preprocessed.
2. **Graph Construction**: Nodes represent counties, and edges capture relationships such as Social and Mobility Connectedness.
3. **Model Training**: GNN models (e.g., ResGatedGraphConv, GATConv) are trained using SCI and MCI as edge features.
4. **Ablation Study**: An ablation study is conducted to evaluate the performance of different GNN models and feature combinations.

Please consult the paper for a more detailed description of the underlying methodology.

### Note on Additional Edge Features
While the current ablation study uses only SCI and MCI as edge features, the preprocessing scripts also process Migration Connectedness (MigCI) and Population Connectedness (PCI). These indices are available in the processed edge data but were not used in the study. They can be leveraged for future experiments and model enhancements.

## 3. Repository Structure

gnn-site-selection/

├── data/                              # Contains raw and processed data files <br>
│   ├── processed/                     # Placeholder for processed data files (node features, edge features) <br>
│   └── raw/                           # Placeholder for raw data files (e.g., commutes, migration flows) <br>
├── models/                            # Placeholder for saved models <br>
├── results/                           # Results from ablation studies and model outputs <br>
│   ├── explanations/                  # Visualizations of feature importance by county <br>
│   ├── ablation_study_round1_results.csv <br>
│   ├── ablation_study_round2_results.csv <br>
│   └── expansion_targets.csv          # Predicted targets for dealership expansion <br>
├── scripts/                           # Scripts for data processing and model execution <br>
│   ├── __init__.py <br>
│   ├── config.py                       # Path and variable configurations <br>
│   ├── run_preprocess_edge_features.py # Preprocess edge features (SCI, MCI, MigCI, PCI) <br>
│   └── ... (additional scripts) <br>
├── .gitignore                         # Files and folders to ignore in version control <br>
├── LICENSE.txt                        # License for the project <br>
├── README.md                          # This README file <br>
└── requirements.txt                   # The required packages and versions fir this project

**Note**: Some raw data files cannot be provided due to size/copyright constraints. Below are the sources of publicly available files used in this study:
* **Social Connectedness Data**: https://data.humdata.org/dataset/social-connectedness-index
* **Commuting Data**: https://www.census.gov/topics/employment/commuting/guidance/flows.html
* **Population Flow Data**: https://github.com/GeoDS/COVID19USFlows-WeeklyFlows
* **Migration Data**: https://www.census.gov/topics/population/migration/guidance/county-to-county-migration-flows.html
* 

## 4. Installation
1. **Clone the repository**:

git clone https://github.com/your_username/gnn-site-selection.git <br>
cd gnn-site-selection

2. **Install dependencies**: Ensure you have Python 3.7+ installed. Install required Python packages:

pip install -r requirements.txt

3. **Data Setup**: Place the raw data files in the data/raw/ directory. Preprocess the node and edge data, including SCI, MCI, MigCI, and PCI, by running:

python scripts/run_preprocess_node_features.py
python scripts/run_preprocess_edge_features.py

## 5. Usage
### 5.1 Running the Ablation Study
To run the ablation study (which currently uses SCI and MCI as edge features), execute the following command:

python scripts/run_modelling_ablation_study.py

### 5.2 Edge Features
Social Connectedness Index (SCI) and Mobility Connectedness Index (MCI) are used as edge features in the ablation study.
Migration Connectedness (MigCI) and Population Connectedness (PCI) are preprocessed and available in the edge_data_processed.csv file but are not included in the current ablation study. You can easily extend the study by incorporating them.

## 6. Ablation Study
The ablation study evaluates node and edge feature combinations in two rounds:

* **Round 1**: Tests individual node and edge features (SCI, MCI) with various GNN models.
* **Round 2**: Combines the top-performing features from Round 1 (see the paper for a detailed description).
For each round, the results are stored in results/ablation_study_round1_results.csv and results/ablation_study_round2_results.csv.

## 7. Extensions and Future Work
Additional Edge Features
While the current study only evaluates SCI and MCI as edge features, the repository includes the pre-processing for Migration Connectedness (MigCI) and Population Connectedness (PCI). These can be integrated into future experiments by updating the ablation study configuration to assess their impact on dealership site selection predictions.

## 8. Results
The final results indicate that competition and basic demographic attributes, combined with mobility data, are strong predictors of dealership site selection. Visualization of feature importance for selected counties is available in the results/explanations/ directory.

## 9. Authors
This project was developed by the following authors:

* Luca Silvano Carocci - Nova School of Business and Economics | Universidade NOVA de Lisboa | Carcavelos, Portugal (53942@novasbe.pt)
* Qiwei Han - Nova School of Business and Economics | Universidade NOVA de Lisboa | Carcavelos, Portugal (qiwei.han@novasbe.pt)

## 9. References
This work is based on the methodology described in the original paper (https://arxiv.org/abs/2408.13961).

