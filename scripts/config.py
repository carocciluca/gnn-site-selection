# Configuration file for constants and parameters
import os


# Paths
# Check if running in an interactive environment
try:
    __file__
except NameError:
    # Use the current working directory if __file__ is not defined
    BASE_DIR = os.getcwd()
else:
    # Use the directory of the config.py file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

COUNTY_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'county_data_anonymized_raw.csv')
ADJACENCY_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'county_adjacency.txt')
POPULATION_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'county_populations.csv')
SOCIAL_CONNECTEDNESS_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'c2c_social_connectedness.tsv')
COMMUTES_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'c2c_commutes_16-20.xlsx')
COMMUTES_OLD_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'c2c_commutes_11-15.xlsx')
MIGRATION_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'c2c_migration_flows_16-20.xlsx')
POPULATION_FLOWS_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'population_flows.csv')

COUNTY_PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'county_data_processed.csv')
EDGES_PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'edge_data_processed.csv')


RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')

ABL_ROUND1_RESULTS_PATH = os.path.join(RESULTS_DIR, 'ablation_study_round1_results.csv')
ABL_ROUND2_RESULTS_PATH = os.path.join(RESULTS_DIR, 'ablation_study_round2_results.csv')
EXPANSION_TARGETS_PATH = os.path.join(RESULTS_DIR, 'expansion_targets.csv')
FEATURE_EXPLANATIONS_PATH = os.path.join(RESULTS_DIR, 'explanations')


MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')

# Filtering criteria
STATES_TO_EXCLUDE = [ # Exclude outlying areas and non-contiguous states
    "02", # Alaska
    "15", # Hawaii
    "60", # American Samoa
    "64", # Federated States of Micronesia
    "66", # Guam
    "68", # Marshall Islands
    "69", # Commonwealth of the Northern Mariana Islands
    "70", # Palau
    "72", # Puerto Rico
    "74", # U.S. Minor Outlying Islands
    "78"  # U.S. Virgin Islands
 ]

COUNTIES_TO_EXCLUDE = [ # Exclude counties with populations <300 as of 2023
    "48261", # Kenedy County, TX
    "48269", # King County, TX
    "48301"  # Loving County, TX
]
