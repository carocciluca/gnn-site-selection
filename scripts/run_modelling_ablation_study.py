# Standard library imports
import os
from ast import literal_eval

# Third-party library imports
import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.explain import CaptumExplainer, Explainer
from torch_geometric.nn import (ResGatedGraphConv, GATConv, GATv2Conv, TransformerConv, GINEConv, GMMConv, NNConv,
                                GENConv, PDNConv, GeneralConv)

# Local application/library-specific imports
from scripts.config import (ABL_ROUND1_RESULTS_PATH, ABL_ROUND2_RESULTS_PATH, BEST_MODEL_PATH,
                            COUNTY_PROCESSED_DATA_PATH, EDGES_PROCESSED_DATA_PATH, FEATURE_EXPLANATIONS_PATH,
                            EXPANSION_TARGETS_PATH)
from modelling import (create_masks, config_model, create_graph_dataset, train_and_evaluate_model,
                       perform_ablation_study, apply)


def main():
    # Load the county-level and county-to-county edge data for the GNN model, ensuring FIPS codes are treated as strings
    counties_df = pd.read_csv(COUNTY_PROCESSED_DATA_PATH, dtype={"fips": str})
    c2c_df = pd.read_csv(EDGES_PROCESSED_DATA_PATH, dtype={"origin": str, "destination": str})
    print(c2c_df)

    # Set the seed for reproducibility, ensuring consistent results across runs
    seed_everything(seed=42)

    # Generate masks for training, validation, testing, and application sets, ensuring stratification of target variable
    masks = create_masks(nodes_df=counties_df,
                         target_colname="has_compX_dealer",
                         apply_size=0.2,  # Reserve 20% of counties w/o a dealership to apply the model after training
                         n_splits=5)  # 5 splits for 5-fold cross-validation

    # Unpack the masks for later use
    apply_mask, train_masks, val_masks, test_masks = masks

    # Define the variable combinations for Round 1 of the ablation study
    r1_county_var_combs = [
        ["BD"],   # Basic Demographics
        ["WE"],   # Wealth
        ["TB"],   # Transportation Behavior
        ["LB"],   # Luxury Behavior
        ["CO"]    # Competition
    ]

    # Define edge feature combinations for Round 1 (Social Connectedness and Mobility Connectedness)
    r1_c2c_var_combs = [
        ["scaled_sci", "scaled_mci"]  # Social Connectedness Index (SCI) and Mobility Connectedness Index (MCI)
    ]

    # Only use adjacent edges for Round 1 of the study
    r1_adjacent_edges_only = [
        True  # True means only adjacent counties are considered connected, i.e., share an edge in the network
    ]

    # List of GNN models to be evaluated in the ablation study (from PyTorch Geometric)
    r1_models = [
        ResGatedGraphConv,
        GATConv,
        GATv2Conv,
        TransformerConv,
        GINEConv,
        GMMConv,
        NNConv,
        GENConv,
        PDNConv,
        GeneralConv
    ]

    # Perform Round 1 of the ablation study using the defined variable combinations, edge features, and models
    round1_results = perform_ablation_study(nodes_df=counties_df, edges_df=c2c_df, node_vars=r1_county_var_combs,
                                            edge_vars=r1_c2c_var_combs, adjacent_edges_only=r1_adjacent_edges_only,
                                            gnn_models=r1_models, masks=masks, print_logs=False, print_summaries=False)

    # Save the results from Round 1 to a CSV file
    round1_results.to_csv(ABL_ROUND1_RESULTS_PATH, index=False)

    # Prepare the results from Round 1 for analysis by adjusting variable group columns
    round1_results_adjusted = round1_results[["Node Feature Group", "Average F-beta"]]
    round1_results_adjusted.loc[:, "Node Feature Group"] = round1_results_adjusted["Node Feature Group"].astype(str)

    # Rank the variable groups by their median F-beta scores
    round1_ranking = round1_results_adjusted.groupby("Node Feature Group").median().reset_index()
    round1_ranking.sort_values(by="Average F-beta", ascending=False, inplace=True)  # Sort in descending order of F-beta

    # Create the order of variable combinations for Round 2 based on Round 1 rankings
    combination_order_round2 = round1_ranking["Node Feature Group"].apply(literal_eval).to_list()

    # Initialize the list for Round 2 with the best-performing combination from Round 1
    r2_county_var_combs = [combination_order_round2[0]]

    # Build cumulative combinations for Round 2 by appending combinations sequentially
    # Starting with best-performing feature category alone, then best- and second-best-performing categories jointly,...
    for combination in combination_order_round2[1:]:
        r2_county_var_combs.append(r2_county_var_combs[0] + combination)

    # Define edge feature combinations for Round 2 of the ablation study
    r2_c2c_var_combs = [
        ["scaled_sci"],  # Social Connectedness only
        ["scaled_mci"],  # Mobility Connectedness only
        ["scaled_sci", "scaled_mci"]  # Both SCI and MCI
    ]

    # Evaluate both adjacent-only and non-adjacent edges in Round 2
    r2_adjacent_edges_only = [
        True,   # Only adjacent counties
        False   # Include non-adjacent counties as well
    ]

    # Use the same GNN models as Round 1
    r2_models = r1_models

    # Perform Round 2 of the ablation study using the new combinations of node and edge variables, adjacency, and models
    round2_results = perform_ablation_study(nodes_df=counties_df, edges_df=c2c_df, node_vars=r2_county_var_combs,
                                            edge_vars=r2_c2c_var_combs, adjacent_edges_only=r2_adjacent_edges_only,
                                            gnn_models=r2_models, masks=masks, print_logs=False, print_summaries=False)

    # Save the results from Round 2 to a CSV file
    round2_results.to_csv(ABL_ROUND2_RESULTS_PATH, index=False)

    # Sort Round 2 results by F-beta score to identify the best-performing combination
    round2_ranking = round2_results.sort_values(by="Average F-beta", ascending=False)

    # Retrieve the best-performing model and feature combination from Round 2
    best_node_features = round2_ranking["Node Feature Group"].head(1).item()
    best_edge_features = round2_ranking["Edge Feature Group"].head(1).item()
    best_adjacency_edges_only = round2_ranking["Adjacent Edges Only"].head(1).item()
    best_model_class = round2_ranking["Model"].head(1).item()

    # Create the graph dataset using the best node and edge feature combinations identified in Round 2
    best_data, best_node_features_included = create_graph_dataset(nodes_df=counties_df, edges_df=c2c_df,
                                                                  node_feature_groups=best_node_features,
                                                                  edge_features=best_edge_features,
                                                                  adjacent_only=best_adjacency_edges_only)

    # Assign the appropriate masks (apply, train, validation, test) to the data object
    best_data.apply_mask = apply_mask
    best_data.train_mask = train_masks[0]
    best_data.val_mask = val_masks[0]
    best_data.test_mask = test_masks[0]

    # Ensure all tensors are converted to the correct data type (float32)
    best_data.x = best_data.x.float()  # Node features
    best_data.edge_attr = best_data.edge_attr.float()  # Edge attributes
    best_data.y = best_data.y.float()  # Target labels

    # Print summary of the best model-feature combination identified from the ablation study
    print(f"========== TRAINING AND TESTING BEST MODEL-FEATURE COMBINATION FROM ABLATION STUDY ==========")
    print(f"County Variable Combination: {best_node_features} | Number of node features: {best_data.x.shape[1]}")
    print(f"Edge Variable Combination: {best_edge_features} | Number of edge features: {best_data.edge_attr.shape[1]}")
    print(f"Adjacent Edges Only: {best_adjacency_edges_only} | Number of edges: {best_data.edge_index.shape[1]}")
    print(f"Model Class: {best_model_class}")

    # Configure and instantiate the best GNN model based on the identified model class and data
    best_model_config = config_model(model_class=best_model_class, data=best_data)

    # Train and evaluate the model using the best configuration and print training/testing logs
    best_model, test_metrics, logs = train_and_evaluate_model(model=best_model_config, data=best_data, print_logs=True,
                                                              print_summaries=True)

    # Save the best-trained model to disk for future use
    # torch.save(best_model, BEST_MODEL_PATH)

    # Load the trained model for deployment
    try:
        model = torch.load(BEST_MODEL_PATH)
    except:
        model = best_model

    # Apply the trained model to make predictions on the apply/test set
    predictions, probabilities, output = apply(model, best_data)

    # Filter and store probabilities for the apply/test set counties
    application_probs = probabilities[np.array(apply_mask)]
    application_probs_positive = application_probs[application_probs >= 0.5]

    # Filter predictions for expansion targets (counties without an existing dealership but likely for expansion)
    predictions = np.array(predictions)
    probabilities = probabilities[apply_mask == 1].detach().numpy()
    expansion_predictions = counties_df.reset_index()[np.array(apply_mask)][predictions == 1]

    # Create a DataFrame containing the predictions and relevant information for each expansion target
    expansion_df = expansion_predictions[
        ["fips", "county_name", "has_compA_dealer", "has_compB_dealer"]]
    prediction_probs = application_probs_positive.detach().numpy()
    expansion_df.loc[:, "prediction_probs"] = prediction_probs
    expansion_df.sort_values(by="prediction_probs", ascending=False, inplace=True)

    expansion_df.to_csv(EXPANSION_TARGETS_PATH, index=False)

    # Instantiate an explainer using Captum to interpret the model's predictions
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='regression',
            task_level="node",
            return_type='raw',
        ),
    )

    # Loop through expansion predictions and store visualization of top 10 most important features per expansion
    # prediction
    for idx in expansion_predictions.index:

        fig_name = expansion_predictions.loc[idx, 'county_name'].lower().replace(" ", "_").replace(",", "")
        explanation = explainer(
            x=best_data.x,
            edge_index=best_data.edge_index,
            edge_attr=best_data.edge_attr,
            index=idx)

        fig_save_path = os.path.join(FEATURE_EXPLANATIONS_PATH, fig_name + '.png')
        explanation.visualize_feature_importance(path=fig_save_path, top_k=10, feat_labels=best_node_features_included)


if __name__ == "__main__":
    main()
