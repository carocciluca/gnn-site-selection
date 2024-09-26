import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# PyTorch Geometric-specific imports
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import (
    ResGatedGraphConv, GATConv, GATv2Conv, TransformerConv, GINEConv,
    GMMConv, NNConv, GENConv, PDNConv, GeneralConv
)

# Metrics from scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split, StratifiedKFold

seed_everything(42)


def create_masks(nodes_df: pd.DataFrame, target_colname: str, apply_size: float, n_splits: int):
    """
    Creates masks for k-fold cross-validation for training, validation, and testing sets.

    Parameters:
    nodes_df (pd.DataFrame): DataFrame containing the node features and target labels.
    target_colname (str): The name of the target column to stratify on.
    apply_size (float): Fraction of the dataset to be used as the apply mask (test set).
    n_splits (int): Number of splits for k-fold cross-validation.

    Returns:
    tuple: apply_mask (torch.Tensor), train_masks (dict), val_masks (dict), test_masks (dict)
    """

    # Create apply mask: Select a fraction of data where target_colname == 0 for the apply/test set
    apply_df = nodes_df[nodes_df[target_colname] == 0].sample(frac=apply_size, random_state=42)
    apply_mask = torch.tensor(nodes_df.index.isin(apply_df.index))

    # Remove apply/test data from the original dataframe to create train/val/test sets
    train_val_test_data = nodes_df.loc[~apply_mask.numpy(), :]

    # Initialize dictionaries to hold training, validation, and test masks for each fold
    train_masks = {}
    val_masks = {}
    test_masks = {}

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    X = train_val_test_data.drop(target_colname, axis=1)  # Features
    y = train_val_test_data[target_colname]  # Labels

    # Generate training, validation, and testing masks for each fold
    for i, (train_val_index, test_index) in enumerate(skf.split(X, y)):
        # Get the training+validation and test set rows for this fold
        train_val_rows = train_val_test_data.iloc[train_val_index]
        test_rows = train_val_test_data.iloc[test_index]

        # Create test mask for this fold
        test_mask = torch.tensor(nodes_df.index.isin(test_rows.index))
        test_masks[i] = test_mask

        # Exclude test mask from training/validation data
        train_val_data = nodes_df.loc[(~apply_mask.numpy()) & (~test_mask.numpy()), :]

        # Split the remaining data into training and validation sets (stratified by target column)
        train_set, val_set = train_test_split(
            train_val_data, test_size=0.25, random_state=42, shuffle=True,
            stratify=train_val_data[target_colname]
        )

        # Create training and validation masks for this fold
        train_masks[i] = torch.tensor(nodes_df.index.isin(train_set.index))
        val_masks[i] = torch.tensor(nodes_df.index.isin(val_set.index))

    return apply_mask, train_masks, val_masks, test_masks


def create_graph_dataset(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                         node_feature_groups: list, edge_features: list, adjacent_only: bool):
    """
    This function creates a graph dataset from a nodes DataFrame and an edges DataFrame.
    It filters, processes, and maps the data to create graph representations suitable for use
    in graph neural networks (GNNs).

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node-level data (e.g., county-level features).
        edges_df (pd.DataFrame): DataFrame containing edge-level data (e.g., county-to-county relationships).
        node_feature_groups (list): List of feature group names to include from the nodes_df.
        edge_features (list): List of edge feature names to include from the edges_df.
        adjacent_only (bool): If True, only include edges that represent adjacent counties.

    Returns:
        ### mapping (dict): Dictionary mapping FIPS codes to the node indices in nodes_df.
        graph_object (torch_geometric.data.Data): A PyTorch Geometric Data object representing the graph.
    """

    # Create a mapping from FIPS codes to the row indices of the nodes_df DataFrame
    mapping = pd.Series(nodes_df.index, index=nodes_df["fips"].values).to_dict()

    # Drop the 'fips' and 'county_name' columns as they are not needed as features
    nodes_df = nodes_df.drop(columns=["fips", "county_name"])

    # Define feature groups and their corresponding columns in the nodes DataFrame
    feature_groups = {
        "BD": ['pop_density', '1_person_hhs', '2_person_hhs', '3_person_hhs', '4_person_hhs',
               '5_person_hhs', '6_person_hhs', '7_plus_person_hhs', 'family_hhs', '18_to_19_yos',
               '20_to_24_yos', '25_to_34_yos', '35_to_44_yos', '45_to_54_yos', '55_to_64_yos',
               '65_to_74_yos', '75_to_84_yos', '85_plus_yos', 'bachelors_or_higher'],
        "WE": ['median_hh_income', 'hh_income_smaller_10k', 'hh_income_10k_to_20k', 'hh_income_20k_to_30k',
               'hh_income_30k_to_40k', 'hh_income_40k_to_50k', 'hh_income_50k_to_60k', 'hh_income_60k_to_75k',
               'hh_income_75k_to_100k', 'hh_income_100k_to_125k', 'hh_income_125k_to_150k', 'hh_income_150k_to_200k',
               'hh_income_200k_plus', 'hh_wealth_smaller_50k', 'hh_wealth_50k_to_100k', 'hh_wealth_100k_to_150k',
               'hh_wealth_150k_to_200k', 'hh_wealth_200k_to_250k', 'hh_wealth_250k_to_300k', 'hh_wealth_300k_to_350k',
               'hh_wealth_350k_to_400k', 'hh_wealth_400k_to_500k', 'hh_wealth_500k_to_750k', 'hh_wealth_750k_to_1m',
               'hh_wealth_1m_plus', 'housing_units_wo_mortgage', 'renter_occupied_housing', 'poverty_rate',
               'unemployment_rate'],
        "TB": ['hhs_1_vehicle', 'hhs_2_vehicles', 'hhs_3_vehicles', 'hhs_4_vehicles', 'hhs_5_plus_vehicles',
               'car_commuters', 'hh_spending_new_cars_and_trucks', 'hh_spending_airline_fares'],
        "LB": ['segment_auto_luxury_lovers', 'segment_status_cars', 'population_with_luxury_vehicle',
               'google_trends_compX', 'gift_fine_jewelery', 'gift_watches', 'fine_dining'],
        "CO": ['has_compA_dealer', 'has_compB_dealer']
    }

    # Gather all features to include based on the provided node_feature_groups
    features_to_include = []
    for feature_group in node_feature_groups:
        features_to_include += feature_groups[feature_group]

    # Convert the selected node features into a PyTorch tensor
    x = torch.from_numpy(nodes_df[features_to_include].values)

    # Create a tensor for the target variable, assuming it indicates the presence of a specific dealer
    y = torch.from_numpy(nodes_df["has_compX_dealer"].values)

    # If only adjacent counties should be considered, filter edges DataFrame accordingly
    if adjacent_only:
        edges_df = edges_df[edges_df["is_adjacent"] == 1.0].copy()

    # Select the relevant columns for edges, including origin, destination, and specified edge features
    edges_df = edges_df[["origin", "destination"] + edge_features].copy()

    # Drop rows where all specified edge features are missing (NaN)
    edges_df.dropna(subset=edge_features, how="all", inplace=True)

    # Fill any remaining NaN values in edge features with 0, indicating no relationship
    edges_df.fillna(0, inplace=True)

    # Map the origin and destination FIPS codes to node indices using the previously created mapping
    edges_df["origin"] = edges_df["origin"].map(mapping)
    edges_df["destination"] = edges_df["destination"].map(mapping)

    # Drop any rows where the mapping failed (i.e., if origin or destination is NaN after mapping)
    edges_df.dropna(how='any', inplace=True)

    # Create the edge index tensor (containing pairs of node indices representing the edges)
    # Transpose to match the format expected by PyTorch Geometric (2 x num_edges)
    edge_index = torch.from_numpy(np.transpose(edges_df[["origin", "destination"]].values))

    # Create the edge attributes tensor (containing the features associated with each edge)
    edge_attr = torch.from_numpy(edges_df.drop(columns=["origin", "destination"]).values)

    # Create a PyTorch Geometric Data object representing the graph, with node features, edges, and edge attributes
    graph_object = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    # Return the mapping from FIPS codes to node indices and the constructed graph object
    return graph_object, features_to_include


def train(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module):
    """
    Train the model for one epoch on the provided data.

    Parameters:
    model (torch.nn.Module): The graoh neural network model to be trained.
    data (torch_geometric.data.Data): The graph data containing features, edge index, and labels.
    optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
    criterion (torch.nn.Module): The loss function used to compute the training loss.

    Returns:
    tuple: A tuple containing the training loss, accuracy, precision, recall, and F1/3 score.
    """
    model.train()  # Set the model to training mode.

    optimizer.zero_grad()  # Clear gradients from the previous step.

    # Perform a forward pass to get the model output.
    out = model(data.x, data.edge_index, data.edge_attr)

    # Extract the labels for the training set (masked by data.train_mask).
    # Unsqueeze adds an extra dimension to match the output shape.
    labels = data.y[data.train_mask].unsqueeze(1)

    # Compute the loss using the criterion (loss function).
    loss = criterion(out[data.train_mask], labels)

    # Perform backpropagation to calculate gradients.
    loss.backward()

    # Update model parameters based on calculated gradients.
    optimizer.step()

    # Apply sigmoid to convert logits to probabilities.
    proba = torch.sigmoid(out[data.train_mask])

    # Generate binary predictions using a threshold of 0.5.
    pred = (proba >= 0.5).int()

    # Compute training metrics using the actual labels and predictions.
    train_acc = accuracy_score(labels.cpu(), pred.cpu())  # Accuracy
    train_prec = precision_score(labels.cpu(), pred.cpu(), zero_division=0)  # Precision
    train_rec = recall_score(labels.cpu(), pred.cpu())  # Recall
    train_fbeta = fbeta_score(labels.cpu(), pred.cpu(), beta=1 / 3)  # F1/3 score

    # Return the computed loss and training metrics.
    return loss.item(), train_acc, train_prec, train_rec, train_fbeta


def validate(model: torch.nn.Module, data: Data, criterion: torch.nn.Module):
    """
    Validate the model on the provided validation set.

    Parameters:
    model (torch.nn.Module): The graph neural network model to be evaluated.
    data (torch_geometric.data.Data): The graph data containing features, edge index, and labels.
    criterion (torch.nn.Module): The loss function used to compute the validation loss.

    Returns:
    tuple: A tuple containing the validation loss, accuracy, precision, recall, and F1/3 score.
    """
    model.eval()  # Set the model to evaluation mode.

    with torch.no_grad():  # Disable gradient computation to save memory and computations.
        # Perform a forward pass to get the model output.
        out = model(data.x, data.edge_index, data.edge_attr)

        # Extract the labels for the validation set (masked by data.val_mask).
        # Unsqueeze adds an extra dimension to match the output shape.
        labels = data.y[data.val_mask].unsqueeze(1)

        # Compute the loss using the criterion (loss function) on the validation set.
        loss = criterion(out[data.val_mask], labels)

        # Apply sigmoid to convert logits to probabilities.
        proba = torch.sigmoid(out[data.val_mask])

        # Generate binary predictions using a threshold of 0.5.
        pred = (proba >= 0.5).int()

        # Compute validation metrics using the actual labels and predictions.
        val_acc = accuracy_score(labels.cpu(), pred.cpu())  # Accuracy
        val_prec = precision_score(labels.cpu(), pred.cpu(), zero_division=0)  # Precision
        val_rec = recall_score(labels.cpu(), pred.cpu())  # Recall
        val_fbeta = fbeta_score(labels.cpu(), pred.cpu(), beta=1 / 3)  # F1/3 score

    # Return the computed loss and validation metrics.
    return loss.item(), val_acc, val_prec, val_rec, val_fbeta


def test(model: torch.nn.Module, data: Data, criterion: torch.nn.Module):
    """
    Evaluate the model on the test set.

    Parameters:
    model (torch.nn.Module): The graph neural network model to be evaluated.
    data (torch_geometric.data.Data): The graph data containing features, edge index, and labels.
    criterion (torch.nn.Module): The loss function used to compute the test loss.

    Returns:
    tuple: A tuple containing the test accuracy, precision, recall, and F1/3 score.
    """
    model.eval()  # Set the model to evaluation mode.

    with torch.no_grad():  # Disable gradient computation to save memory and computations.
        # Perform a forward pass to get the model output.
        out = model(data.x, data.edge_index, data.edge_attr)

        # Extract the labels for the test set (masked by data.test_mask).
        # Unsqueeze adds an extra dimension to match the output shape.
        labels = data.y[data.test_mask].unsqueeze(1)

        # Compute the loss using the criterion (loss function) on the test set.
        loss = criterion(out[data.test_mask], labels)

        # Apply sigmoid to convert logits to probabilities.
        proba = torch.sigmoid(out[data.test_mask])

        # Generate binary predictions using a threshold of 0.5.
        pred = (proba >= 0.5).int()

        # Compute test metrics using the actual labels and predictions.
        test_acc = accuracy_score(labels.cpu(), pred.cpu())  # Accuracy
        test_prec = precision_score(labels.cpu(), pred.cpu(), zero_division=0)  # Precision
        test_rec = recall_score(labels.cpu(), pred.cpu())  # Recall
        test_fbeta = fbeta_score(labels.cpu(), pred.cpu(), beta=1 / 3)  # F1/3 score

    # Return the computed test metrics.
    return test_acc, test_prec, test_rec, test_fbeta


def apply(model: torch.nn.Module, data: Data):
    """
    Apply the model to the provided data to get predictions and probabilities.

    Parameters:
    model (torch.nn.Module): The graph neural network model to be used for inference.
    data (torch_geometric.data.Data): The graph data containing features, edge index, and optional masks.

    Returns:
    tuple: A tuple containing the predicted labels, probabilities, and raw model outputs.
    """
    model.eval()  # Set the model to evaluation mode.

    with torch.no_grad():  # Disable gradient computation to save memory and computations.
        # Perform a forward pass to get the model output.
        out = model(data.x, data.edge_index, data.edge_attr)

        # Apply sigmoid to convert logits to probabilities.
        proba = torch.sigmoid(out)

        # Generate binary predictions using a threshold of 0.5.
        pred = (proba >= 0.5).int()

        # Select predictions corresponding to the apply_mask.
        predictions = pred[data.apply_mask]

    # Return the predictions, probabilities, and raw model outputs.
    return predictions, proba, out


def config_model(model_class: torch.nn.Module, data: Data) -> 'torch.nn.Module':
    """
    Configure and instantiate the GNN model based on the provided model class and graph data.

    Parameters:
    model_class (torch.nn.Module): The class of the graph neural network (GNN) model to be instantiated.
    data (torch_geometric.data.Data): The graph data object containing node features, edge features, and edge indices.

    Returns:
    torch.nn.Module: The instantiated model, configured with the appropriate input and output dimensions based on
                     the data.
    """
    # Dictionary that defines parameters for each model class based on the input data
    model_params = {
        ResGatedGraphConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'edge_dim': data.num_edge_features
        },
        GATConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'edge_dim': data.num_edge_features
        },
        GATv2Conv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'edge_dim': data.num_edge_features
        },
        TransformerConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'edge_dim': data.num_edge_features
        },
        GINEConv: {
            'nn': nn.Linear(data.num_node_features, 1, 5),
            'edge_dim': data.num_edge_features
        },
        GMMConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'dim': data.num_edge_features,
            'kernel_size': 5
        },
        NNConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'nn': nn.Linear(data.num_edge_features, data.num_node_features, 5)
        },
        GENConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'edge_dim': data.num_edge_features
        },
        PDNConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'edge_dim': data.num_edge_features,
            'hidden_channels': data.num_edge_features
        },
        GeneralConv: {
            'in_channels': data.num_node_features,
            'out_channels': 1,
            'in_edge_channels': data.num_edge_features
        }
    }

    # Instantiate the model using the corresponding parameters for the given model class
    configured_model = model_class(**model_params[model_class])

    return configured_model


def train_and_evaluate_model(model: torch.nn.Module, data: Data, print_logs: bool,
                             print_summaries: bool, n_epochs: int = 2000, patience: int = 50)\
        -> tuple[torch.nn.Module, dict, dict]:
    """
    Train, validate, and test the GNN model with early stopping, logging, and summary visualization.

    Parameters:
    model (torch.nn.Module): The graph neural network model to train.
    data (torch_geometric.data.Data): The graph dataset containing node features, edge attributes, and masks.
    print_logs (bool): Whether to print detailed logs after each epoch during training.
    print_summaries (bool): Whether to print and plot training summaries after each fold.
    n_epochs (int, optional): The maximum number of epochs to train the model. Default is 200.
    patience (int, optional): The number of epochs to wait for validation improvement before early stopping.
                              Default is 50.

    Returns:
    tuple: A tuple containing:
        - torch_geometric.nn.Module: The trained model with the best state restored after early stopping.
        - dict: A dictionary containing test metrics (accuracy, precision, recall, F-beta score).
        - dict: A dictionary containing training and validation logs (losses, F-beta scores for each epoch).
    """

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Initialize training parameters for early stopping
    patience_counter = 0
    best_val_loss = float('inf')

    # Initialize lists to store losses and F-beta scores
    train_losses = []
    val_losses = []
    train_fbetas = []
    val_fbetas = []

    # Training loop
    for epoch in range(1, n_epochs+1):
        # Train and validate the model
        train_loss, train_acc, train_prec, train_rec, train_fbeta = train(model, data, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_fbeta = validate(model, data, criterion)

        if print_logs:
            print(
                f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f}, Train F-Beta: {train_fbeta:.4f} | \
                Val Loss: {val_loss:.4f}, Val F-Beta: {val_fbeta:.4f}')

        # Store metrics for later analysis
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_fbetas.append(train_fbeta)
        val_fbetas.append(val_fbeta)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()  # Save the best model state.
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if print_summaries:
                    print(f"Stopping early at epoch {epoch}!")
                break

    # Restore the best model state
    model.load_state_dict(best_model)

    # Test the model after training
    test_acc, test_prec, test_rec, test_fbeta = test(model, data, criterion)

    if print_summaries:
        print("\nTraining Complete!\nSummary:")

        # Plot training and validation metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plotting training and validation losses
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='orange')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plotting training and validation F-betas
        ax2.plot(train_fbetas, label='Train F-Beta', color='blue')
        ax2.plot(val_fbetas, label='Validation F-Beta', color='orange')
        ax2.set_title('Training and Validation F-Beta')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F-Beta Score')
        ax2.legend()

        plt.show()

        # Print test metrics
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test F-beta: {test_fbeta:.4f}")

    # Prepare test metrics to return
    test_metrics = {
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f_beta": test_fbeta
    }

    # Prepare logs to return
    logs = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_fbetas": train_fbetas,
        "val_fbetas": val_fbetas
    }

    return model, test_metrics, logs


def perform_ablation_study(nodes_df, edges_df, node_vars, edge_vars, adjacent_edges_only, gnn_models, masks,
                           print_logs=True, print_summaries=True) -> pd.DataFrame:
    """
    Perform an ablation study by evaluating different combinations of node features, edge features, and graph neural
    network models.

    Parameters:
    node_vars (list): List of node feature groups to test.
    edge_vars (list): List of edge feature groups to test.
    adjacent_edges_only (list): List of configurations for using only adjacent edges.
    gnn_models (list): List of GNN model classes to test.
    print_logs (bool): Whether to print detailed logs during training.
    print_summaries (bool): Whether to print and plot summaries after training.

    Returns:
    pd.DataFrame: A DataFrame summarizing the average performance metrics for each combination of parameters.
    """
    apply_mask, train_masks, val_masks, test_masks = masks

    # Initialize dictionary to store results
    results_dict = {
        "Node Feature Group": [],
        "Edge Feature Group": [],
        "Adjacent Edges Only": [],
        "Model": [],
        "Average Accuracy": [],
        "Average Precision": [],
        "Average Recall": [],
        "Average F-beta": []
    }

    # Calculate the number of combinations
    n_combinations = len(node_vars) * len(edge_vars) * len(adjacent_edges_only) * len(gnn_models)
    n_counter = 1

    # Iterate over all combinations of node variables, edge variables, and adjacency configurations
    for node_var in node_vars:
        for edge_var in edge_vars:
            for adjacent_edges_only_config in adjacent_edges_only:

                # Create graph dataset for the current combination
                data, node_features_included = create_graph_dataset(
                    nodes_df=nodes_df,
                    edges_df=edges_df,
                    node_feature_groups=node_var,
                    edge_features=edge_var,
                    adjacent_only=adjacent_edges_only_config
                )

                # Ensure tensors have the correct data type
                data.x = data.x.float()  # Convert node features to float32
                data.edge_attr = data.edge_attr.float()  # Convert edge attributes to float32
                data.y = data.y.float()  # Convert target labels to float32

                # Iterate over each GNN model class
                for model_class in gnn_models:
                    print("=================================== NOW EVALUATING ===================================")
                    print(f"COMBINATION {n_counter} of {n_combinations} IN THIS ABLATION STUDY ROUND")
                    print(f"County Variable Combination: {node_var} | Number of node features: {data.x.shape[1]}")
                    print(f"Edge Variable Combination: {edge_var} | Number of edge features: {data.edge_attr.shape[1]}")
                    print(f"Adjacent Edges Only: {adjacent_edges_only_config} | \
                    Number of edges: {data.edge_index.shape[1]}")
                    print(f"Model Class: {model_class}")

                    # Number of folds in cross-validation
                    n_folds = len(train_masks.keys())

                    # Initialize lists to store metrics for each fold
                    fold_test_accs = []
                    fold_test_precs = []
                    fold_test_recs = []
                    fold_test_fbetas = []

                    # Iterate over each fold
                    for n in range(n_folds):
                        if print_logs:
                            print(f"\nFold {n + 1}/{n_folds}:")

                        # Set the masks for the current fold
                        data.train_mask = train_masks[n]
                        data.val_mask = val_masks[n]
                        data.test_mask = test_masks[n]
                        data.apply_mask = apply_mask

                        # Instantiate the model using the class and its corresponding parameters
                        model = config_model(model_class, data)

                        # Call the refactored function for training, validation, and testing
                        model, test_metrics, logs = train_and_evaluate_model(
                            model=model,
                            data=data,
                            print_logs=print_logs,
                            print_summaries=print_summaries
                        )

                        # Store fold metrics
                        fold_test_accs.append(test_metrics["accuracy"])
                        fold_test_precs.append(test_metrics["precision"])
                        fold_test_recs.append(test_metrics["recall"])
                        fold_test_fbetas.append(test_metrics["f_beta"])

                    n_counter += 1

                    # Store results for the current combination
                    results_dict["Node Feature Group"].append(node_var)
                    results_dict["Edge Feature Group"].append(edge_var)
                    results_dict["Adjacent Edges Only"].append(adjacent_edges_only_config)
                    results_dict["Model"].append(model_class)
                    results_dict["Average Accuracy"].append(np.mean(fold_test_accs))
                    results_dict["Average Precision"].append(np.mean(fold_test_precs))
                    results_dict["Average Recall"].append(np.mean(fold_test_recs))
                    results_dict["Average F-beta"].append(np.mean(fold_test_fbetas))

    # Convert results to DataFrame and print summary
    results_df = pd.DataFrame.from_dict(results_dict)
    print("================================ EVALUATION COMPLETE =================================")
    print(
        "Ablation study round is complete! All combinations were evaluated using cross-validation.\nHere are the \
        respective average performances on the test sets across all folds:\n"
    )
    print(results_df)

    return results_df

