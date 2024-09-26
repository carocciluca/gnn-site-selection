import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, MinMaxScaler


def filter_rows(county_df: pd.DataFrame, cols_to_filter: list,
                states_to_exclude: list, counties_to_exclude: list) -> pd.DataFrame:
    """
    Filters out rows from the county DataFrame based on specified states and counties to exclude.

    Parameters:
    county_df (pd.DataFrame): DataFrame containing county data with a 'fips' column and other columns to be filtered.
    cols_to_filter (list): List of column names in the DataFrame to apply the filtering on.
    states_to_exclude (list): List of state FIPS codes (as strings) to exclude.
    counties_to_exclude (list): List of county FIPS codes (as strings) to exclude.

    Returns:
    pd.DataFrame: Filtered DataFrame with rows from specified states and counties removed from the specified columns.
    """

    # Loop through each column specified in cols_to_filter
    for col in cols_to_filter:
        # Filter out rows where the state FIPS code (first two characters of the column value) is in states_to_exclude
        county_df = county_df[county_df[col].apply(lambda x: x[:2] not in states_to_exclude)]

        # Further filter out rows where the full county FIPS code is in counties_to_exclude
        county_df = county_df[~county_df[col].isin(counties_to_exclude)]

    return county_df


def interpolate_node_features(county_df: pd.DataFrame, adjacency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates missing values in the county DataFrame using the mean values of adjacent counties.

    Parameters:
    county_df (pd.DataFrame): DataFrame containing county data with missing values and a 'fips' column.
    adjacency_df (pd.DataFrame): DataFrame containing county adjacency information with 'County GEOID' and
                                 'Neighbor GEOID' columns.

    Returns:
    pd.DataFrame: DataFrame with interpolated missing values based on adjacent counties.
    """

    # Initialize iteration counter
    i = 0

    # Continue interpolation until there are no more NaN values or the maximum number of iterations is reached
    while county_df.isna().any().any() and i <= 5:
        for index, row in county_df.iterrows():
            for column in county_df.columns:
                # Check if the current cell has a missing value
                if pd.isna(row[column]):
                    # Get the FIPS codes of adjacent counties
                    adjacent_counties = adjacency_df[adjacency_df["County GEOID"] == row["fips"]]["Neighbor GEOID"]

                    # Calculate the mean value of the adjacent counties for the current column
                    interpolation_value = np.mean(county_df[county_df["fips"].isin(adjacent_counties)][column])

                    # If the interpolation value is not NaN, fill the missing value with it
                    if not np.isnan(interpolation_value):
                        county_df.loc[index, column] = interpolation_value

        # Increment the iteration counter
        i += 1

    print(f"Interpolation stopped after {i} iterations.")
    if i < 5:
        print(f"All missing values were interpolated.")
    else:
        print(f"Check for further missing values.")
    return county_df


def transform_node_features(county_df: pd.DataFrame, features_to_transform: list) -> pd.DataFrame:
    """
    Transforms specified features in the county DataFrame using Power Transformation and Min-Max Scaling.

    Parameters:
    county_df (pd.DataFrame): DataFrame containing county data with features to be transformed.
    features_to_transform (list): List of feature (column) names to be transformed.

    Returns:
    pd.DataFrame: DataFrame with transformed features.
    """

    # Initialize the PowerTransformer with the 'yeo-johnson' method, without standardization and copying
    pt = PowerTransformer(method='yeo-johnson', standardize=False, copy=False)

    # Initialize the MinMaxScaler without copying
    scaler = MinMaxScaler(copy=False)

    # Apply the PowerTransformer to the specified features
    county_df[features_to_transform] = pt.fit_transform(county_df[features_to_transform])

    # Apply the MinMaxScaler to the same features
    county_df[features_to_transform] = scaler.fit_transform(county_df[features_to_transform])

    return county_df


def load_adjacency_data(adjacency_data_path: str) -> pd.DataFrame:
    # Load adjacency data from a CSV file, using '|' as the delimiter.
    # Only load the "County GEOID" and "Neighbor GEOID" columns, treating them as strings (to preserve leading zeros).
    adjacency_df = pd.read_csv(adjacency_data_path, sep="|", usecols=["County GEOID", "Neighbor GEOID"], dtype=str)

    # Set the index of the DataFrame to be the combination of "County GEOID" and "Neighbor GEOID"
    # to facilitate fast lookups and merging.
    adjacency_df.set_index(keys=["County GEOID", "Neighbor GEOID"], inplace=True)

    # Create a new column 'is_adjacent' and set its value to 1 for all rows, indicating adjacency.
    adjacency_df['is_adjacent'] = 1

    return adjacency_df


def load_population_flow_data(people_flows_path: str, county_pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads and processes people flow data and merges it with county population data.

    Parameters:
    people_flows_path (str): File path to the CSV containing people flow data with 'geoid_o', 'geoid_d',
                             and 'pop_flows' columns.
    county_pop_df (pd.DataFrame): DataFrame containing county population data with 'FIPS' and population columns.

    Returns:
    pd.DataFrame: DataFrame containing merged data with 'origin', 'destination', 'people_in_flow',
                  and 'origin_population' columns.
    """

    # Load the people flows data from a CSV file, ensuring that geoid columns are read as strings
    people_flows_df = pd.read_csv(people_flows_path, dtype={"geoid_o": str, "geoid_d": str})

    # Drop the "Name" column from the county population DataFrame as it is not needed for further calculations
    county_pop_df.drop(labels="Name", axis=1, inplace=True)

    # Set the "FIPS" column as the index for easy merging with the people flows data later
    county_pop_df.set_index(keys="FIPS", inplace=True)

    # Drop the "date_range" column from the people flows DataFrame since it is not needed for the calculations
    people_flows_df.drop(labels="date_range", axis=1, inplace=True)

    # Group the people flows data by origin (geoid_o) and destination (geoid_d)
    # and compute the median flow for each pair
    people_flows_df = people_flows_df.groupby(["geoid_o", "geoid_d"]).median().reset_index()

    # Filter out any flow records where the number of people flows is zero or negative
    people_flows_df = people_flows_df[people_flows_df["pop_flows"] > 0]

    # Merge the median people flows data with the county population data using the origin county FIPS code
    people_flows_df = pd.merge(people_flows_df, county_pop_df, how="left", left_on="geoid_o", right_index=True)

    # Rename columns for clarity: 'geoid_o' becomes 'origin', 'geoid_d' becomes 'destination',
    # 'pop_flows' becomes 'people_in_flow', and population data is labeled 'origin_population'
    people_flows_df.rename(columns={"geoid_o": "origin",
                                    "geoid_d": "destination",
                                    "pop_flows": "people_in_flow",
                                    "# Total Population, 2023 [Estimated]": "origin_population"}, inplace=True)

    # Return the processed DataFrame with merged people flows and population data
    return people_flows_df


def load_migration_flow_data(migration_flows_path: str) -> pd.DataFrame:
    """
    Loads and processes migration flow data from an Excel file.

    Parameters:
    migration_flows_path (str): The file path to the Excel file containing migration flows data.

    Returns:
    pd.DataFrame: A DataFrame containing the processed migration data with 'origin', 'destination',
                  'people_in_flow', and 'origin_population' columns.
    """

    # Load the Excel file and retrieve all the sheet names
    file = pd.ExcelFile(migration_flows_path)
    sheet_names = file.sheet_names  # List all sheet names in the Excel file

    # Define the column names for the DataFrame that will store the processed migration data
    col_names = ["dest_state", "dest_county", "origin_state", "origin_county", "origin_population", "people_in_flow"]

    # Initialize an empty DataFrame with the defined column names
    migration_flows_df = pd.DataFrame(columns=col_names)

    # Loop through each sheet in the Excel file to process and concatenate the data
    for sheet_name in sheet_names:
        # Read the current sheet into a DataFrame, skipping the first row
        df = pd.read_excel(migration_flows_path, sheet_name=sheet_name, dtype=str, skiprows=1)

        # Select specific columns by their names from the current sheet
        df = df.loc[:, ["Current Residence State Code",
                        "Current Residence FIPS County Code",
                        "Residence 1 Year Ago State/U.S. Island Area/Foreign Region Code",
                        "Residence 1 Year Ago FIPS County Code",
                        "County of Residence 1 Year Ago.1",
                        "Movers in County-to-County Flow"]]

        # Rename the selected columns to the predefined column names
        df.columns = col_names

        # Drop any rows that contain missing values
        df.dropna(axis=0, how='any', inplace=True)

        # Concatenate the processed data from the current sheet with the main DataFrame
        migration_flows_df = pd.concat([migration_flows_df, df], axis=0, ignore_index=True)

    # Create the 'origin' and 'destination' columns by concatenating state and county codes
    migration_flows_df["origin"] = (migration_flows_df["origin_state"].apply(lambda x: x[1:])
                                    + migration_flows_df["origin_county"])
    migration_flows_df["destination"] = (migration_flows_df["dest_state"].apply(lambda x: x[1:])
                                         + migration_flows_df["dest_county"])

    # Select only relevant columns for further analysis and convert the data types
    migration_flows_df = migration_flows_df[["origin", "destination", "people_in_flow", "origin_population"]]
    migration_flows_df[["people_in_flow", "origin_population"]] = (
        migration_flows_df[["people_in_flow", "origin_population"]].astype(int))

    return migration_flows_df


def load_commuting_flow_data(commuting_flows_path: str, commuting_flows_old_path: str,
                             county_pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads and processes commuting flow data from two Excel files, and merges it with county population data.

    Parameters:
    commuting_flows_path (str): The file path to the current commuting flows Excel file.
    commuting_flows_old_path (str): The file path to the old commuting flows Excel file.
    county_pop_df (pd.DataFrame): DataFrame containing county population data with 'FIPS' and population columns.

    Returns:
    pd.DataFrame: A DataFrame containing processed commuting data with origin, destination, people_in_flow, and
                  origin_population columns.
    """

    # Load the current commuting flows data from the Excel file, specifying column types for FIPS codes
    commuting_flows_df = pd.read_excel(commuting_flows_path, skiprows=7, dtype={"State FIPS Code": str,
                                                                                "County FIPS Code": str,
                                                                                "State FIPS Code.1": str,
                                                                                "County FIPS Code.1": str})

    # Drop rows with any missing values from the current commuting flows data
    commuting_flows_df.dropna(axis=0, how='any', inplace=True)

    # Filter out rows where the origin or destination state FIPS code is "09" (Connecticut) due to newly defined MSAs
    commuting_flows_df = commuting_flows_df[(commuting_flows_df["State FIPS Code"] != "09") &
                                            (commuting_flows_df["State FIPS Code.1"] != "009")]

    # Load the old commuting flows data from the Excel file, specifying column types for FIPS codes
    commutes_df_old = pd.read_excel(commuting_flows_old_path, skiprows=6, dtype={"State FIPS Code": str,
                                                                                 "County FIPS Code": str,
                                                                                 "State FIPS Code.1": str,
                                                                                 "County FIPS Code.1": str})

    # Drop rows with any missing values from the old commuting flows data
    commutes_df_old.dropna(axis=0, how='any', inplace=True)

    # Filter for rows where the origin or destination state FIPS code is "09" (Connecticut) in the old data
    commutes_df_old = commutes_df_old[(commutes_df_old["State FIPS Code"] == "09") |
                                      (commutes_df_old["State FIPS Code.1"] == "009")]

    # Concatenate the current and old commuting flows data into a single DataFrame
    commuting_flows_df = pd.concat([commuting_flows_df, commutes_df_old], axis=0)

    # Create the 'origin' and 'destination' columns by combining state and county FIPS codes
    commuting_flows_df["origin"] = commuting_flows_df["State FIPS Code"] + commuting_flows_df["County FIPS Code"]
    commuting_flows_df["destination"] = (commuting_flows_df["State FIPS Code.1"].apply(lambda x: x[1:])
                                         + commuting_flows_df["County FIPS Code.1"])

    # Merge the commuting flows data with the county population data based on the 'origin' FIPS code
    commuting_flows_df = pd.merge(commuting_flows_df, county_pop_df, left_on="origin", right_on="FIPS", how="left")

    # Rename columns for clarity
    commuting_flows_df.rename(columns={"Workers in Commuting Flow": "people_in_flow",
                                       "# Total Population, 2023 [Estimated]": "origin_population"}, inplace=True)

    # Select and reorder the relevant columns for the final DataFrame
    commuting_flows_df = commuting_flows_df[["origin", "destination", "people_in_flow", "origin_population"]]

    # Return the final processed DataFrame
    return commuting_flows_df


def compute_scaled_index(index_name: str, processed_flow_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a scaled index for flow data based on the ratio of people in flow to origin population.

    Parameters:
    index_name (str): The name to assign to the computed index column.
    processed_flow_df (pd.DataFrame): A DataFrame containing the flow data with 'origin' and 'destination' fips codes,
                                      'people_in_flow' and 'origin_population' columns.

    Returns:
    pd.DataFrame: A DataFrame with the scaled index column and original flow-related columns dropped.
    """

    # Compute the index as the ratio of people in flow to origin population
    processed_flow_df[index_name] = processed_flow_df["people_in_flow"] / processed_flow_df["origin_population"]

    # Compute the scaled index by dividing the index by its maximum value to scale it between 0 and 1
    processed_flow_df[f"scaled_{index_name}"] = processed_flow_df[index_name] / np.max(processed_flow_df[index_name])

    # Drop the original flow-related columns and the unscaled index column as they are no longer needed
    processed_flow_df.drop(columns=["people_in_flow", "origin_population", index_name], inplace=True)

    # Return the DataFrame with only the scaled index column
    return processed_flow_df


def load_social_connectedness_index_data(social_connectedness_data_path: str) -> pd.DataFrame:
    """
    Loads and processes Social Connectedness Index (SCI) data from a tab-delimited text file.

    Parameters:
    social_connectedness_data_path (str): The file path to the tab-delimited text file containing SCI data.

    Returns:
    pd.DataFrame: A DataFrame containing the processed SCI data with 'origin', 'destination', and scaled SCI columns.
    """

    # Load the SCI data from a tab-delimited file, ensuring 'user_loc' and 'fr_loc' columns are treated as strings
    sci_df = pd.read_csv(social_connectedness_data_path, sep='\t', dtype={"user_loc": str, "fr_loc": str})

    # Scale the SCI values by dividing them by the maximum SCI value to normalize them between 0 and 1
    sci_df["scaled_sci"] = sci_df["scaled_sci"] / np.max(sci_df["scaled_sci"])

    # Rename the 'user_loc' and 'fr_loc' columns to 'origin' and 'destination' for consistency with other dataframes
    sci_df.rename(columns={"user_loc": "origin", "fr_loc": "destination"}, inplace=True)

    # Return the processed DataFrame with 'origin', 'destination', and scaled SCI columns
    return sci_df

