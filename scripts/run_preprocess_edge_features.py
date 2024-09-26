import pandas as pd
from scripts.config import (POPULATION_DATA_PATH, ADJACENCY_DATA_PATH, SOCIAL_CONNECTEDNESS_DATA_PATH,
                            COMMUTES_DATA_PATH, COMMUTES_OLD_DATA_PATH, MIGRATION_DATA_PATH, POPULATION_FLOWS_DATA_PATH,
                            STATES_TO_EXCLUDE, COUNTIES_TO_EXCLUDE, EDGES_PROCESSED_DATA_PATH)
from data_preprocessing import (filter_rows, load_adjacency_data, load_social_connectedness_index_data,
                                load_commuting_flow_data, load_migration_flow_data, load_population_flow_data,
                                compute_scaled_index)


def main():

    # Load county population data from a CSV file.
    # The 'FIPS' column is loaded as a string to preserve leading zeros (important for FIPS codes).
    county_pop_df = pd.read_csv(POPULATION_DATA_PATH, dtype={"FIPS": str})

    #################################
    # 1. Social Connectedness
    #################################

    # Load the Social Connectedness Index (SCI) data from a tab-delimited text file.
    sci_df_unfiltered = load_social_connectedness_index_data(
        social_connectedness_data_path=SOCIAL_CONNECTEDNESS_DATA_PATH)

    # Filter out outlying areas and counties with population less than 300 to reduce the impact of outliers.
    sci_df = filter_rows(county_df=sci_df_unfiltered, cols_to_filter=["origin", "destination"],
                         states_to_exclude=STATES_TO_EXCLUDE, counties_to_exclude=COUNTIES_TO_EXCLUDE)

    #################################
    # 2. Mobility Connectedness
    #################################

    # Load and process commuting flow data.
    # Includes merging with population data and replacing Connecticut data with an older dataset due to discrepancies.
    commuting_flows_df = load_commuting_flow_data(commuting_flows_path=COMMUTES_DATA_PATH,
                                                  commuting_flows_old_path=COMMUTES_OLD_DATA_PATH,
                                                  county_pop_df=county_pop_df)

    # Filter out outlying areas and counties with population less than 300 to reduce the impact of outliers.
    commuting_flows_df = filter_rows(county_df=commuting_flows_df, cols_to_filter=["origin", "destination"],
                                     states_to_exclude=STATES_TO_EXCLUDE, counties_to_exclude=COUNTIES_TO_EXCLUDE)

    # Compute the Commuting Connectedness Index (CCI) by scaling the data.
    mci_df = compute_scaled_index(index_name="mci", processed_flow_df=commuting_flows_df)

    #################################
    # 3. Migration Connectedness
    #################################

    # Load and process migration flow data. The data already includes population figures.
    migration_flows_df = load_migration_flow_data(migration_flows_path=MIGRATION_DATA_PATH)

    # Filter out outlying areas and counties with population less than 300 to reduce the impact of outliers.
    migration_flows_df = filter_rows(county_df=migration_flows_df, cols_to_filter=["origin", "destination"],
                                     states_to_exclude=STATES_TO_EXCLUDE, counties_to_exclude=COUNTIES_TO_EXCLUDE)

    # Compute the Migration Connectedness Index (MigCI) by scaling the data.
    migci_df = compute_scaled_index(index_name="migci", processed_flow_df=migration_flows_df)

    #################################
    # 4. Population Connectedness
    #################################

    # Load and process population flow data, merging it with population data for further computations.
    population_flows_df = load_population_flow_data(people_flows_path=POPULATION_FLOWS_DATA_PATH,
                                                    county_pop_df=county_pop_df)

    # Filter out outlying areas and counties with population less than 300 to reduce the impact of outliers.
    population_flows_df = filter_rows(county_df=population_flows_df, cols_to_filter=["origin", "destination"],
                                      states_to_exclude=STATES_TO_EXCLUDE, counties_to_exclude=COUNTIES_TO_EXCLUDE)

    # Compute the Population Connectedness Index (PCI) by scaling the data.
    pci_df = compute_scaled_index(index_name="pci", processed_flow_df=population_flows_df)

    #################################
    # Combine All Index DataFrames
    #################################

    # Load adjacency data, which identifies which counties are geographically adjacent to each other.
    adjacency_df = load_adjacency_data(adjacency_data_path=ADJACENCY_DATA_PATH)

    # Create a list of the computed index DataFrames (SCI, CCI, MCI, PCI).
    index_dfs = [sci_df, mci_df, migci_df, pci_df]

    # Set the index of each DataFrame to ['origin', 'destination'] to facilitate merging.
    for index_df in index_dfs:
        index_df.set_index(keys=["origin", "destination"], inplace=True)

    # Concatenate all index DataFrames and adjacency data along the columns, keeping all data (outer join).
    edge_feature_df = pd.concat(index_dfs + [adjacency_df], join="outer", axis=1)
    edge_feature_df["is_adjacent"].fillna(value=0, axis=0, inplace=True)

    # Reset the index to turn 'origin' and 'destination' back into regular columns.
    edge_feature_df.reset_index(inplace=True, names=["origin", "destination"])

    # Save the final combined DataFrame with computed indices to a CSV file at the specified path.
    edge_feature_df.to_csv(EDGES_PROCESSED_DATA_PATH, index=False)
    print(f"Data saved to {EDGES_PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
