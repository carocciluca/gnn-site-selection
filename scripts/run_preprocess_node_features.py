import pandas as pd
from scripts.config import (COUNTY_DATA_PATH, ADJACENCY_DATA_PATH, COUNTY_PROCESSED_DATA_PATH,
                            STATES_TO_EXCLUDE, COUNTIES_TO_EXCLUDE)
from data_preprocessing import filter_rows, interpolate_node_features, transform_node_features


def main():
    # Read raw demographic county features for pre-processing
    county_df = pd.read_csv(COUNTY_DATA_PATH, encoding="ISO-8859-1", dtype={"fips": str, "google_trends_compX": float})

    # Convert number of dealerships into boolean values solely indicating dealer presence (=1) or no presence (=0)
    features_to_convert = ["has_compA_dealer", "has_compB_dealer", "has_compX_dealer"]

    for feature in features_to_convert:
        county_df[feature] = county_df[feature].apply(lambda x: 1 if x > 0 else 0)

    # Read county adjacency information for interpolation
    adjacency_df = pd.read_csv(ADJACENCY_DATA_PATH, sep="|", dtype={"County GEOID": str, "Neighbor GEOID": str})

    # Remove non-contiguous states, outlying areas, and counties with population <300
    filtered_county_df = filter_rows(county_df=county_df, cols_to_filter=["fips"], states_to_exclude=STATES_TO_EXCLUDE,
                                     counties_to_exclude=COUNTIES_TO_EXCLUDE)

    # Interpolate missing values using averages of adjacent counties
    interpolated_county_df = interpolate_node_features(county_df=filtered_county_df, adjacency_df=adjacency_df)

    # Transform and scale features between 0 and 1
    features_to_transform = [feature for feature in county_df.columns if county_df[feature].dtype == "float64"]

    transformed_county_df = transform_node_features(county_df=interpolated_county_df,
                                                    features_to_transform=features_to_transform)

    # Store preprocessed node features
    transformed_county_df.to_csv(COUNTY_PROCESSED_DATA_PATH, index=False)

    print(f"Data saved to {COUNTY_PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
