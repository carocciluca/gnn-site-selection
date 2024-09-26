import pandas as pd


def extract_weekly_people_flows(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extracts weekly people flow data from a given start date to an end date.

    Parameters:
    start_date (str): The start date for data extraction in 'YYYY-MM-DD' format.
    end_date (str): The end date for data extraction in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the concatenated weekly people flow data within the specified date range.
    """

    # Convert start and end dates from string to datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Initialize an empty DataFrame to store the concatenated results
    df_people_flows = pd.DataFrame()

    # Initialize a counter to track the number of successfully extracted weekly flows
    i = 0
    current_date = start_date

    # Loop through each day from start_date to end_date
    while current_date <= end_date:
        # Extract the year, month, and day from the current date
        start_year = current_date.year
        start_month = "0" + str(current_date.month) if len(str(current_date.month)) == 1 else current_date.month
        start_day = "0" + str(current_date.day) if len(str(current_date.day)) == 1 else current_date.day

        # Construct the URL for the CSV file based on the current date
        try:
            link = f"https://raw.githubusercontent.com/GeoDS/COVID19USFlows-WeeklyFlows/master/weekly_flows/county2county/weekly_county2county_{start_year}_{start_month}_{start_day}.csv"

            # Read the CSV file, selecting specific columns and setting data types
            df = pd.read_csv(link,
                             usecols=["geoid_o", "geoid_d", "date_range", "pop_flows"],
                             dtype={"geoid_o": str, "geoid_d": str})

            # Concatenate the new DataFrame with the existing data
            df_people_flows = pd.concat([df_people_flows, df])

            # Increment the counter for successfully extracted weekly flows
            i += 1
            print(f"Extracted Week {i}. File size: {len(df_people_flows)}")

        except:
            pass

        # Move to the next day
        current_date += pd.DateOffset(days=1)

    # Print a completion message with the number of successfully extracted weekly flows
    print(f"Extraction complete! {i} weekly flows between {start_date} and {end_date} were extracted.")

    df_people_flows = df_people_flows.pivot(index=["geoid_o", "geoid_d"], columns="date_range",
                                            values="pop_flows").reset_index()
    df_people_flows.fillna(value=0, inplace=True)
    df_people_flows = pd.melt(df_people_flows, id_vars=["geoid_o", "geoid_d"], value_name='pop_flows')

    print(f"Final file rows: {len(df_people_flows)}.")
    return df_people_flows
