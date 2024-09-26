from scripts import POPULATION_FLOWS_DATA_PATH
from data_collection import extract_weekly_people_flows


def main():
    # Extract weekly people flows
    population_flows_df = extract_weekly_people_flows(start_date="2019-01-01", end_date="2019-12-31")

    # Save the extracted data
    population_flows_df.to_csv(POPULATION_FLOWS_DATA_PATH, index=False)
    print(f"Data saved to {POPULATION_FLOWS_DATA_PATH}")


if __name__ == "__main__":
    main()
