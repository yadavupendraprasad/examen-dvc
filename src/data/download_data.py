import requests
import os

def main():
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    output_dir = "data/raw_data"
    output_path = os.path.join(output_dir, "raw.csv")

    os.makedirs(output_dir, exist_ok=True)  # Make sure the folder exists

    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raise error if download fails

    with open(output_path, "wb") as f:
        f.write(response.content)  # Save content to file

    print(f"Data saved to {output_path}")

# 👇 This block should NOT be indented
if __name__ == "__main__":
    main()
