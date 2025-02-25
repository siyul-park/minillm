from datasets import load_dataset
import os

def download_and_process_data():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["train"]["text"]))

    with open(os.path.join(data_dir, "valid.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["validation"]["text"]))

    with open(os.path.join(data_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["test"]["text"]))

if __name__ == "__main__":
    download_and_process_data()
