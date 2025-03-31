import os
import pandas as pd

def create_dataset_from_txt(folder_path, output_csv):
    """
    Reads all .txt resumes, assigns score between 90-100, and saves to CSV.
    """
    data = []
    score = 90

    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    data.append({
                        "filename": filename,
                        "resume_text": text,
                        "ats_score": score
                    })
                    score = 90 + (idx % 11)  # Cycle scores from 90–100

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset saved as {output_csv} with {len(df)} resumes.")

if __name__ == "__main__":
    input_folder = "parsed/princeton_texts"
    output_csv = "princeton_resume_dataset.csv"
    create_dataset_from_txt(input_folder, output_csv)
