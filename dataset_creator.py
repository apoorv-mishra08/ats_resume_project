import os
import pandas as pd
import random

def extract_sections(text):
    sections = {
        "skills": "",
        "education": "",
        "experience": "",
        "projects": ""
    }

    lines = text.lower().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if "skill" in line:
            current_section = "skills"
        elif "education" in line:
            current_section = "education"
        elif "experience" in line:
            current_section = "experience"
        elif "project" in line:
            current_section = "projects"
        elif current_section:
            sections[current_section] += line + " "

    return sections

def create_dataset_from_txt(folder_path, output_csv):
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        existing_files = set(existing_df["filename"])
    else:
        existing_df = pd.DataFrame()
        existing_files = set()

    data = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt") and filename not in existing_files:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if text:
                sections = extract_sections(text)
                ats_score = random.randint(85, 100)

                data.append({
                    "filename": filename,
                    "resume_text": text,
                    "skills": sections["skills"],
                    "education": sections["education"],
                    "experience": sections["experience"],
                    "projects": sections["projects"],
                    "ats_score": ats_score
                })

    new_df = pd.DataFrame(data)

    if not new_df.empty:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"Appended {len(new_df)} new .txt resumes to {output_csv}.")
    else:
        print("No new resumes to add.")

if __name__ == "__main__":
    input_folder = "parsed/princeton_texts"
    output_csv = "princeton_resume_dataset.csv"
    create_dataset_from_txt(input_folder, output_csv)
