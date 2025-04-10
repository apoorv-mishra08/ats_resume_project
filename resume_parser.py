import pdfplumber
import os
import pandas as pd
import random

def extract_sections(text):
    sections = {
        "skills": "",
        "education": "",
        "experience": "",
        "projects": "",
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

def parse_resumes(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'

            sections = extract_sections(text)

            ats_score = random.randint(85, 100)

            data.append({
                "resume_text": text,
                "skills": sections["skills"],
                "education": sections["education"],
                "experience": sections["experience"],
                "projects": sections["projects"],
                "ats_score": ats_score
            })

    df = pd.DataFrame(data)
    df.to_csv("structured_resume_dataset.csv", index=False)
    print("Structured resume dataset with random ATS scores saved successfully.")

if __name__ == "__main__":
    folder_path = r"C:\Users\apoor\OneDrive\Desktop\ats_resume_project\data\princeton_resumes"
    parse_resumes(folder_path)
