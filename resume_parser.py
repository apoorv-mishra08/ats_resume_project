import os
import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_all_pdfs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(output_folder, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved: {txt_path}")

if __name__ == "__main__":
    input_folder = "data/princeton_resumes"
    output_folder = "parsed/princeton_texts"
    parse_all_pdfs(input_folder, output_folder)
