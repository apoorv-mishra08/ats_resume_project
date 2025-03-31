from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def improve_resume_text(resume_text, max_length=512):
    """
    Improves a resume text using a T5 transformer model.
    """
    prompt = "improve this resume: " + resume_text.strip()
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    improved_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return improved_text

# Test Block
if __name__ == "__main__":
    sample = "my name is Apoorv. i work in software and know python. i want to grow in my career."
    improved = improve_resume_text(sample)
    print("\n🔧 Original:\n", sample)
    print("\n🚀 Improved Version:\n", improved)
