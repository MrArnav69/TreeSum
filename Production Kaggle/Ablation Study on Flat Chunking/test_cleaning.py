
import re

def clean_document(text: str) -> str:
    # Remove image metadata (common in Multi-News)
    text = re.sub(r'Enlarge this image.*?AP', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'toggle caption.*?AP', '', text, flags=re.IGNORECASE)
    
    # Remove bracketed annotations
    text = re.sub(r'\[.*?\]', '', text)
    
    # Normalize whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()

test_text = """GOP Eyes Gains As Voters In 11 States Pick Governors 
 
 Enlarge this image toggle caption Jim Cole/AP Jim Cole/AP 
 
 Voters in 11 states will pick their governors tonight, and Republicans appear on track to increase their numbers by at least one, with the potential to extend their hold to more than two-thirds of the nation's top state offices."""

cleaned = clean_document(test_text)
print(f"Original len: {len(test_text)}")
print(f"Cleaned len: {len(cleaned)}")
print(f"Cleaned output: [{cleaned}]")
