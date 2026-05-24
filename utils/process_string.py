

def strip_markdown_code_fences(string_to_process):
    """
    Remove markdown code fence markers (```json and ```) from a string.
    
    Args:
        string_to_process: String potentially wrapped in markdown code fences
        
    Returns:
        Cleaned string with code fences removed
    """
    # Clean the string by removing markdown code fences
    cleaned_str = string_to_process.strip()
    
    # Remove ```json or ``` at the start
    if cleaned_str.startswith("```json"):
        cleaned_str = cleaned_str[7:]  # Remove ```json
    elif cleaned_str.startswith("```"):
        cleaned_str = cleaned_str[3:]  # Remove ```
    
    # Remove ``` at the end
    if cleaned_str.endswith("```"):
        cleaned_str = cleaned_str[:-3]
    
    # Strip again after removing code blocks
    cleaned_str = cleaned_str.strip()
    
    return cleaned_str