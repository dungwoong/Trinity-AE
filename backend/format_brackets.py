import sys
import os

def format_code_string(content, indent_size=2):
    formatted_code = ""
    indent_level = 0
    
    i = 0
    while i < len(content):
        char = content[i]
        
        if char == "(":
            # Move to next line, indent, and place the bracket
            formatted_code += "\n" + (" " * (indent_level * indent_size)) + char
            indent_level += 1
        elif char == ")":
            indent_level -= 1
            # Close on a new line at the reduced indent level
            formatted_code += "\n" + (" " * (indent_level * indent_size)) + char
        else:
            # If we just started a new line, apply indentation before the text
            if formatted_code.endswith('\n'):
                formatted_code += (" " * (indent_level * indent_size))
            
            # Avoid excessive newlines if the input has them already
            if char != "\n":
                formatted_code += char
            
        i += 1

    # Clean up leading/trailing empty lines
    return "\n".join([line for line in formatted_code.splitlines() if line.strip()])

def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_file>")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with open(file_path, 'r') as f:
            raw_content = f.read()
        
        print(format_code_string(raw_content))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()