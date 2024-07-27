import os

def process_txt_files():
    current_path = os.getcwd()  # Get current path
    txt_files = [f for f in os.listdir(current_path) if f.endswith('.py')]  # Get all py files

    for file_name in txt_files:
        file_path = os.path.join(current_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            if line.startswith(' '):
                num_spaces = len(line) - len(line.lstrip())
                if num_spaces % 4 != 0:
                    line = line[1:]  # Remove a leading space
            modified_lines.append(line)

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        print(f"Processed file: {file_name}")

process_txt_files()