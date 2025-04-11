import nbformat

def extract_code_from_ipynb(ipynb_file, output_file):
    with open(ipynb_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)
        
    code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, code in enumerate(code_cells, 1):
            file.write(code)
            file.write('\n\n')

# Replace 'notebook.ipynb' and 'output.py' with your file names
extract_code_from_ipynb('main.ipynb', 'output.py')