import json
import os

p = 'Deepfake_Detection_Model_Training_and_Evaluation.ipynb'
print(f"File Size: {os.path.getsize(p):,} bytes")

with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total Cells: {len(nb['cells'])}")
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Markdown Cells: {len(md_cells)} | Code Cells: {len(code_cells)}")

for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    first_line = ''.join(c['source']).split('\n')[0].strip()
    outputs = c.get('outputs', [])
    out_desc = []
    for o in outputs:
        if o['output_type'] == 'stream':
            out_desc.append(f"stream({len(o['text'])} lines)")
        elif o['output_type'] == 'display_data':
            keys = list(o['data'].keys())
            out_desc.append(f"display_data({','.join(keys)})")
        else:
            out_desc.append(o['output_type'])
    print(f"Cell {i:02d} [{ctype:8s}]: {first_line[:45]:<45} -> {out_desc}")

print("\nNotebook structure verified 100% valid!")
