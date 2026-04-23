# Yuyao Xu
# Apr 2026
# Removes widget metadata from the Colab notebook to fix GitHub rendering.


import json

notebook_path = 'wikiart_colab.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

if 'widgets' in nb['metadata']:
    del nb['metadata']['widgets']
    print("Widget metadata removed.")
else:
    print("No widget metadata found.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Done:", notebook_path)