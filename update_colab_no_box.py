import json

with open('komentar_karhutla_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = ''.join(nb['cells'][0]['source'])

# Replace display(data.head()) with print(data.head())
source = source.replace("display(data.head())", "print(data.head())")
# Replace display(data[["komentar", "komentar_bersih", "sentimen"]].head()) with print(...)
source = source.replace('display(data[["komentar", "komentar_bersih", "sentimen"]].head())', 'print(data[["komentar", "komentar_bersih", "sentimen"]].head())')

nb['cells'][0]['source'] = [line + '\n' for line in source.split('\n') if line != '']

with open('komentar_karhutla_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated komentar_karhutla_colab.ipynb: replaced display() with print() to remove table boxes!")
