import pymupdf
import re
import pandas as pd

def build_cmap(doc):
    cmap_bytes = doc.xref_stream(34).decode('latin1')
    char_map = {}
    
    bfchar_matches = re.findall(r'(\d+)\s+beginbfchar\s*(.*?)\s*endbfchar', cmap_bytes, re.DOTALL)
    for count, block in bfchar_matches:
        pairs = re.findall(r'<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>', block)
        for src, dst in pairs:
            char_map[int(src, 16)] = chr(int(dst, 16))
            
    bfrange_matches = re.findall(r'(\d+)\s+beginbfrange\s*(.*?)\s*endbfrange', cmap_bytes, re.DOTALL)
    for count, block in bfrange_matches:
        ranges = re.findall(r'<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>', block)
        for start, end, dst_start in ranges:
            s_code = int(start, 16)
            e_code = int(end, 16)
            d_code = int(dst_start, 16)
            for i in range(e_code - s_code + 1):
                char_map[s_code + i] = chr(d_code + i)
                
    return char_map

def decode_cid_string(raw_bytes, char_map):
    res = []
    for i in range(0, len(raw_bytes), 2):
        if i + 1 < len(raw_bytes):
            code = (raw_bytes[i] << 8) | raw_bytes[i+1]
            res.append(char_map.get(code, ''))
    return ''.join(res)

doc = pymupdf.open("komentar_karhutla_200.pdf")
char_map = build_cmap(doc)

all_strings = []
for pno in range(len(doc)):
    cont = doc[pno].read_contents()
    matches = re.findall(rb'\(((?:\\.|[^\\()])*)\)\s*(?:Tj|TJ|cm|m|l)?', cont)
    for m in matches:
        unescaped = m.replace(b'\\\\', b'\\').replace(b'\\(', b'(').replace(b'\\)', b')')
        dec = decode_cid_string(unescaped, char_map).strip()
        if dec:
            all_strings.append(dec)

# Skip the initial header 'komentar' and 'sentimen'
idx = 0
if len(all_strings) >= 2 and all_strings[0].lower() == 'komentar' and all_strings[1].lower() == 'sentimen':
    idx = 2

pairs = []
curr_comment_parts = []

while idx < len(all_strings):
    item = all_strings[idx]
    if item.lower() in ['marah', 'sedih']:
        comment_text = "".join(curr_comment_parts).strip()
        pairs.append((comment_text, item.lower()))
        curr_comment_parts = []
    else:
        curr_comment_parts.append(item)
    idx += 1

print(f"Total extracted pairs: {len(pairs)}")
df = pd.DataFrame(pairs, columns=['komentar', 'sentimen'])
df.insert(0, 'id', range(1, len(df) + 1))
df.to_csv('dataset_karhutla_200.csv', index=False, encoding='utf-8')
print("Saved to dataset_karhutla_200.csv")
print("Sentiment counts:")
print(df['sentimen'].value_counts())
print("\nFirst 10 rows:")
for i, r in df.head(10).iterrows():
    print(f"{r['id']}. [{r['sentimen']}] {r['komentar']}")
print("\nLast 10 rows:")
for i, r in df.tail(10).iterrows():
    print(f"{r['id']}. [{r['sentimen']}] {r['komentar']}")
