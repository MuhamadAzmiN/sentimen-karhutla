import pymupdf
import re

def build_cmap(doc):
    cmap_bytes = doc.xref_stream(34).decode('latin1')
    char_map = {}
    
    # parse bfchar
    bfchar_matches = re.findall(r'(\d+)\s+beginbfchar\s*(.*?)\s*endbfchar', cmap_bytes, re.DOTALL)
    for count, block in bfchar_matches:
        pairs = re.findall(r'<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>', block)
        for src, dst in pairs:
            src_code = int(src, 16)
            dst_char = chr(int(dst, 16))
            char_map[src_code] = dst_char
            
    # parse bfrange
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
    # raw_bytes is 2 bytes per char (Identity-H)
    res = []
    for i in range(0, len(raw_bytes), 2):
        if i + 1 < len(raw_bytes):
            code = (raw_bytes[i] << 8) | raw_bytes[i+1]
            res.append(char_map.get(code, f'[{code:04x}]'))
    return ''.join(res)

doc = pymupdf.open("komentar_karhutla_200.pdf")
char_map = build_cmap(doc)

all_items = []
for pno in range(len(doc)):
    cont = doc[pno].read_contents()
    # find TJ and Tj operators
    # in PDF: (string) Tj or [(string) num (string)] TJ
    # Let's extract all (string) matches
    # Note: escaped parentheses \( and \) can exist
    # Let's use regex matching raw ( ... )
    matches = re.findall(rb'\(((?:\\.|[^\\()])*)\)\s*(?:Tj|TJ|cm|m|l)?', cont)
    decoded_strings = []
    for m in matches:
        # unescape bytes
        unescaped = m.replace(b'\\\\', b'\\').replace(b'\\(', b'(').replace(b'\\)', b')')
        dec = decode_cid_string(unescaped, char_map)
        if dec.strip():
            decoded_strings.append(dec.strip())
    all_items.append((pno, decoded_strings))

for pno, strings in all_items:
    print(f"Page {pno+1}: {len(strings)} strings found")
    print("Sample first 6 strings:")
    for s in strings[:6]:
        print("  -", repr(s))
