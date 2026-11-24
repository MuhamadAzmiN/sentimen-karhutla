import pymupdf

doc = pymupdf.open("komentar_karhutla_200.pdf")
print("Total pages:", len(doc))

for pno in range(len(doc)):
    page = doc[pno]
    print(f"\n=== PAGE {pno+1} ===")
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            line_spans = []
            for span in line.get("spans", []):
                line_spans.append((span["bbox"], span["text"]))
            print(line_spans)
        if len(block.get("lines", [])) > 0:
            print("---")
    if pno == 0:
        # only print first page for now
        break
