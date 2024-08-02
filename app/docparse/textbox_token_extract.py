
def textbox_token_extract(docpath):
    import fitz
    import json
    import pandas as pd

    doc = fitz.open(docpath)
    data = []
    for page in doc:
        page_id = page.number + 1

        blocks = page.get_text("words") # get_text("dict")["blocks"]
        data.extend([{'text':i[4],
                      'x0': i[0],
                      'y0': i[1],
                      'x1': i[2],
                      'y1': i[3],
                      'blocknumber':i[5], 
                      'linenumber':i[6], 
                      'wordnumber':i[7],
                      'page_id':page_id} for i in blocks])

        # for b in blocks:  # iterate through the text blocks
        #     if b['type'] == 0:  # this block contains text
        #         # REMEMBER: multiple fonts and sizes are possible IN one block
        #         block_string = ""  # text found in block
        #         for l in b["lines"]:  # iterate through the text lines
        #             for s in l["spans"]:  # iterate through the text spans
        #                 if s['text'].strip():  # removing whitespaces:
        #                     text = s['text']
        #                     bbox = s['bbox']
        #                     dic[page_id].append({'text':text, 'bbox': bbox})

    # print(json.dumps(data,indent=4))
    df = pd.DataFrame(data)
    df.to_csv('token_layout.csv')

def get_text_by_bbox(docpath, page_id, bbox):
    import fitz
    import json
    import pandas as pd
    
    doc = fitz.open(docpath)
    page = doc.load_page(page_id-1) #put here the page number
    text = page.get_textbox(bbox)
    
    return text