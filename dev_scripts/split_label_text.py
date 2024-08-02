# A colored annotation and content seperator with PyMUPDF PDF renderer
# Usage: provide a filepath of source annotated PDF and output CSV with headers ["page_id", "text", "label"]
# Prerequisite:
# 1. Please pip install PyMuPDF
# 2. Must contain annotation with font color
# Documentation for PyMUPDF: https://pymupdf.readthedocs.io/en/latest/index.html
import fitz
import pandas as pd
import os
import re

def flags_decomposer(flags):
    """Make font flags human readable.
    @param flags: font properties flags from PyMuPDF
    @type flags: int
    """
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)

def neighborhood(iterable):
    """ Create an iterator that yields previous, current and next item in every loop
    @param iterable: iterator
    @type iterable: iterable object
    """
    iterator = iter(iterable)
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (prev_item, current_item, next_item)
        prev_item = current_item
        current_item = next_item
    yield (prev_item, current_item, None)

def page_size(doc, page_id):
    '''Extract page weight and height with given page id
    @param doc: PyMuPDF class that represent the document
    @type doc: <class 'fitz.fitz.Document'>
    @param page_id: page id (first page id as 0)
    @type page_id: int
    @return w,h : weight, height of the page
    @rtype w,h : tuple
    '''
    w, h = int(doc[page_id].rect.width), int(doc[page_id].rect.height)
    return w, h

def normalize_bbox(bbox, width, height):
    '''Given actual bbox, page height and width, return normalized bbox coordinate (0-1000) on the page
    @param bbox: actual bbox in [x0,y0,x1,y1]
    @type bbox: list
    @param width: width of the page
    @type width: int
    @param height: height of the page
    @type height: int
    @rtype: [x0,y0,x1,y1], list
    @return: integral coordinates range from 0 to 1000 of a normalized bounding box
    '''
    word_bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return [min(1000, max(0, int(word_bbox[0] / width * 1000))),
            min(1000, max(0, int(word_bbox[1] / height * 1000))),
            min(1000, max(0, int(word_bbox[2] / width * 1000))),
            min(1000, max(0, int(word_bbox[3] / height * 1000)))]

def is_adjacent_bbox(tgt_bbox, ref_bbox, xoffset_thres=1000, yoffset_thres=1000):
    '''
    Function to check if the target bounding box is adjacent to the reference bounding box given the tolerance of x- and y-coordinate offsets
    @param tgt_bbox: target bounding box in (x0,y0,x1,y1)
    @type tgt_bbox: list
    @param ref_bbox: reference bounding box in (x0,y0,x1,y1)
    @type ref_bbox: list
    @param xoffset: x-coordinate tolerance to accept bounding box as adjacent bbox
    @type xoffset: int or float
    @param yoffset: y-coordinate tolerance to accept bounding box as adjacent bbox
    @type yoffset: int or float
    @return: boolean variable indicate if the target bbox is adjacent to reference bbox
    @rtype: bool
    '''

    x_offset, y_offset = min_offset(tgt_bbox, ref_bbox)

    if x_offset <= xoffset_thres and y_offset <= yoffset_thres:
        return True
    else:
        return False

def min_offset(tgt_bbox, ref_bbox):
    '''
    Function to obtain the min. x, y coordinate offset between two bounding boxes
    '''
    return (
        min(abs(tgt_bbox[i] - ref_bbox[j]) for i,j in ((0,2),(0,0),(2,0),(2,2))),
        min(abs(tgt_bbox[i] - ref_bbox[j]) for i,j in ((1,3),(1,1),(3,1),(3,3)))
    )

def fonts_line_spacing(doc):
    """Extracts fonts, max. line spacing and their usage in PDF documents.
    @rtype: [(font_size, count), (font_size, count)], dict
    @return: most used fonts sorted by count, font style information
    """
    from statistics import quantiles
    normalized_line_spacings = []
    for page in doc:
        w, h = page.rect.width, page.rect.height
        blocks = page.get_text("dict")["blocks"]
        blocks = sorted(blocks, key=lambda x: (x['bbox'][1], x['bbox'][0], x['bbox'][3], x['bbox'][2]))
        prev_span_bbox = None
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text span
                        normalized_bbox = normalize_bbox(s['bbox'], w, h)
                        if prev_span_bbox:
                            xoffset, yoffset = min_offset(normalized_bbox, prev_span_bbox)
                            line_spacing = yoffset
                            normalized_line_spacings.append(line_spacing)
                        prev_span_bbox = normalized_bbox
    max_line_spacing = max(normalized_line_spacings)
    if len(normalized_line_spacings) > 1:
        upper_quartile_line_spacing = quantiles(normalized_line_spacings, n=100)[84]
    return max_line_spacing, upper_quartile_line_spacing

if __name__=="__main__":
    pdf_path = '/home/data/ldrs_analytics/data/pdf/annotated_ts/7_GF_SYN_TS_mkd_antd_20221130.pdf'
    output_dir = '/home/data/ldrs_analytics/data/split_text_annotation_csv/'
    filename = os.path.basename(pdf_path).split('.')[0]
    doc = fitz.open(pdf_path)
    max_line_spacing, upper_quartile_line_spacing = fonts_line_spacing(doc)
    page_id = 0

    data = {
        "page_id": [],
        "text": [],
        "label": []
    }

    text_info = []
    for page in doc:
        page_id += 1
        blocks = page.get_text("dict")["blocks"]
        blocks = sorted(blocks, key=lambda x: (x['bbox'][1], x['bbox'][0], x['bbox'][3], x['bbox'][2]))
        w, h = page_size(doc, page_id - 1)
        for b in blocks: # iterate through the text blocks
            for l in b["lines"]:  # iterate through the text lines
                for s in l["spans"]: # iterate through the text spans
                    if s['text'].strip(): # if text not whitespace
                        text = s['text']
                        bbox = normalize_bbox(s['bbox'], w, h)
                        # if any x,y coordinate beyond visible page boundaries, discard the text
                        if any((c <= 0) or (c >= 1000) for c in bbox):
                            continue
                        d = {
                            "page_id": page_id,
                            "bbox": bbox,
                            "token": text,
                            "color": s["color"],
                            "flag": flags_decomposer(s["flags"])
                        }
                        text_info.append(d)

    bbox_cluster = []
    output = []
    iter_txt = neighborhood(text_info)
    isFirst = True
    for prev, curr, nxt in iter_txt:
        string = curr['token']
        page_id = curr['page_id']
        bbox = curr['bbox']
        color = curr['color']
        flag = curr['flag']
        if isFirst:  # if first
            s = string
            label = ''
            isFirst = False
            continue
        else:
            prev_pageid = prev['page_id']
            prev_bbox = prev['bbox']
            prev_color = prev['color']
            prev_flag = prev['flag']

        is_adjacent = is_adjacent_bbox(bbox, prev_bbox, xoffset_thres=250, yoffset_thres=upper_quartile_line_spacing)

        if (color == prev_color == 0 and flag==prev_flag) and page_id == prev_pageid and is_adjacent: # current text and previous text has same color, flag, page id and adjacent
            s += ' ' + string
        elif (color == prev_color != 0 or (flag==prev_flag and re.match('.*bold.*|.*italic.*',flag))) and page_id == prev_pageid and is_adjacent: #  or (flag==prev_flag and re.match('bold|italic',flag))
            label += ' ' + string
        elif color != prev_color and color != 0 and page_id == prev_pageid and is_adjacent:
            # append data to dictionary, then assign string as label and reset temporary content
            data["page_id"].append(page_id)
            data["text"].append(s)
            data["label"].append(label)
            s = ''
            label = string
        else:
            # append data to dictionary, then assign string as temporary content and reset label
            data["page_id"].append(page_id)
            data["text"].append(s)
            data["label"].append(label)
            s = string
            label = ''

    df = pd.DataFrame.from_dict(data)
    df['label'] = df['label'].shift(-1)
    df = df[(df['text']!='')]
    df.to_csv(f'{output_dir}{filename}_split_label.csv',index=False,encoding='utf-8-sig')