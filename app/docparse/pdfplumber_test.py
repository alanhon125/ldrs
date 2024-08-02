import pdfplumber
from collections import defaultdict
from operator import itemgetter
from statistics import mode
import os
import re
import numpy as np

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
    word_bbox = (float(bbox[0]), float(bbox[1]),
                 float(bbox[2]), float(bbox[3]))
    return [min(1000, max(0, int(word_bbox[0] / width * 1000))),
            min(1000, max(0, int(word_bbox[1] / height * 1000))),
            min(1000, max(0, int(word_bbox[2] / width * 1000))),
            min(1000, max(0, int(word_bbox[3] / height * 1000)))]


def denormalize_bbox(norm_bbox, width, height):
    '''Given normalized bbox, page height and width, return actual bbox coordinate on the page
    @param norm_bbox: normalized bbox in [x0,y0,x1,y1]
    @type norm_bbox: list
    @param width: width of the page
    @type width: int
    @param height: height of the page
    @type height: int
    @return: integral coordinates of a bounding box [x0,y0,x1,y1]
    @rtype: list
    '''
    norm_bbox = (float(norm_bbox[0]), float(
        norm_bbox[1]), float(norm_bbox[2]), float(norm_bbox[3]))
    return [int(norm_bbox[0] * width / 1000),
            int(norm_bbox[1] * height / 1000),
            int(norm_bbox[2] * width / 1000),
            int(norm_bbox[3] * height / 1000)]

def discard_str_with_unwant_char(a_str):
    # To discard string with Chinese characters, Geometric shape and fullwidth ascii variant characters
    import re
    # The 4E00—9FFF range covers CJK Unified Ideographs (CJK=Chinese, Japanese and Korean)
    ChineseRegexp = re.compile(r'[\u4e00-\u9fff]+')
    geometric_shapeRegexp = re.compile(r'[\u25a0-\u25ff]+')
    # fullwidth_ascii_variantsRegexp = re.compile(r'[\uff01-\uff5e]+')

    # or fullwidth_ascii_variantsRegexp.search(a_str) or geometric_shapeRegexp.search(a_str):
    if ChineseRegexp.search(a_str):
        return None
    else:
        return a_str

def filter_invalid_char(my_str):
    # To remove invalid characters from a string
    import re
    my_str = re.sub(' +', ' ', my_str)
    my_str = re.sub('', '', my_str)
    my_str = re.sub(r' ([\.,;]) ', r'\1 ', my_str)
    my_str = re.sub(r' ([\)\]])', r'\1', my_str)
    my_str = re.sub(r' ([\(\[]) ', r' \1', my_str)
    my_str = re.sub(r'([\u4e00-\u9fff]+)', '', my_str)  # Chinese
    my_str = re.sub(r'([\u2580—\u259f]+)', '•', my_str)  # Block Elements
    my_str = re.sub(r'([\u25a0-\u25ff]+)', '•', my_str)  # geometric shape
    my_str = re.sub(r'([\ue000—\uf8ff]+)', '•', my_str)  # Private Use Area
    my_str = re.sub(r'([§]+)', '▪', my_str)
    my_str = re.sub(r'([Ø]+)','•', my_str)
    my_str = re.sub(r'([\uf06e]+)', '•', my_str)
    my_str = re.sub(r'“', r'"', my_str)
    my_str = re.sub(r'”', r'"', my_str)
    my_str = re.sub(r'’', r"'", my_str)
    # my_str = re.sub(r'([\uff01-\uff5e]+)', '', my_str) # fullwidth ascii variants
    # my_str = re.sub(r'([\u2018\u2019\u201a-\u201d]+)', '', my_str)  # Quotation marks and apostrophe
    return my_str.strip()

def add_space2camel_case(my_str):
    # add whitespace as delimiter to seperate camel case, e.g. theBorrower -> the Borrower
    import re
    import string

    punct_list = [re.escape(i) for i in string.punctuation]
    NLB_BRACK = rf'(?<!{")(?<!".join(punct_list)})'
    my_str = re.sub(rf"""
        (            # start the group
            # alternative 1
        (?<=[a-z])       # current position is preceded by a lower char
                         # (positive lookbehind: does not consume any char)
        [A-Z]            # an upper char
                         #
        |   # or
            # alternative 2
        (?<!\A)          # current position is not at the beginning of the string
                         # (negative lookbehind: does not consume any char)
        {NLB_BRACK} # ignore in case current position is succeeded by punctuation
        [A-Z]            # an upper char
        (?=[a-z])        # matches if next char is a lower char
                         # lookahead assertion: does not consume any char
        )                # end the group""",
                    r' \1', my_str)
    return my_str

def min_offset(tgt_bbox, ref_bbox):
    '''
    Function to obtain the min. x, y coordinate offset between two bounding boxes
    '''
    return (
        min(abs(tgt_bbox[i] - ref_bbox[j])
            for i, j in ((0, 2), (0, 0), (2, 0), (2, 2))),
        min(abs(tgt_bbox[i] - ref_bbox[j])
            for i, j in ((1, 3), (1, 1), (3, 1), (3, 3)))
    )

def translate_fontname(fontname):
    '''
    translate font properties into bit
    e.g. 'AAAAAE+TimesNewRomanPS-BoldMT' means 'Bold', translate it into 2**4
    '''
    splits = fontname.split('-')
    '''
    flag represents font properties except for the first bit 0. They are to be interpreted like this: 
    {
        0: 'Regular|Normal',
        2**0: 'Superscripted', # flags = 1
        2**1: 'Italic', # flags = 2
        2**2: 'Serifed', # flags = 4
        2**3: 'Monospaced', # flags = 8
        2**4: 'Bold', # flags = 16
    }
    '''
    if len(splits) > 1:
        if 'Bold' in splits[-1]:
            flag = 2 ** 4
        elif 'Monospaced' in splits[-1]:
            flag = 2 ** 3
        elif 'Serifed' in splits[-1]:
            flag = 2 ** 2
        elif 'Italic' in splits[-1]:
            flag = 2 ** 1
        elif 'Superscripted' in splits[-1]:
            flag = 2 ** 0
        else:
            flag = 0
    else:
        flag = 0

    return flag

def bboxes_cluster_outline(bboxes_list):
    '''
    Given a list of bounding boxes (x0, y0, x1, y1),
    return a rectilinear outline bounding box that includes all bounding boxes
    @param bboxes_list: a list of bounding boxes coordinates (x0, y0, x1, y1)
    @type bboxes_list: list
    @rtype: tuple
    @return: tuple of bounding box coordinate (x0, y0, x1, y1))
    '''
    all_x0, all_y0, all_x1, all_y1 = list(zip(*bboxes_list))
    return (min(all_x0), min(all_y0), max(all_x1), max(all_y1))


class DocParser(object):
    def __init__(self, pdf_inpath, img_outdir):
        self.pdf_inpath = pdf_inpath
        self.doc = pdfplumber.open(self.pdf_inpath)
        self.page_id2size = defaultdict()
        self.img_outdir = img_outdir
        self.fname = os.path.basename(os.path.splitext(pdf_inpath)[0])
        self.font_counts = {}
        self.mode_line_spacing = {}
        self.upper_percentile_line_spacing = {}
        self.max_line_spacing = {}
        self.mode_word_spacing = {}
        self.max_word_spacing = {}
        self.min_word_spacing = {}
        self.pageid_bbox2word = {}
        self.font_property_tag = {}
        self.dev_mode = False
        self.use_model = True
        self.styles = {}
        self.footer_lower_tolerance = 80
        self.footer_upper_tolerance = 1000 - 80
        self.tokens = []
        self.model_input_data = []

    def lookup_font_properties(self, word, page_id):
        '''
        word['x0']: Distance of left side of character from left side of page. = x0
        word['top']: Distance of top of character from top of page. = y0
        word['x1']: Distance of right side of character from left side of page. = x1
        word['bottom']: Distance of bottom of the character from top of page. = y1
        '''
        wx0 = word['x0']
        wy0 = word['top']
        wx1 = word['x1']
        wy1 = word['bottom']
        page = self.doc.pages[page_id-1]
        chars = page.chars
        for char in chars:
            fontname = char['fontname']
            size = round(char['size'],2)
            color = char['non_stroking_color']
            if char['object_type'] == 'char' and char['x0'] >= wx0 and char['bottom'] <= wy1 and char['top'] >= wy0 and char['x1'] <= wx1:
                return fontname, size, color

    def word_spacing2(self):

        for page in self.doc.pages:
            normalized_word_spacings = []
            page_id = page.page_number
            w = page.width
            h = page.height
            words = page.extract_words()
            prev_word_bbox = None
            for word in words:
                x0 = word['x0']
                y0 = word['top']
                x1 = word['x1']
                y1 = word['bottom']
                text = word['text']
                self.pageid_bbox2word.update({(page_id, x0, y0, x1, y1): text})
                bbox = (x0, y0, x1, y1)
                normalized_bbox = normalize_bbox(bbox, w, h)
                if prev_word_bbox:
                    xoffset, yoffset = min_offset(normalized_bbox, prev_word_bbox)
                    if yoffset <= self.max_line_spacing[page_id]:
                        word_spacing = xoffset
                        normalized_word_spacings.append(word_spacing)
                prev_word_bbox = normalized_bbox
            if len(normalized_word_spacings) > 1:
                self.mode_word_spacing[page_id] = mode(normalized_word_spacings)
                self.max_word_spacing[page_id] = max(normalized_word_spacings)
                self.min_word_spacing[page_id] = min(normalized_word_spacings)
            else:
                self.mode_word_spacing[page_id] = 0
                self.max_word_spacing[page_id] = 0
                self.min_word_spacing[page_id] = 0


    def fonts_line_spacing2(self):
        """
        Extracts fonts, max. line spacing and their usage in PDF documents.
        @rtype: [(font_size, count), (font_size, count)], dict
        @return: most used fonts sorted by count, font style information

        """
        textline = []
        for page in self.doc.pages:

            w = page.width
            h = page.height
            page_id = page.page_number
            words = page.extract_words()
            chars = page.chars

            # counting fontname occurrence in character-level
            for c in chars:
                if c and isinstance(c, dict):
                    fontname = c['fontname']
                    flag = translate_fontname(fontname)
                    size = round(c['size'],2)
                    identifier = fontname + '@' + str(size) + '^' + str(tuple(c['non_stroking_color']))
                    if identifier not in self.font_counts:
                        self.font_counts[identifier] = 1
                        self.styles[identifier] = {'size': round(c['size'],2), 'flags': flag, 'font': c['fontname'], 'color': c['non_stroking_color']}
                    else:
                        self.font_counts[identifier] += 1

            normalized_line_spacings = []
            string = ''

            # in each page, forming textline and calculate the line spacing
            for i, word in enumerate(words):
                bbox = [word['x0'], word['top'], word['x1'], word['bottom']]
                normalized_bbox = normalize_bbox(bbox, w, h)
                if i==0:
                    string += word['text']
                    prev_textline_norm_bbox = normalized_bbox
                else:
                    if abs(normalized_bbox[1] - prev_textline_norm_bbox[1])<=5:
                        string += ' ' + word['text']
                        prev_textline_norm_bbox = [prev_textline_norm_bbox[0], prev_textline_norm_bbox[1], normalized_bbox[2], normalized_bbox[3]]
                    else:
                        textline.append({'textline': string, 'line_norm_bbox': prev_textline_norm_bbox, 'page_id': page_id})
                        xoffset, yoffset = min_offset(normalized_bbox, prev_textline_norm_bbox)
                        line_spacing = yoffset
                        string = word['text']
                        prev_textline_norm_bbox = normalized_bbox
                        normalized_line_spacings.append(line_spacing)

            if len(normalized_line_spacings) > 1:
                self.mode_line_spacing[page_id] = mode(normalized_line_spacings)
                self.upper_percentile_line_spacing[page_id] = np.percentile(np.array(normalized_line_spacings), 75)
                self.max_line_spacing[page_id] = max(normalized_line_spacings)
            else:
                self.mode_line_spacing[page_id] = 0
                self.upper_percentile_line_spacing[page_id] = 0
                self.max_line_spacing[page_id] = 0
        self.font_counts = sorted(self.font_counts.items(), key=itemgetter(1), reverse=True)
        if len(self.font_counts) < 1:
            raise ValueError(f"Zero discriminating fonts found for document {self.fname}!")
        
    def font_tags2(self):
        '''
        Returns dictionary with font sizes as keys and tags as value.
        @rtype: dict
        @return: all element tags based on font-sizes
        source: https://pymupdf.readthedocs.io/en/latest/recipes-text.html?highlight=flags#how-to-analyze-font-characteristics
        flag represents font properties except for the first bit 0. They are to be interpreted like this:
        {
            0: 'Regular|Normal',
            2**0: 'Superscripted', # flags = 1
            2**1: 'Italic', # flags = 2
            2**2: 'Serifed', # flags = 4
            2**3: 'Monospaced', # flags = 8
            2**4: 'Bold', # flags = 16
        }
        '''
        p_style = self.styles[self.font_counts[0][0]]  # get style for most used font by count (paragraph)
        p_size = round(float(p_style['size']),2)  # get the paragraph's size
        p_color = p_style['color']  # get the paragraph's color
        p_font = p_style['font']  # get the paragraph's font
        p_flags = p_style['flags']  # get the paragraph's flag

        font_sizes_color = []
        for identifier, font_features in self.styles.items():
            font_size = round(float(font_features['size']),2)
            font_flag = int(font_features['flags'])
            font = font_features['font']
            font_color = font_features['color']
            if font_size == p_size and font == p_font and font_color == p_color and font_flag == p_flags:
                font_style = -1
            else:
                font_style = font_flag
            font_sizes_color.append((font_size, font, font_color, font_style))
        font_sizes_color = sorted(font_sizes_color, key=lambda tup: (tup[0], tup[-1]), reverse=True)
        # aggregating the tags for each font size
        idx = tmp_idx = 0
        for seq, (size, font, color, font_style) in enumerate(font_sizes_color):
            same_p_size = size == p_size
            same_p_color = color == p_color
            bold = font_style >= 2 ** 4
            if seq > 0:  # if previous size and font are the same but different color, cancel the update heading/subscript counter
                prev_size = font_sizes_color[seq - 1][0]
                prev_font = font_sizes_color[seq - 1][1]
                prev_tag = list(self.font_property_tag.values())[-1]
                if prev_size == size and prev_font == font:
                    idx -= 1
            if (same_p_size and ((not same_p_color) or (same_p_color and bold))) or size > p_size:
                if seq > 0 and prev_tag == 'paragraph':
                    idx = tmp_idx
                self.font_property_tag[(size, font, tuple(color), font_style)] = 'heading{0}'.format(idx)
                idx += 1
                tmp_idx = idx
            elif same_p_size and (same_p_color and not bold):
                idx = 0
                self.font_property_tag[(size, font, tuple(color), font_style)] = 'paragraph'
            elif size < p_size:
                self.font_property_tag[(size, font, tuple(color), font_style)] = 'subscript{0}'.format(idx)
                idx += 1
        self.num_head = len(set([e for e in self.font_property_tag.values() if re.search('heading', e)]))
        self.num_sub = len(set([e for e in self.font_property_tag.values() if re.search('subscript', e)]))

    def process_font_tag(self):
        '''Process document text font analysis and tagging'''
        self.fonts_line_spacing2()
        self.word_spacing2()
        self.font_tags2()

    def style2tag(self, style: str) -> str:
        """
        Map fine-grained font style (heading_n, subscript_n, paragraph) to general element tag (title, section, caption, paragraph, list, footer)
        @param style: font style expressed in heading_n, subscript_n, paragraph
        @type style: str
        @rtype: str
        @return: general element tag (title, section, caption, paragraph, list, footer)
        """
        if re.search('heading|subscript', style):
            style_id = int(re.findall(r'\d+', style)[0])
        if re.search('heading', style):
            if self.num_head < 20:
                if style_id <= self.num_head / 2:
                    return 'title'
                elif self.num_head / 2 < style_id <= self.num_head * 2 / 3:
                    return 'section'
                else:
                    return 'caption'
            else:
                if style_id <= self.num_head * 0.3:
                    return 'title'
                elif self.num_head * 0.3 < style_id <= self.num_head * 0.4:
                    return 'section'
                elif self.num_head * 0.4 < style_id <= self.num_head * 0.6:
                    return 'caption'
                else:
                    return 'paragraph'
        elif re.search('subscript', style):
            if self.num_sub < 20:
                if style_id <= self.num_sub / 2:
                    return 'list'
                else:
                    return 'footer'
            else:
                if style_id <= self.num_sub * 0.6:
                    return 'paragraph'
                elif self.num_sub * 0.6 < style_id <= self.num_sub * 0.8:
                    return 'list'
                else:
                    return 'footer'
        else:
            return 'paragraph'

    def extract_token2(self):
        input_data_path = OUTPUT_LAYOUTLM_INPUT_DIR + self.fname + '.json'
        token_rule_tag = []  # list with headers and paragraphs
        bbox_cluster = []

        for page in self.doc.pages:
            w = page.width
            h = page.height
            page_id = page.page_number
            self.page_id2size[int(page_id)] = {'width': w, 'height': h}

        if (not os.path.exists(input_data_path) and self.dev_mode) or not self.dev_mode:
            for page in self.doc.pages:
                first = True  # boolean operator for first header
                words = page.extract_words()
                page_id = page.page_number
                img_path = os.path.join(self.img_outdir, self.fname + '_{}_ori.jpg'.format(str(page_id)))
                model_input = {
                    'id': int(page_id),
                    'tokens': [],
                    'bboxes': [],
                    'ner_tags': [],
                    'image_path': img_path
                }
                width = self.page_id2size[int(page_id)]['width']
                height = self.page_id2size[int(page_id)]['height']
                is_adjacent = False
                textline_id = 0
                block_string = ''
                for word in words:
                    '''
                    word['x0']: Distance of left side of character from left side of page. = x0
                    word['top']: Distance of top of character from top of page. = y0
                    word['x1']: Distance of right side of character from left side of page. = x1
                    word['bottom']: Distance of bottom of the character from top of page. = y1
                    '''
                    bbox = [word['x0'], word['top'], word['x1'], word['bottom']]
                    bbox = normalize_bbox(bbox, width, height)
                    x0, y0, x1, y1 = bbox
                    if not first:
                        is_adjacent = (abs(x0 - px1) <= self.mode_word_spacing[page_id] or abs(y0 - py0) <= self.upper_percentile_line_spacing[page_id])
                        if not is_adjacent:
                            textline_id += 1
                    token = word['text']
                    if token is not None:
                        token = discard_str_with_unwant_char(token)
                    if self.use_model and token:
                        token = filter_invalid_char(token)
                    if token:  # if token is not ''
                        # Handle surrogates characters that do not have a valid representation in Unicode
                        try:
                            token = token.encode('utf-8', 'ignore').decode('utf-8')
                        except UnicodeEncodeError:
                            continue
                        token = token.expandtabs(1)
                        if token == 'n':  # sometimes bullet-point character recognize as 'n', replace it with geometric shape
                            token = '•'
                        # add whitespace as delimiter to seperate camel case, e.g. theBorrower -> the Borrower
                        token = add_space2camel_case(token)
                        fontname, size, color = self.lookup_font_properties(word, page_id)
                        identifier = fontname + '@' + str(size) + '^' + str(tuple(color))
                        if identifier == self.font_counts[0][0]:
                            font_style = -1
                        else:
                            font_style = translate_fontname(fontname)
                        tag = self.font_property_tag[(size, fontname, tuple(color), font_style)]
                        # rename tag from headingn to title/section/caption and from subscriptn to list/footer
                        s_tag = self.style2tag(tag)
                        _, _, _, ny1 = bbox
                        # if any x,y coordinate beyond page boundaries, discard the text
                        if any((c <= 0) or (c >= 1000) for c in bbox):
                            continue
                        if page_id != 1 and (ny1 <= self.footer_lower_tolerance or ny1 >= self.footer_upper_tolerance):
                            s_tag = 'footer'
                        # previous text and current text very close to each other
                        if len(bbox_cluster)>1 and not first and s_tag == previous_s_tag and (abs(x0 - px1) <= 0.015 and abs(y0 - py0) <= 2) and textline_id == previous_textline_id + 1:
                            previous_token = model_input['tokens'].pop(-1)
                            model_input['bboxes'].pop(-1)
                            model_input['ner_tags'].pop(-1)
                            self.tokens.pop(-1)
                            bbox_cluster.pop(-1)
                            token = previous_token.strip() + ' ' + token.strip()
                            bbox = (px0, py0, x1, y1)
                        token = filter_invalid_char(token)
                        model_input['tokens'].append(token)
                        model_input['bboxes'].append(bbox)
                        model_input['ner_tags'].append(s_tag)
                        self.tokens.append({
                            'token': token,
                            'bbox': bbox,
                            'page_id': page_id,
                            'rule_tag': s_tag,
                        })
                        bbox_cluster.append(bbox)
                        if first:
                            first = False
                            block_string = word['text']
                        else:
                            if s_tag == previous_s_tag and previous_identifier == identifier and is_adjacent:
                                if block_string and all((c == "\n") for c in block_string):
                                    # block_string only contains pipes
                                    block_string = ''
                                    block_string += word['text']
                                if block_string == "":
                                    # new block has started, so append size tag
                                    block_string = word['text']
                                else:  # in the same block, so concatenate strings
                                    block_string += " " + word['text']
                            else:
                                bbox_cluster.pop(-1)
                                element_cluster_outline_bbox = bboxes_cluster_outline(bbox_cluster)
                                # count the length of string ignore tag and linebreak
                                token_rule_tag.append({
                                    'token': block_string,
                                    'bbox': element_cluster_outline_bbox,
                                    'page_id': page_id,
                                    'tag': previous_s_tag,
                                })
                                bbox_cluster = []
                                bbox_cluster.append(bbox)
                                block_string = word['text']
                        px0, py0, px1, py1 = (x0, y0, x1, y1)
                        previous_s_tag = s_tag
                        previous_textline_id = textline_id
                        previous_identifier = identifier
                self.model_input_data.append(model_input)
            self.txt_ele = token_rule_tag
            # if self.dev_mode:
            #     with open('test_txt_ele_rule2.json', 'w') as output:
            #         json.dump(self.txt_ele, output, indent=4, ensure_ascii=False)
            if self.dev_mode and not os.path.exists(OUTPUT_LAYOUTLM_INPUT_DIR):
                os.makedirs(OUTPUT_LAYOUTLM_INPUT_DIR)
            if self.dev_mode:
                with open(input_data_path, 'w') as output:
                    ndjson.dump(self.model_input_data,output, ensure_ascii=False)
        else:
            token_rule_tag = []
            with open(input_data_path, 'r') as f:
                for line in f:
                    d = json.loads(line)
                    self.model_input_data.append(d)
                    tokens = d["tokens"]
                    bboxes = d["bboxes"]
                    page_id = [d["id"]] * len(d["tokens"])
                    ner_tags = d["ner_tags"]
                    keys = ['token', 'bbox', 'page_id', 'tag']
                    pairs = list(zip(tokens, bboxes, page_id, ner_tags))
                    dic = [dict(zip(keys, v)) for v in pairs]
                    token_rule_tag.extend(dic)
            self.txt_ele = token_rule_tag
        self.doc_tag_set = set([e['tag'] for e in self.txt_ele])

if __name__ == '__main__':
    pdf_path = "/Users/data/Documents/GitHub/ldrs/analytics/data/pdf/unannotated_fa_ts/TS/(1) sample TS_1.pdf"
    img_outdir = "/Users/data/Documents/GitHub/ldrs/analytics/data/image_share"
    OUTPUT_LAYOUTLM_INPUT_DIR = '/Users/data/Documents/GitHub/ldrs/analytics/data/layoutlm_input_data'
    obj = DocParser(pdf_path, img_outdir)
    v = obj.process_font_tag()
    k = obj.extract_token2()
