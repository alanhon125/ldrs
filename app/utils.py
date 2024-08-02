from spacy.language import Language
import logging
import traceback
import sys

logger = logging.getLogger(__name__)

try:
    from regexp import *
except ImportError:
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import sys
    sys.path.insert(0, '..')
    from regexp import *


async def POST(url, data):
    from django.http import JsonResponse
    import json
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            headers = {'Content-type': 'application/json',
                    'Accept': 'application/json'}
            response = await client.post(url, data=json.dumps(data), headers=headers)
            response.raise_for_status()
            data = response.json()

    except httpx.HTTPError as e:
        # Handle the HTTP error
        return {'success': False, 'errorMessage': str(e)} # JsonResponse({'success': False, 'errorMessage': str(e)}, status=500)
    
    # Process the data and return an HTTP response
    return data # JsonResponse(data)



async def update_doc_error_status(fileid, message):
    try:
        from config import UPDATE_DOC_STATUS_URL
    except:
        import os
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        import sys
        sys.path.insert(0, '..')
        from config import UPDATE_DOC_STATUS_URL
    update_status = {
        "id": fileid,
        "fileStatus": "ERROR",
        "message": message
    }
    await POST(UPDATE_DOC_STATUS_URL, update_status)

async def update_task_error_status(taskId, makerGroup, name, message):
    from config import UPDATE_TASK_URL
    update_status = {
                "id": taskId,
                "makerGroup": makerGroup,
                "name": name,
                "status": "ERROR",
                "systemStatus": "ERROR",
                "message": message
            }
    await POST(UPDATE_TASK_URL, update_status)

def romanToInt(s):
    """
    :type s: str
    :rtype: int
    """
    s = s.upper()
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000, 'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90,
             'CD': 400, 'CM': 900}
    i = 0
    num = 0
    try:
        while i < len(s):
            if i + 1 < len(s) and s[i:i + 2] in roman:
                num += roman[s[i:i + 2]]
                i += 2
            else:
                # print(i)
                num += roman[s[i]]
                i += 1
        return num
    except:
        return None


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


def neighborhood(iterable):
    iterator = iter(iterable)
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (prev_item, current_item, next_item)
        prev_item = current_item
        current_item = next_item
    yield (prev_item, current_item, None)


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


def is_bbox_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap.

    Args:
        bbox1: A tuple of four values (x0, y0, x1, y1) representing the first bounding box.
        bbox2: A tuple of four values (x0, y0, x1, y1) representing the second bounding box.

    Returns:
        A boolean indicating whether the two bounding boxes overlap.
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # Check if there is no overlap
    if x0_1 > x1_2 or x0_2 > x1_1 or y0_1 > y1_2 or y0_2 > y1_1:
        return False

    # Otherwise, there is overlap
    return True


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


def create_save_pdf_img(pdf_inpath, img_outdir, fname):
    '''
    create pdf page images by pdf2image libary and output to image folder
    @param pdf_inpath: path to pdf
    @type pdf_inpath: str
    @param img_outdir: the output directory where the output page images will be written
    @type img_outdir: str
    @param fname: filename
    @type fname: str
    '''

    from pdf2image import pdfinfo_from_path, convert_from_path
    import os

    info = pdfinfo_from_path(pdf_inpath, userpw=None, poppler_path=None)

    maxPages = info["Pages"]
    all_img_paths = [os.path.join(img_outdir, fname + f'_{str(i)}_ori.jpg') for i in range(1, maxPages + 1)]
    check_existence = [os.path.exists(img_path) for img_path in all_img_paths]
    if all(check_existence):
        return

    chunkPages = 100
    count = 0
    for page in range(1, maxPages + 1, chunkPages):
        pdf_images = convert_from_path(
            pdf_inpath, dpi=200, first_page=page, last_page=min(page + chunkPages - 1, maxPages))
        for page_id in range(chunkPages):  # save document page images
            if count > maxPages-1:
                break
            outpath = os.path.join(img_outdir, fname + '_{}_ori.jpg'.format(str(page_id + page)))
            if not os.path.exists(outpath):
                try:
                    pdf_images[page_id].save(outpath)
                except Exception as e:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    tb = traceback.TracebackException(exc_type, exc_value, exc_tb)
                    invalid_msg = 'In task name: "PDF-to-images"\n'+''.join(tb.format())
                    logger.error(invalid_msg)
                    
            count += 1


def annot_pdf_page(doc, page_id, text_label, norm_bbox, color=None):
    '''
    Annotate the page with given page id, label and bounding box
    @param doc: PyMuPDF class that represent the document
    @type doc: <class 'fitz.fitz.Document'>
    @param page_id:page id
    @type page_id: int
    @param text_label: text label above bounding box on an annotated page
    @type text_label: str
    @param norm_bbox: normalized bounding box in (x0, y0, x1, y1)
    @type norm_bbox: list
    @param color: list
    @type color: list
    @return: annotated document page
    @rtype: <class 'fitz.fitz.Document'>
    '''
    import time
    import random

    if color is None:
        random.seed(time.process_time())
        color = (random.random(), random.random(), random.random())
    w, h = page_size(doc, page_id)
    doc[page_id].clean_contents()
    x0, y0, x1, y1 = denormalize_bbox(norm_bbox, w, h)
    doc[page_id].insert_text((x0, y0 - 2), text_label, fontsize=8, color=color)
    doc[page_id].draw_rect((x0, y0, x1, y1), color=color, width=1)
    return doc

def create_folder(out_folders):
    import os

    '''Create folder for data output if the directory doesn't exist'''
    if isinstance(out_folders, list):
        for f in out_folders:
            isExist = os.path.exists(f)
            if not isExist:
                os.makedirs(f)
    else:
        isExist = os.path.exists(out_folders)
        if not isExist:
            os.makedirs(out_folders)


def multiprocess(function, input_list, args=None):
    '''
    multiprocessing the function at a time

    @param function: a function
    @type function: def
    @param input_list: a list of input that accept by the function
    @type input_list: list
    @param args: arguments variables
    @type args: any argument type as required by function
    '''
    import multiprocessing

    def contains_explicit_return(function):
        # Check if function has return statement
        import ast
        import inspect

        return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(function))))

    input_length = len(input_list)
    num_processor = multiprocessing.cpu_count()
    batch_size = max(input_length // num_processor, 1)
    num_batch = int(input_length / batch_size) + (input_length % batch_size > 0)
    logger.info(f'It is going to apply multi-processing on function {function.__name__} with following setting:\nCPU cores count: {num_processor}\nList length: {input_length}\nBatch size: {batch_size}\nBatch no.: {num_batch}')
    pool = multiprocessing.Pool(num_processor)
    if not args:
        processes = [pool.apply_async(function, args=(input_list[idx * batch_size:(idx + 1) * batch_size],)) for idx in range(num_batch)]
    else:
        processes = [pool.apply_async(function, args=(input_list[idx * batch_size:(idx + 1) * batch_size], args)) for idx in range(num_batch)]

    if contains_explicit_return(function):  # if function has return statement
        results = [p.get() for p in processes]
        results = [i for r in results if isinstance(r, list) or isinstance(r, tuple) for i in r]
        # close the process pool
        pool.close()
        pool.join()

        return results
    else:
        for p in processes:
            p.get()
        # close the process pool
        pool.close()
        pool.join()
    logger.info(f'Completed multi-processing on function {function.__name__}')


def list2dict(text, nlp, use_nested=True, override_key=None, delimiter_pattern=PARA_SPLIT_PATTERN, special_split=False):
    import re
    import string

    bullet_list_symbol = '(?<!\[)(?<!\()' + '(?!\))(?!\])|(?<!\[)(?<!\()'.join(SYMBOLS) + '(?!\))(?!\])'

    def spliting_text(text):

        NLA_BRACK_PAT = rf'(?!{")(?!".join(NLA_NUM_BRACK)})'
        NLB_BRACK_PAT = rf'(?<!{")(?<!".join(NLB_NUM_BRACK)})'

        # numbering pattern in brackets should not precede by ), non-whitespace character and not succeed by (
        brackets_pattern = f'(?<!\b\w{1}\), )(?<!\b\w{2}\), )(?<!\b\w{1}\))(?<!\b\w{2}\))(?<![^\s\[]){NLB_BRACK_PAT}\(' + f'\)(?!\()(?!:){NLA_BRACK_PAT}' \
            '|' \
            f'(?<!\b\w{1}\), )(?<!\b\w{2}\), )(?<!\b\w{1}\))(?<!\b\w{2}\))(?<![^\s\[]){NLB_BRACK_PAT}\('.join(
                NUMBERING_LIST) + f'\)(?!\()(?!:){NLA_BRACK_PAT}'
        # numbering pattern with right bracket only should not succeed by (
        right_bracket_pattern = f'(?<!\b\w{1}\), )(?<!\b\w{2}\), )(?<!\()(?<![^\s\[])(?<!\) and )(?<!\) or ){NLB_BRACK_PAT}' + f'\)(?!\()(?!:){NLA_BRACK_PAT}' \
            '|' \
            f'(?<!\b\w{1}\), )(?<!\b\w{2}\), )(?<!\()(?<![^\s\[])(?<!\) and )(?<!\) or ){NLB_BRACK_PAT}'.join(
                NUMBERING_LIST) + f'\)(?!\()(?!:){NLA_BRACK_PAT}'
        # numbering pattern with dot should not succeed by any digit
        dot_pattern = '(?<![a-zA-Z]\.)' + '\.(?!\d)(?![a-zA-Z]\.)|(?<![a-zA-Z]\.)'.join([i for i in NUMBERING_LIST if i not in [CAPITAL_ALPHABET,CAPITAL_XYZ]]) + '\.(?!\d)(?![a-zA-Z]\.)'
        pattern = '(' + brackets_pattern + '|' + right_bracket_pattern + '|' + dot_pattern + ')'

        splits = [i.strip() for i in re.split(pattern, text) if i and i.strip()]
        splits = [x for x in splits if not re.match(rf'^{ROMAN_NUM}$|^{TWO_LOWER_ALPHABET}$|^{TWO_CAPITAL_ALPHABET}$', str(x), flags=re.IGNORECASE)]
        splits = [x for x in splits if x not in string.punctuation]

        if len(splits) <= 1:
            return splits

        def insert_numbering_at_even(curr_txt, split_list):
            if re.match(pattern, curr_txt) and len(split_list) % 2 == 0:
                split_list.append(curr_txt)
            elif not re.match(pattern, curr_txt) and len(split_list) % 2 == 1:
                split_list.append(curr_txt)
            else:
                if len(split_list) > 0 and not re.match(pattern, split_list[-1]):
                    last = split_list.pop(-1)
                    split_list.append(last + ' ' + curr_txt)
                else:
                    split_list.append(' ')
                    split_list.append(curr_txt)
            return split_list

        iterator = neighborhood(splits)
        tmp_splits = []
        continue_again = False
        for prev, curr, nxt in iterator:
            if continue_again:
                continue_again = False
                continue
            if not prev:
                tmp_splits = insert_numbering_at_even(curr, tmp_splits)
                continue
            else:
                prev = tmp_splits[-1]
                if re.match(pattern, curr) and ((curr.endswith(')') and re.search(r'.*' + rf'{r"$|.*".join(NLB_BRACK)}' + r'$', prev)) or ((curr.endswith('.') and re.search(r'.*' + rf'{r"$|.*".join(NLB_DOT)}' + r'$', prev)))):
                    prev = tmp_splits.pop(-1)
                    if not prev.endswith('.'):
                        prev += ' '
                    if not curr.endswith('.'):
                        curr += ' '
                    if nxt:
                        tmp_splits = insert_numbering_at_even(
                            prev + curr + nxt, tmp_splits)
                    else:
                        tmp_splits = insert_numbering_at_even(
                            prev + curr, tmp_splits)
                    continue_again = True
                    continue
                # elif re.match(pattern, prev) and re.match(pattern, curr):
                #     tmp_splits.append(' ')
                #     tmp_splits.append(curr)
                else:
                    tmp_splits = insert_numbering_at_even(curr, tmp_splits)
            if nxt:
                if re.match(pattern, curr) and ((curr.endswith(')') and re.search(r'^' + rf'{r"|^".join(NLA_NUM_BRACK2)}', nxt)) or ((curr.endswith('.') and re.search(r'^' + rf'{r"|^".join(NLA_NUM_DOT2)}', nxt)))):
                    curr = tmp_splits.pop(-1)
                    if not curr.endswith('.'):
                        curr += ' '
                    tmp_splits = insert_numbering_at_even(curr + nxt, tmp_splits)
                    continue_again = True
                    continue

        return tmp_splits

    splits = spliting_text(text)

    def split_bullet(text):
        # split string into list with bullet point characters
        # if len(splits) <= 1:
        splits = re.split(bullet_list_symbol, text)
        splits = ['• ' + x.strip() for x in splits if x != '' and x != ' ']
        if len(splits) > 1:
            return splits
        else:
            return phrase_tokenize(text, nlp, delimiter_pattern=delimiter_pattern, special_split=special_split)

    if len(splits) <= 1:
        splits = split_bullet(text)
        return splits, None

    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    splits = pairwise(splits)

    def hierarchical_numbering(splits):
        dic = {}
        primaryLevel = secondaryLevel = tertiaryLevel = quaternaryLevel = quinaryLevel = ''
        last_primary_key = last_secondary_key = last_tertiary_key = last_quaternary_key = last_quinary_key = ''
        last_pattern = ''
        if override_key:
            keys = re.findall(r'(\(*\w+\)*\.*)', override_key)
            if len(keys) <= 4:
                i = 1
                while keys:
                    if i == 1:
                        last_primary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_primary_key.strip("().")):
                                primaryLevel = p
                                break
                    elif i == 2:
                        last_secondary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_secondary_key.strip("().")):
                                secondaryLevel = p
                                break
                    elif i == 3:
                        last_tertiary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_tertiary_key.strip("().")):
                                tertiaryLevel = p
                                break
                    elif i == 4:
                        last_quaternary_key = keys.pop(0)
                        for p in NUMBERING_LIST:
                            if re.match(p, last_quaternary_key.strip("().")):
                                quaternaryLevel = p
                                break
                    i += 1

        for prev, curr, nxt in neighborhood(list(splits)):
            curr_key, curr_value = curr
            # curr_value = tuple(re.split(bullet_list_symbol, curr_value))
            # if len(curr_value) == 1:
            #     curr_value = curr_value[0]
            c_key = curr_key.strip("().")
            if prev is None and override_key is None:
                for p in NUMBERING_LIST:
                    if re.match(p, c_key):
                        primaryLevel = p
                        break
                p_key = None
            elif prev is None and override_key:
                p_key = last_primary_key.strip("().")
                for p in NUMBERING_LIST:
                    if re.match(p, c_key):
                        if quaternaryLevel:
                            quinaryLevel = p
                        elif tertiaryLevel:
                            quaternaryLevel = p
                        elif tertiaryLevel:
                            tertiaryLevel = p
                        elif secondaryLevel:
                            tertiaryLevel = p
                        elif primaryLevel:
                            secondaryLevel = p
                        break
            else:
                prev_key, prev_value = prev
                p_key = prev_key.strip("().")
            if nxt is not None:
                next_key, next_value = nxt
                n_key = next_key.strip("().")
            else:
                n_key = None

            def isNextNum(prev_num, curr_num):
                # if previous number is None or empty string
                if prev_num is None or prev_num == '':
                    return True
                else:
                    prev_num = prev_num.strip('()')
                # if two numbers are digits
                if str(curr_num).isdigit() and str(prev_num).isdigit():
                    if int(curr_num) == int(prev_num) + 1:
                        return True
                try:
                    # if two numbers are alphabet character
                    if curr_num == chr(ord(prev_num) + 1):
                        return True
                    else:
                        try:
                            prev_num = romanToInt(prev_num)
                            curr_num = romanToInt(curr_num)
                            # if two numbers are roman numbers character
                            if prev_num and curr_num:
                                if curr_num == prev_num + 1:
                                    return True
                        except:
                            return False
                except:
                    return False

            if re.match(primaryLevel, c_key):
                last_primary_key = curr_key
                last_primary_value = curr_value
            elif re.match(secondaryLevel, c_key):
                last_secondary_key = curr_key
                last_secondary_value = curr_value
            elif re.match(tertiaryLevel, c_key):
                last_tertiary_key = curr_key
                last_tertiary_value = curr_value
            elif re.match(quaternaryLevel, c_key):
                last_quaternary_key = curr_key
                last_quaternary_value = curr_value
            
            # always put digit with dot (e.g. 1. / 2. /3. etc.) as the primary list item key
            if re.match('^\d+\.', curr_key):
                primaryLevel = secondaryLevel = tertiaryLevel = quaternaryLevel = quinaryLevel = ''
                last_primary_key = last_secondary_key = last_tertiary_key = last_quaternary_key = last_quinary_key = ''
                last_primary_key = curr_key
                last_primary_value = curr_value
                for p in NUMBERING_LIST:
                    if re.match(p, c_key):
                        primaryLevel = p

            for i, k in enumerate(NUMBERING_LIST):
                if n_key is not None:
                    match_c = re.match(k, c_key)
                    match_n = re.match(k, n_key)
                    if match_c and match_n:
                        match_n = isNextNum(c_key, n_key)
                    not_same_pattern = (match_c and (
                        not match_n or match_n is None))
                    if not_same_pattern or match_n == False:
                        if k == primaryLevel:
                            for p in NUMBERING_LIST:
                                if p == primaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    secondaryLevel = p
                                    break
                            tmp = {curr_key: {curr_value: {}}}
                        elif k == secondaryLevel:
                            for p in NUMBERING_LIST:
                                if p == secondaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    tertiaryLevel = p
                                    break
                            if re.match(primaryLevel, n_key):
                                tmp = {curr_key: curr_value}
                            else:
                                tmp = {curr_key: {curr_value: {}}}
                        elif k == tertiaryLevel:
                            for p in NUMBERING_LIST:
                                if p == tertiaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    quaternaryLevel = p
                                    break
                            if re.match(primaryLevel, n_key) or re.match(secondaryLevel, n_key):
                                tmp = {curr_key: curr_value}
                            else:
                                tmp = {curr_key: {curr_value: {}}}
                        elif k == quaternaryLevel:
                            for p in NUMBERING_LIST:
                                if p == quaternaryLevel:
                                    continue
                                if re.match(p, n_key):
                                    quinaryLevel = p
                                    break
                            if re.match(primaryLevel, n_key) or re.match(secondaryLevel, n_key) or re.match(
                                    tertiaryLevel, n_key):
                                tmp = {curr_key: curr_value}
                            else:
                                tmp = {curr_key: {curr_value: {}}}
                        else:
                            tmp = {curr_key: curr_value}
                        break
                else:
                    tmp = {curr_key: curr_value}
                    break
                if i == len(NUMBERING_LIST) - 1:
                    tmp = {curr_key: curr_value}

            if re.match(primaryLevel, c_key) or (last_pattern == primaryLevel and isNextNum(p_key, c_key)):
                last_pattern = primaryLevel
                if use_nested:
                    dic.update(tmp)
                else:
                    while True:
                        if curr_key in dic.keys():
                            curr_key += ' '
                        if not curr_key in dic.keys():
                            break
                    final_key = curr_key
                    dic.update({final_key: curr_value})
            elif re.match(secondaryLevel, c_key) or (last_pattern == secondaryLevel and isNextNum(p_key, c_key)):
                last_pattern = secondaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value].update(tmp)
                else:
                    while True:
                        if last_primary_key + curr_key in dic.keys():
                            curr_key += ' '
                        if not last_primary_key + curr_key in dic.keys():
                            break
                    final_key = last_primary_key + curr_key
                    dic.update({final_key: curr_value})
            elif re.match(tertiaryLevel, c_key) or (last_pattern == tertiaryLevel and isNextNum(p_key, c_key)):
                last_pattern = tertiaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value].update(
                        tmp)
                else:
                    while True:
                        if last_primary_key + last_secondary_key + curr_key in dic.keys():
                            curr_key += ' '
                        if not last_primary_key + last_secondary_key + curr_key in dic.keys():
                            break
                    final_key = last_primary_key + last_secondary_key + curr_key
                    dic.update(
                        {final_key: curr_value})
            elif re.match(quaternaryLevel, c_key) or (last_pattern == quaternaryLevel and isNextNum(p_key, c_key)):
                last_pattern = quaternaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value][last_tertiary_key][last_tertiary_value].update(tmp)
                else:
                    while True:
                        if last_primary_key + last_secondary_key + last_tertiary_key + curr_key in dic.keys():
                            curr_key += ' '
                        if not last_primary_key + last_secondary_key + last_tertiary_key + curr_key in dic.keys():
                            break
                    final_key = last_primary_key + last_secondary_key + last_tertiary_key + curr_key
                    dic.update({final_key: curr_value})
            elif re.match(quinaryLevel, c_key) or (last_pattern == quinaryLevel and isNextNum(p_key, c_key)):
                last_pattern = quinaryLevel
                if use_nested:
                    dic[last_primary_key][last_primary_value][last_secondary_key][last_secondary_value][last_tertiary_key][last_tertiary_value][last_quaternary_key][last_quaternary_value].update(tmp)
                else:
                    while True:
                        if last_primary_key + last_secondary_key + last_tertiary_key + last_quaternary_key + curr_key in dic.keys():
                            curr_key += ' '
                        if not last_primary_key + last_secondary_key + last_tertiary_key + last_quaternary_key + curr_key in dic.keys():
                            break
                    final_key = last_primary_key + last_secondary_key + last_tertiary_key + last_quaternary_key + curr_key
                    dic.update({final_key: curr_value})
            
        return dic, final_key

    dic, final_key = hierarchical_numbering(splits)

    # if isinstance(dic, dict):
    #     dic = dict_value_phrase_tokenize(
    #         dic, nlp, delimiter_pattern=delimiter_pattern)

    return dic, final_key

def remove_punctuation(s):
    '''
    Remove all punctuations in a string
    '''
    import string
    translator = str.maketrans('', '', string.punctuation.replace('-', ''))
    return s.translate(translator).strip() if s else None


def string_with_whitespace(str_list):
    '''
    insert whitespace that interrupt the string
    :param str_list: list of string
    :return: list of combination of strings interrupted by whitespace
    '''
    def insert_space(string, pos):
        return string[0:pos] + ' ' + string[pos:]
    if isinstance(str_list, str):
        str_list = [str_list]
    result = []
    for idx, s in enumerate(str_list):
        length = len(s)
        for i in range(length):
            result.append(insert_space(s, i).strip())
    return result


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
    my_str = re.sub(' / ', '/', my_str)
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
    my_str = re.sub(r'‘', r"'", my_str)
    # my_str = re.sub(r'([\uff01-\uff5e]+)', '', my_str) # fullwidth ascii variants
    # my_str = re.sub(r'([\u2018\u2019\u201a-\u201d]+)', '', my_str)  # Quotation marks and apostrophe
    return my_str.strip()


def replace_char(my_str):
    import re

    my_str = re.sub(' +', ' ', my_str)
    my_str = re.sub('', '', my_str)
    my_str = re.sub(r' ([\.,;]) ', r'\1 ', my_str)
    my_str = re.sub(r' ([\)\]])', r'\1', my_str)
    my_str = re.sub(r' ([\(\[]) ', r' \1', my_str)
    my_str = re.sub(r'“', r'"', my_str)
    my_str = re.sub(r'”', r'"', my_str)
    my_str = re.sub(r'’', r"'", my_str)

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


@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text in [".(", ").", ".[", "]."]:
            doc[token.i + 1].is_sent_start = True
        elif token.text in [")", "]", ";"]:
            doc[token.i + 1].is_sent_start = False
        elif token.text in ["(", "["]:
            doc[token.i].is_sent_start = False
        
        # if current token is "." and previous token is non-digit, not to consider next token as sentence beginning
        if token.text == "." and re.match(r'\D\.', doc[token.i - 1].text):
            doc[token.i + 1].is_sent_start = False

        # if current token is "." and next token is a quote character, consider next token as sentence beginning
        if token.text == "." and doc[token.i + 1].text in ('"',"'"):
            doc[token.i + 1].is_sent_start = True

        # when period (.) appear in present token, if match the following regular expression patterns ,then consider next token is sentence start
        regex_patterns = ["[A-Z][a-z]+"]  # Capital words
        for regex in regex_patterns:
            if token.text == "." and re.match(regex, doc[token.i - 1].text):
                doc[token.i + 1].is_sent_start = True
            elif doc[token.i - 1].text != "." and re.match(regex, token.text):
                doc[token.i].is_sent_start = False
        
        # when the following regular expression patterns match in previous token ,then not to consider current token is sentence start
        regex_patterns = ["\d{1,2}[a-zA-Z]*\.\d{0,2}\.*\d{0,2}\.*\d{0,2}"]
        for regex in regex_patterns:
            if re.match(regex, token.text):
                doc[token.i + 1].is_sent_start = False
            if doc[token.i - 1].text == "." and re.match(regex, token.text):
                doc[token.i].is_sent_start = True

        # if current token is not "." and previous token is digit, not to consider current token as sentence beginning
        if token.text != "." and re.match('\d+', doc[token.i - 1].text):
            doc[token.i].is_sent_start = False

        # when period (.) appear in present token, if match the following abbreviations ,then not to consider next token is sentence start
        special_abbrev = ["cent", "approx", "sect", "para", "cap", "sq", "ft", "sc", "\d+[a-zA-Z]*"]
        if token.text == "." and re.match(r'|'.join(special_abbrev), doc[token.i - 1].text, re.IGNORECASE):
            doc[token.i + 1].is_sent_start = False

    return doc


special_abbrev = ["cent.", "approx."]


@Language.factory("exc_retokenizer")
class ExceptionRetokenizer:
    def __init__(self, nlp, name="exc_retokenizer"):
        from spacy.matcher import PhraseMatcher
        self.name = name
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        for exc in special_abbrev:
            pattern_docs = [
                nlp.make_doc(text)
                for text in [exc, exc.upper(), exc.lower(), exc.title()]
            ]
            self.matcher.add("A", pattern_docs)

    def __call__(self, doc):
        from spacy.util import filter_spans
        with doc.retokenize() as retokenizer:
            for match in filter_spans(self.matcher(doc, as_spans=True)):
                retokenizer.merge(match)
        return doc


def config_nlp_model(nlp):
    '''
    configure special cases into spacy Langugage model in sentence tokenizer
    '''
    from spacy.language import Language
    from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
    from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
    from spacy.util import compile_infix_regex
    from spacy.matcher import PhraseMatcher
    from spacy.language import Language
    from spacy.util import filter_spans

    if "set_custom_boundaries" not in nlp.pipe_names:
        nlp.add_pipe("set_custom_boundaries", before="parser")

    # if "exc_retokenizer" not in nlp.pipe_names:
    #     nlp.add_pipe("exc_retokenizer")

    # Modify tokenizer infix patterns
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # ✅ Commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    return nlp


def phrase_tokenize(string, nlp, delimiter_pattern=r'( and/or;|; and/or| and;|; and| or;|; or|and/or|;)', special_split=False):
    '''
    Tokenize string into sentence, then split sentence into phrases with given delimiter pattern
    @param string: string of paragraph
    @type string: str
    @param nlp: spacy nlp Language model object for sentence tokenization
    @type nlp: spacy.load('en_core_web_sm') object
    @param delimiter_pattern: regular expression pattern of DELIMITERS
    @type delimiter_pattern: str
    @rtype string: List[str]
    '''
    import re

    string = re.sub(r'No\.|no\.', 'number', string)
    # string = re.sub(r'(?<= [.(a-zA-z]{3})\.(?!=(\n))', '', string)
    # string = re.sub(r'(?<= [a-zA-z]{2})\.(?!=(\n))', '', string)

    doc = nlp(string)
    string = [sent.text for sent in doc.sents]
    lsts = []
    for s in string:
        if (len(s.split(','))>=5 or 'but not limited to' in s) and special_split:
            delimiter_pattern = COMMA_SPLIT_PATTERN
        lsts.append([i for i in re.split(delimiter_pattern, s) if i and i.strip() != ''])

    new_lst = []
    for lst in lsts:
        for i, v in enumerate(lst):
            if re.match(delimiter_pattern, v):
                if new_lst:
                    tmp = new_lst.pop(-1) + v + ' '
                    new_lst.append(tmp)
                else:
                    new_lst.append(v)
            elif v and not re.match(delimiter_pattern, v):
                if i!=0 and len(new_lst[-1].split())>4 and len(v.split())>4 and ',' in delimiter_pattern:
                    tmp = new_lst.pop(-1) + v + ' '
                    new_lst.append(tmp)
                else:
                    new_lst.append(v)

        string = [i.strip() for i in new_lst]
    # if there are more than one colon (:), string will split into two phrases, need to combine it back into one string
    if all([i.endswith(':') for i in string]) and len(string) > 1:
        string = ' '.join(string)
    if len(string) > 1:
        string = [re.sub(' +', ' ', re.sub('number', 'No.', i)) for i in string]
        # remove pure punctuations and digits
        string = [l for l in string if not re.match(PURE_PUNCT_DIGIT, l)]
    elif len(string) == 1:
        string = string[0]
        string = re.sub(' +', ' ', re.sub('number', 'No.', string))
    return string

def dict_value_phrase_tokenize(dic,nlp,delimiter_pattern=r'( and/or;|; and/or| and;|; and| or;|; or|and/or|;)', special_split=False):
    for k,v in dic.items():
        if isinstance(v,str):
            values = phrase_tokenize(v,nlp,delimiter_pattern=delimiter_pattern, special_split=special_split)
        else:
            continue
        if len(values)==1:
            values = values[0]

        if all([len(i) == 1 for i in values]):
            values = ''.join(values)

        dic[k] = values
    return dic

def flatten(dictionary, parent_key='', separator='_'):
    '''
    flatten a nested dictionary like:
    {'a': {'b': 'XXX', 'c': 'XXXX'}}
    into {'a_b': 'XXX', 'a_c': 'XXXX'}
    where '_' is the default separator and may be customized
    '''
    from collections.abc import MutableMapping
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def log_task2csv(csv_name, row):
    '''
    log the record 'row' into path 'csv_name'
    '''
    import csv
    import os

    file_exists = os.path.isfile(csv_name)
    # Open the CSV file in "append" mode
    with open(csv_name, 'a', newline='') as f:
        # Create a dictionary writer with the dict keys as column fieldnames
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
            # Append single row to CSV
        writer.writerow(row)


def remove_black_background(img):
    # source: https://www.geeksforgeeks.org/removing-black-background-and-make-transparent-using-python-opencv/

    import cv2

    # Convert image to image gray
    if len(img.shape) > 2:
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        tmp = img

    # Applying thresholding technique
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

    # Using cv2.split() to split channels
    # of coloured image
    splits = cv2.split(img)

    if len(splits) == 4:
        rgba = splits
    elif len(splits) == 3:
        # Making list of Red, Green, Blue
        # Channels and alpha
        rgba = list(splits) + [alpha]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        splits = cv2.split(img)
        rgba = list(splits) + [alpha]

    # Using cv2.merge() to merge rgba
    # into a coloured/multi-channeled image
    dst = cv2.merge(rgba, 4)

    return dst


def white_pixels_to_transparent(img):
    # remove white pixels to transparent
    # source: https://stackoverflow.com/questions/55673060/how-to-set-white-pixels-to-transparent-using-opencv

    import cv2
    import numpy as np

    # get the image dimensions (height, width and channels)
    h, w, c = img.shape

    if c == 4:  # image is BGRA image
        # Making list of Red, Green, Blue
        # Channels and alpha
        image_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        image_bgra = img
    elif c == 3:  # image is BGR image
        image_bgr = img
        # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
        image_bgra = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    else:
        image_bgr = img
        tmp = img
        for _ in range(2):
            # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
            image_bgra = np.concatenate([tmp, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
            tmp = image_bgra
        image_bgra = tmp

    if c < 3:
        # create a mask where white pixels ([255, 255, 255]) are True
        white = np.all(image_bgr == 255, axis=-1)
    else:
        # create a mask where white pixels ([255, 255, 255]) are True
        white = np.all(image_bgr == [255, 255, 255], axis=-1)
    # change the values of Alpha to 0 for all the white pixels
    image_bgra[white, -1] = 0

    return image_bgra


def erase_cache_files(filePath):
    import os
    if os.path.exists(filePath):
        os.remove(filePath)
