try:
    from app.utils import *
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, '..')
    from utils import *

def is_ngram_substring(a_str, b_str):
    '''
    N-grams are continuous sequences of words or symbols, or tokens in a document.
    e.g. Trigram of “I reside in Bengaluru” = [“I reside in”, “reside in Bengaluru”]
    Parameters
    ----------
    a_str: string to be split into (n-1)grams tuples and compare with b_str
    b_str: to be tested as a whole string

    Returns: boolean that indicate if (n-1)grams of a_str is in b_str
    -------
    '''

    from nltk import ngrams
    n = len(str(a_str).split())
    m = len(str(b_str).split())
    if n <= 1 or m <= 1:
        return False, 0

    for k in reversed(range(2, n - 1)):
        n_minus_1_grams = ngrams(str(a_str).lower().split(), k)
        for grams in n_minus_1_grams:
            gram_str = ' '.join(grams)
            if gram_str in str(b_str).lower():
                word_count = len(gram_str.split(' '))
                return True, word_count

    for k in reversed(range(2, m - 1)):
        m_minus_1_grams = ngrams(str(b_str).lower().split(), k)
        for grams in m_minus_1_grams:
            gram_str = ' '.join(grams)
            if gram_str in str(a_str).lower():
                word_count = len(gram_str.split(' '))
                return True, word_count

    return False, 0

def match_word_count(a_str,b_str):
    import nltk
    from nltk import word_tokenize
    import re
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    a_str = re.sub(r'[^\w\s]', '', str(a_str)) # removing punctuations in string using regex
    b_str = re.sub(r'[^\w\s]', '', str(b_str))
    a_str_words = word_tokenize(a_str.lower()) # tokenize sentence into list of words
    b_str_words = word_tokenize(b_str.lower())
    a_in_b_count = sum([a in b_str_words for a in a_str_words])
    b_in_a_count = sum([b in a_str_words for b in b_str_words])
    return max(a_in_b_count, b_in_a_count)

def get_text_identifier_by_bbox(data, ref_text, page_id, bbox, width, height):
    '''
    data = [{
        "createTime": "2023-10-27T08:05:50.009Z",
        "createUser": "string",
        "updateTime": "2023-10-27T08:05:50.009Z",
        "updateUser": "string",
        "id": 0,
        "fileId": 1,
        "indexId": 1,
        "textBlockId": 1,
        "pageId": "1",
        "sectionId": "string",
        "sectionContent": "string",
        "subSectionId": "string",
        "subSection": "string",
        "scheduleId": "string",
        "schedule": "string",
        "partId": "string",
        "part": "string",
        "textElement": "string",
        "listId": "string",
        "bbox": [
            0
        ],
        "definition": "string",
        "identifier": "string",
        'textContent': "string"
        },...]
    '''
    import pandas as pd

    norm_bbox = normalize_bbox(bbox, width, height)
    df = pd.DataFrame(data)
    # df['bbox'] = df['bbox'].map(lambda x: ast.literal_eval(x) if x else x)
    df['isMatch'] = df['bbox'].map(lambda x: is_bbox_overlap(x, bbox))
    df['pageId'] = df['pageId'].astype(int)
    df['isEqualText'] = df['textContent'].map(lambda x: str(x).lower() == str(ref_text).lower() if x else False)
    df['isSubstring'] = df['textContent'].map(lambda x: (str(ref_text).lower() in str(x).lower()) if x else False) # str(x).lower() in str(ref_text).lower() or 
    df['isNgramsSubstring'], df['matchWordCount'] = zip(*df['textContent'].map(lambda x: is_ngram_substring(x,ref_text) if x else False, None))

    matched_group = df.loc[(df['pageId']==int(page_id)) & (df['isMatch'].eq(True))][['isEqualText', 'isSubstring', 'isNgramsSubstring', 'matchWordCount', 'identifier', 'textContent']]
    
    if matched_group['identifier'].any():
        existEqualText = matched_group['isEqualText'].any()
        existSubstring = matched_group['isSubstring'].any()
        existNgramsSubstring = matched_group['isNgramsSubstring'].any()
        # print(f'existEqualText: {existEqualText},\n existSubstring: {existSubstring},\n existNgramsSubstring: {existNgramsSubstring}')
        
        if existEqualText:
            indices = matched_group[matched_group['isEqualText'] == True].index.values[0]
            identifier = matched_group.loc[indices, ['identifier']].values[0]
            textContent = matched_group.loc[indices, ['textContent']].values[0]
        elif existSubstring:
            indices = matched_group[matched_group['isSubstring'] == True].index.values[0]
            identifier = matched_group.loc[indices, ['identifier']].values[0]
            textContent = matched_group.loc[indices, ['textContent']].values[0]
        elif existNgramsSubstring:
            indices = matched_group[matched_group['isNgramsSubstring'] == True]['matchWordCount'].idxmax()
            identifier = matched_group.loc[indices, ['identifier']].values[0]
            textContent = matched_group.loc[indices, ['textContent']].values[0]
        else:
            indices = matched_group['matchWordCount'].idxmax()
            identifier = matched_group.loc[indices, ['identifier']].values[0]
            textContent = matched_group.loc[indices, ['textContent']].values[0]
        result = {
            "identifier": identifier,
            'textContent': textContent
        }
    else:
        result = {
            "identifier": None,
            'textContent': None
        }
    return result