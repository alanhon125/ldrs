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
    # import cpuinfo
    
    def contains_explicit_return(function):
        # Check if function has return statement
        import ast
        import inspect
        
        return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(function))))

    # processor_name = cpuinfo.get_cpu_info()["brand_raw"]
    input_length = len(input_list)
    num_processor = multiprocessing.cpu_count()
    batch_size = max(input_length // num_processor, 1)
    num_batch = int(input_length / batch_size) + (input_length % batch_size > 0)
    print(f'CPU cores count: {num_processor}\nList length: {input_length}\nBatch size: {batch_size}\nBatch no.: {num_batch}') # Processor: {processor_name}\n
    pool = multiprocessing.Pool(num_processor)
    if not args:
        processes = [pool.apply_async(function, args=(input_list[idx * batch_size:(idx + 1) * batch_size],)) for idx in range(num_batch)]
    else:
        processes = [pool.apply_async(function, args=(input_list[idx * batch_size:(idx + 1) * batch_size], args)) for idx in range(num_batch)]
    
    if contains_explicit_return(function): # if function has return statement
        results = [p.get() for p in processes]
        results = [i for r in results if isinstance(r,list) or isinstance(r,tuple) for i in r]
        
        return results
    else:
        for p in processes:
            p.get()

def log_task2csv(csv_name, row):
    '''
    log the record 'row' into path 'csv_name'
    '''
    import csv
    # Open the CSV file in "append" mode
    with open(csv_name, 'a', newline='') as f:
        # Create a dictionary writer with the dict keys as column fieldnames
        writer = csv.DictWriter(f, fieldnames=row.keys())
        # Append single row to CSV
        writer.writerow(row)


def get_filepath_with_filename(parent_dir, filename):
    '''
    Given the target filename and a parent directory, traverse file system under the parent directory to lookup for the filename
    return the relative path to the file with filename if match is found
    otherwise raise a FileNotFoundError
    '''
    import errno
    import os

    for dirpath, subdirs, files in os.walk(parent_dir):
        for x in files:
            if x in filename:
                return os.path.join(dirpath, x)
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)


def remove_punctuation(s):
    '''
    Remove all punctuations in a string
    '''
    import string
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator).strip() if s else None

def remove_particular_punct(s):
    '''
    Remove particular punctuations in a string
    '''

    import string
    import re

    keep_punct = ['(', ')', ':', ',', '.','\\','$']
    punct_list = string.punctuation
    remove_punct = [re.escape(i) for i in punct_list.translate({ord(c): None for c in keep_punct})]
    s = s.replace('\r\n', ' ')
    cleaned_text = re.sub('|'.join(remove_punct), '', s)
    cleaned_text = re.sub(r'(:|\(|\)|,| )\1+', r'\1', cleaned_text)

    return cleaned_text

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

def filter_invalid_char(my_str):
    # To remove invalid characters from a string
    import re
    my_str = re.sub(' +', ' ', my_str)
    my_str = re.sub('','', my_str)
    my_str = re.sub(r' ([\.,;]) ',r'\1 ',my_str)
    my_str = re.sub(r' ([\)\]])', r'\1', my_str)
    my_str = re.sub(r' ([\(\[]) ',r' \1',my_str)
    my_str = re.sub(r'([\u4e00-\u9fff]+)', '', my_str) # Chinese
    my_str = re.sub(r'([\u25a0-\u25ff]+)', '', my_str) # geometric shape
    my_str = re.sub(r'([\uff01-\uff5e]+)', '', my_str) # fullwidth ascii variants
    my_str = re.sub(r'([\u2018\u2019\u201a-\u201d]+)', '', my_str)  # Quotation marks and apostrophe
    return my_str.strip()

def discard_str_with_unwant_char(a_str):
    # To discard string with Chinese characters, Geometric shape and fullwidth ascii variant characters
    import re
    # The 4E00—9FFF range covers CJK Unified Ideographs (CJK=Chinese, Japanese and Korean)
    ChineseRegexp = re.compile(r'[\u4e00-\u9fff]+')
    # geometric_shapeRegexp = re.compile(r'[\u25a0-\u25ff]+')
    fullwidth_ascii_variantsRegexp = re.compile(r'[\uff01-\uff5e]+')
    if ChineseRegexp.search(a_str) or fullwidth_ascii_variantsRegexp.search(a_str): # geometric_shapeRegexp.search(a_str)
        return None
    else:
        return filter_invalid_char(a_str)

def remove_articles(text):
    import re
    return re.sub('(a|an|and|the)(\s+)', '\2', text, flags=re.I).strip('\x02')

def stem_identifier(identifier, ps):
    '''
    To stem the words inside identifiers, e.g:
    Cl_1.1-Materials Subsidiaries_(a) -> Cl_1.1-Material Subsidiari_(a)
    Cl_1.1-Material Subsidiary_(a) -> Cl_1.1-Material Subsidiari_(a)
    @param identifier: facility agreement identifier with naming convention:
        1. Cl_<SECTION_ID>.<SUB-SECTION_ID>-<DEFINITION_OR_PARTIES>_<LIST_ITEM_ID>
        2. Sched_<ID>.<PART_ID>_<SECTION_ID>_<LIST_ITEM_ID>
        3. Parties-<PARTY_ROLE_NAME>
    @type identifier: str
    @param ps: PorterStemmer from NLTK
    @type ps: nltk.stem.PorterStemmer()
    @rtype: str
    '''
    new_str = ''

    if identifier:
        underscore_split = identifier.split('_')
        hyphen_split = identifier.split('-')
        suffix = None

        if '-' in identifier:
            if len(hyphen_split) == 2:
                prefix = hyphen_split[0]
                words = hyphen_split[1]
            else:
                words = identifier
                prefix = None
        else:
            if len(underscore_split) == 3:
                prefix = underscore_split[0]
                words = underscore_split[1]
                suffix = underscore_split[-1]
            elif len(underscore_split) == 2:
                prefix = None
                words = underscore_split[0]
                suffix = underscore_split[1]
            else:
                prefix = None
                words = identifier
                suffix = None

        words = remove_articles(words)
        words = words.split()

        if prefix:
            prefix = ps.stem(prefix)
        if suffix:
            suffix = ps.stem(suffix)

        new_words = ''
        for i, word in enumerate(words):
            rootWord = ps.stem(word)

            if i < len(words) - 1:
                new_words += rootWord + ' '
            else:
                new_words += rootWord

        if len(underscore_split) == 3:
            if prefix and suffix:
                new_str = prefix + '_' + new_words + '_' + suffix
            elif suffix:
                new_str = new_words + '_' + suffix
            else:
                new_str = new_words
        else:
            if '-' in identifier and prefix and suffix:
                new_str = prefix + '-' + new_words + '_' + suffix
            elif prefix and suffix:
                new_str = prefix + '_' + new_words + '_' + suffix
            elif suffix:
                new_str = new_words + '_' + suffix
            else:
                new_str = new_words
    return new_str

def stem_string(words, ps):
    words = words.split(' ')
    new_words = ''
    for i, word in enumerate(words):
        rootWord = ps.stem(word)
        if i<len(words)-1:
            new_words += rootWord + ' '
        else:
            new_words += rootWord
    return new_words