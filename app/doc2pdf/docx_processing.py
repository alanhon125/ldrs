'''
The following script is used to convert MS Words doc into PDF, step involves:
1. Rename capitalized file extension in lower case (e.g. '.DOCX' to '.docx')
(not applicable in Linux) 2. If the document with file extension '.doc' (the default extension of MS Word 2003), convert it to '.docx' (Microsoft Word 2007 and above) using doc2docx
3. in order to prevent converting list numbering into bitmap during exporting as PDF from MS Word, we need to do following:
    a) rename file extension as .zip from .docx
    b) unarchive the .zip and extract all files in a zip object
    c) delete namespace 'w14:scene3d' from 'word/numbering.xml' and 'word/document.xml' in the zip object
    d) re-zip the altered zip object
    e) rename file extension back to .docx from the altered zip object .zip
(not applicable in Linux) 4. convert the altered .docx into .pdf using docx2pdf
'''
from config import LOG_FILEPATH
import logging
import os

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename=LOG_FILEPATH,
#     filemode='a',
#     format='【%(asctime)s】【%(filename)s:%(lineno)d】【%(levelname)-8s】%(message)s',
#     level=os.environ.get("LOGLEVEL", "INFO"),
#     datefmt='%Y-%m-%d %H:%M:%S'
#     )

def delete_namespace(my_namespace, file):
    # parse xml file and delete a particular namespace in the xml file, save and overwrite original xml file

    import lxml.etree as le
    from lxml import etree

    '''
    A value that is the correct lexical form of The xs:QName.
    QName data type has the xs:string data type.
    fn is the namespace prefix
    '''
    fn, QName = my_namespace.split(':')

    with open(file, 'r') as f:
        doc = le.parse(f)

        # The namespace portion of an expanded name.
        # URI has the xs:string data type, or is an empty string or sequence.
        URI = doc.xpath(f'//*')[0].nsmap[fn]
        nsmap = {fn: URI}
        elements = doc.xpath(f'//{my_namespace}', namespaces=nsmap)
        for elem in elements:
            parent = elem.getparent()
            parent.remove(elem)

    doc.write(file, pretty_print=True)

def lowercase_file_ext(input_fpath):
    '''
    lower-casing the file extension
    '''
    from pathlib import Path
    
    # rename file extension in lowercase instead of uppercase
    ext = input_fpath.split('.')[-1].lower()
    p = Path(input_fpath)
    newfile_path = p.with_suffix(f'.{ext}')
    p.rename(newfile_path)
    input_fpath = str(newfile_path)
    
    return input_fpath

def docx_preprocess(input_fpaths):
    import os
    from pathlib import Path
    from zipfile import ZipFile
    import shutil
    
    if isinstance(input_fpaths,str):
        input_fpaths = [input_fpaths]
        
    for input_fpath in input_fpaths:
        assert input_fpath.endswith(('.docx','.DOCX')), logger.error('please aware that document preprocessing only support Microsoft Word 2007 and above document, i.e. .docx file type')
        logger.info(f'Converting file: {input_fpath}')
        input_dir = os.path.dirname(input_fpath)
        output_fpath = input_fpath.split('.')[0]+'.pdf'
        if not os.path.exists(output_fpath):
            if input_fpath.endswith(('.DOCX', '.DOC')):
                # rename file extension in lowercase instead of uppercase
                input_fpath = lowercase_file_ext(input_fpath)

            # if file extension is '.doc' or '.DOC', save as '.docx' first (Convert doc to docx on Windows or macOS directly using Microsoft Word (must be installed))
            # if input_fpath.endswith(('.doc','.DOC')):
            #     from doc2docx import convert
            #     convert(input_fpath, input_fpath.split('.')[0]+'.docx', keep_active=True)
            #     os.remove(input_fpath) # remove document with file extension '.doc'
            #     input_fpath = input_fpath.split('.')[0]+'.docx'

            # Codes below follow the step in https://superuser.com/questions/837529/preventing-word-from-converting-heading-numbers-into-images-during-pdf-export
            # in order to prevent converting list numbering into bitmap during exporting as PDF from MS Word

            # rename file extension as .zip instead of .docx
            p = Path(input_fpath)
            zipfile_path = os.path.abspath(p.with_suffix('.zip'))
            p.rename(zipfile_path)

            # loading the .zip and extract all files in a zip object
            with ZipFile(zipfile_path, 'r') as zObject:

                # Extracting all files in the zip into a specific location.
                unarchive_dir = str(zipfile_path).replace('.zip','')
                zObject.extractall(unarchive_dir)
                zObject.close()

            # target xml to be revised
            numbering_xml_path = os.path.join(unarchive_dir, 'word/numbering.xml')
            document_xml_path = os.path.join(unarchive_dir, 'word/document.xml')

            # delete namespace 'w14:scene3d' from 'word/numbering.xml' and 'word/document.xml' in the zip object
            delete_namespace('w14:scene3d', numbering_xml_path)
            delete_namespace('w14:scene3d', document_xml_path)

            # re-zip the altered zip object
            output_zip = os.path.join(input_dir, unarchive_dir)
            shutil.make_archive(output_zip, 'zip', unarchive_dir)

            # remove the by-product, i.e. unarchive directory
            shutil.rmtree(unarchive_dir)

            # Again, rename file extension as .docx instead of .zip
            p2 = Path(zipfile_path)
            docxfile_path = os.path.abspath(p.with_suffix('.docx'))
            p2.rename(docxfile_path)

            assert str(docxfile_path) == os.path.abspath(input_fpath), logger.error(f'Final converted document is with path {str(docxfile_path)}, which is not the same as original input path {input_fpath}. Please check.')

            # from docx2pdf import convert
            # # convert .docx into .pdf
            # convert(input_fpath, output_fpath, keep_active=True)

if __name__ == "__main__":
    import os
    import argparse
    try:
        from app.utils import multiprocess
        from config import DOC_DIR
    except ImportError:
        import sys
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        currentdir = os.path.dirname(os.path.abspath(__file__))
        parentdir = os.path.dirname(currentdir)
        parentparentdir = os.path.dirname(parentdir)
        sys.path.insert(0, parentdir)
        from utils import multiprocess
        sys.path.insert(0, parentparentdir)
        from config import DOC_DIR
    
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dir_to_words",
        default=f'{DOC_DIR}/FA',
        type=str,
        required=False,
        help="The input MS Words files directory or path to a single MS Words file. If it is a directory, it should contain the MS Words files.",
    )
    parser.add_argument(
        "--target_file_ext",
        default=('.docx','.DOCX'),
        type=tuple,
        required=False,
        help="The target MS Words file extension that accepted for conversion.",
    )

    args = parser.parse_args()
    
    # checks if path is a file
    isFile = os.path.isfile(args.dir_to_words)
    # checks if path is a directory
    isDirectory = os.path.isdir(args.dir_to_words)

    if isDirectory:
        input_fpaths = sorted([os.path.join(args.dir_to_words,f) for f in os.listdir(args.dir_to_words) if not f.startswith('~$') and f.endswith(args.target_file_ext)])
        multiprocess(docx_preprocess, input_fpaths)
    elif isFile:
        input_fpath = args.dir_to_words
        docx_preprocess(input_fpath)