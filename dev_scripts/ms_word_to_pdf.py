'''
The following script is used to convert MS Words doc into PDF, step involves:
1. Rename capitalized file extension in lower case (e.g. '.DOCX' to '.docx')
2. If the document with file extension '.doc' (the default extension of MS Word 2003), convert it to '.docx' (Microsoft Word 2007 and above) using doc2docx
3. in order to prevent converting list numbering into bitmap during exporting as PDF from MS Word, we need to do following:
    a) rename file extension as .zip from .docx
    b) unarchive the .zip and extract all files in a zip object
    c) delete namespace 'w14:scene3d' from 'word/numbering.xml' and 'word/document.xml' in the zip object
    d) re-zip the altered zip object
    e) rename file extension back to .docx from the altered zip object .zip
4. convert the altered .docx into .pdf using docx2pdf
'''

import os
import json
from pathlib import Path
from zipfile import ZipFile
import shutil

dir_to_words = '/Users/data/Downloads/test'

input_fname = sorted([f for f in os.listdir(dir_to_words) if not f.startswith('~$') and f.endswith(('.doc','.docx','.DOC','.DOCX'))])
output_fname = [f.split('.')[0]+'.pdf' for f in input_fname]
print(json.dumps(output_fname, indent=4))

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


for input_f, output_f in list(zip(input_fname,output_fname)):
    print(f'Converting file: {input_f}')
    input_fpath = os.path.join(dir_to_words,input_f)
    output_fpath = os.path.join(dir_to_words,output_f)
    if not os.path.exists(output_fpath):
        if input_fpath.endswith(('.DOCX', '.DOC')):
            # rename file extension in lowercase instead of uppercase
            ext = input_fpath.split('.')[-1].lower()
            p = Path(input_fpath)
            newfile_path = p.with_suffix(f'.{ext}')
            p.rename(newfile_path)
            input_fpath = str(newfile_path)

        # if file extension is '.doc' or '.DOC', save as '.docx' first
        if input_fpath.endswith(('.doc','.DOC')):
            from doc2docx import convert
            convert(input_fpath, input_fpath.split('.')[0]+'.docx', keep_active=True)
            os.remove(input_fpath) # remove document with file extension '.doc'
            input_fpath = input_fpath.split('.')[0]+'.docx'

        # Codes below follow the step in https://superuser.com/questions/837529/preventing-word-from-converting-heading-numbers-into-images-during-pdf-export
        # in order to prevent converting list numbering into bitmap during exporting as PDF from MS Word

        # rename file extension as .zip instead of .docx
        p = Path(input_fpath)
        zipfile_path = p.with_suffix('.zip')
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
        output_zip = os.path.join(dir_to_words, unarchive_dir)
        shutil.make_archive(output_zip, 'zip', unarchive_dir)

        # remove the by-product, i.e. unarchive directory
        shutil.rmtree(unarchive_dir)

        # Again, rename file extension as .docx instead of .zip
        p2 = Path(zipfile_path)
        docxfile_path = p.with_suffix('.docx')
        p2.rename(docxfile_path)

        assert str(docxfile_path) == input_fpath, 'please check output path'

        from docx2pdf import convert
        # convert .docx into .pdf
        convert(input_fpath, output_fpath, keep_active=True)
