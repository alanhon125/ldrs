import json
import os
import pathlib
import re
import sys
import logging
import subprocess

try:
    from config import (
        DOC_DIR,
        LOG_DIR,
        PDF_DIR,
        LOG_FILEPATH
    )
    from app.utils import log_task2csv
except ModuleNotFoundError:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    currentdir = os.path.dirname(os.path.abspath(__file__))
    parentdir = os.path.dirname(currentdir)
    parentparentdir = os.path.dirname(parentdir)
    sys.path.insert(0, parentdir)
    from utils import log_task2csv

    sys.path.insert(0, parentparentdir)
    from config import (
        DOC_DIR,
        LOG_DIR,
        PDF_DIR,
        LOG_FILEPATH
    )

logger = logging.getLogger(__name__)


# logging.basicConfig(
#     filename=LOG_FILEPATH,
#     filemode='a',
#     format='【%(asctime)s】【%(filename)s:%(lineno)d】【%(levelname)-8s】%(message)s',
#     level=os.environ.get("LOGLEVEL", "INFO"),
#     datefmt='%Y-%m-%d %H:%M:%S'
#     )

def ms_words_convert_to_pdf(folder, source, fileExtension=None, timeout=None):
    import os
    from docx2pdf import convert

    assert fileExtension in ['pdf'], logger.error('docx2pdf converter only support file conversion from .doc, .docx to .pdf')

    output_fname = os.path.basename(source.split('.')[0] + f'.{fileExtension}')
    output_fpath = os.path.join(folder, output_fname)
    if not os.path.exists(output_fpath):
        convert(source, output_fpath, keep_active=True)

def libreoffice_exec():
    # TODO: Provide support for more platforms
    if sys.platform == 'darwin':
        return '/Applications/LibreOffice.app/Contents/MacOS/soffice'
    return 'libreoffice'

def libreOffice_convert_to_pdf(folder, source, fileExtension=None, timeout=None):
    assert fileExtension in ['pdf', 'html', 'txt', 'docx',
                             None], logger.error('LibreOffice converter only support file conversion from .doc to .pdf, .html, .txt')
    ext = fileExtension
    if not fileExtension:
        fileExtension = 'pdf:writer_pdf_Export'
    elif fileExtension == 'html':
        fileExtension += ":XHTML Writer File:UTF8"
    elif fileExtension == 'pdf':
        fileExtension += ':calc_pdf_Export'
    elif fileExtension == 'txt':
        fileExtension += ":Text (encoded):UTF8"
    elif fileExtension == 'docx':
        fileExtension += ':MS Word 2007 XML:UTF8'

    args = [libreoffice_exec(), '--headless', '--language=zh_HK.utf8', '--convert-to', fileExtension, '--outdir', folder, source]
    source_ext = source.split('.')[-1]
    if source_ext == 'docx':
        args += ['--infilter="Microsoft Word 2007/2010/2013 XML"']
    filename = os.path.basename(source).split('.')[0] + '.' + ext
    outpath = os.path.join(folder, filename)
    if os.path.exists(outpath):
        return outpath

    try:
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.CalledProcessError as e:
        logger.exception(e)
        try:
            args = ['/usr/bin/libreoffice', '--headless', '--language=zh_HK.utf8', '--convert-to', fileExtension, '--outdir',
            folder, source, '--infilter="Microsoft Word 2007/2010/2013 XML"']
            subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.CalledProcessError as e:
            logger.exception(e)

    # filename = re.search('-> (.*?) using filter', process.stdout.decode())
    component_attempts = [':calc_pdf_Export',':impress_pdf_Export',':writer_web_pdf_Export',':draw_pdf_Export']
    while not os.path.exists(outpath) and ext == 'pdf' and source_ext == 'docx' and component_attempts:
        writer = component_attempts.pop(0)
        logger.error(f"Fail to make document conversion from {source} to {outpath}. Attempt to use writer【{writer.lstrip(':')}】for conversion.")
        fileExtension = ext + writer
        args = [libreoffice_exec(), '--headless', '--language=zh_HK.utf8', '--convert-to', fileExtension, '--outdir',
            folder, source, '--infilter="Microsoft Word 2007/2010/2013 XML"']
        try:
            subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.CalledProcessError as e:
            logger.exception(e)
            try:
                args = ['/usr/bin/libreoffice', '--headless', '--language=zh_HK.utf8', '--convert-to', fileExtension, '--outdir',
                folder, source, '--infilter="Microsoft Word 2007/2010/2013 XML"']
                subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            except subprocess.CalledProcessError as e:
                logger.exception(e)
        
    if not os.path.exists(outpath):
        logger.exception(f"Output file {outpath} doesn't exist. Fail to make document conversion from {source} to {outpath}")
        raise FileExistsError(f"Output file {outpath} doesn't exist. Fail to make document conversion from {source} to {outpath}")
    else:
        return outpath

class LibreOfficeError(Exception):
    def __init__(self, output):
        self.output = output


if __name__ == "__main__":

    key_ts = ['TS', 'term sheet']
    key_fa = ['FA', 'facility agreement', 'facilities agreement']
    
    for dirpath, subdirs, files in os.walk(DOC_DIR):
        for x in files:
            fpath = os.path.join(dirpath, x)
            fname = os.path.basename(fpath)
            ext = pathlib.Path(fpath).suffixes
            if ext and re.match(r'.docx|.doc|.DOCX|.DOC', ext[0], flags=re.IGNORECASE):
                filename = os.path.basename(fpath).split('.')[0] + '.pdf'
                # if os.path.basename(fpath) in completed_list:
                #     print(f'{os.path.basename(fpath)} has already been converted.')
                #     continue
                logger.info(f'converting file: {os.path.basename(fpath)}')
                # search 'TS' or 'FA' from filename to distinguish the result doc type, and store in corresponding subfolder
                if re.match(r'.*' + r'.*|.*'.join(key_ts), fname, flags=re.IGNORECASE):
                    sub_folder = 'TS/'
                elif re.match(r'.*' + r'.*|.*'.join(key_fa), fname, flags=re.IGNORECASE):
                    sub_folder = 'FA/'
                else:
                    sub_folder = ''
                # ms_words_convert_to_pdf(PDF_DIR, fpath)
                libreOffice_convert_to_pdf(PDF_DIR, fpath)


