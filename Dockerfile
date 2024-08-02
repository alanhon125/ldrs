FROM python:3.10.11
RUN mkdir /ldrs_analytics
COPY ./packages.txt /
COPY ./requirements.txt /

# install some library files
# install request package
RUN apt-get update -y
# RUN apt-get -y install < packages.txt
RUN apt-get install -y --fix-missing build-essential
RUN apt-get install -y --fix-missing wget
RUN apt-get install -y --fix-missing poppler-utils
RUN apt-get install -y --fix-missing libcap2
RUN apt-get install -y --fix-missing apparmor
RUN apt-get install -y --fix-missing libapparmor1
RUN apt-get install -y --fix-missing argon2
RUN apt-get install -y --fix-missing lvm2
RUN apt-get install -y --fix-missing iptables
RUN apt-get install -y --fix-missing libreoffice
RUN apt-get install -y --fix-missing python3-pil
RUN apt-get install -y --fix-missing tesseract-ocr
RUN apt-get install -y --fix-missing libtesseract-dev
RUN apt-get install -y --fix-missing tesseract-ocr-eng
RUN apt-get install -y --fix-missing tesseract-ocr-script-latn
RUN pip install --no-cache-dir -r /requirements.txt
RUN python -m spacy download en_core_web_md

# RUN useradd -ms /bin/bash admin
COPY . /ldrs_analytics

# define the directory to work in
WORKDIR /ldrs_analytics

EXPOSE 8000

# runs the production server
ENTRYPOINT ["python", "-u", "manage.py"]
CMD ["runserver", "0.0.0.0:8000"]
