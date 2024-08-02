# LDRS

Bank Loan Document Review System

## Getting started

When working with Python, it’s highly recommended to use a virtual environment. This prevents dependency collisions and undesired behaviors that would otherwise arise from premature or unintended version changes. As such, we highly encourage you to set up a virtual environments using conda for the Anaconda Python distribution.

### Check conda is installed and in your PATH

Open a terminal client.

Enter `conda -V` into the terminal command line and press enter.

If conda is installed you should see somehting like the following.

```bash
$ conda -V
conda 3.7.0
```

If not, please install Anaconda Distribution from [here](https://www.anaconda.com/products/distribution).

### Installing requirements and packages for Analytics

```bash
conda create -n boc python=3.10.11
conda activate boc
git clone --branch dev http://10.6.55.124/bigdata/ldrs.git
cd ~/ldrs/analytics/
pip install -r requirements.txt
sudo xargs -a packages.txt apt-get install -y
```

# Starting loan document review system (LDRS) analytics with Docker

To build an application of loan document review system (LDRS) analytics, we start building the docker image according to the `Dockerfile` with:
```bash
cd ~/ldrs/analytics/
docker build -t ldrs_analytics .
```

Here is the `Dockerfile` that powers the image creation.

```
FROM python:3.10.11
RUN mkdir /ldrs_analytics
COPY ./packages.txt /
COPY ./requirements.txt /

# install some library files
# install request package
RUN apt-get -y install < packages.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN python -m spacy download en_core_web_md

COPY . /ldrs_analytics

# define the directory to work in
WORKDIR /ldrs_analytics

EXPOSE 8000

# runs the production server
ENTRYPOINT ["python", "manage.py"]
CMD ["runserver", "0.0.0.0:8000"]
```

Note: Please assure `packages.txt`, `requirements.txt` and `Dockerfile` locate in your current working directory when you build the docker image.

After building the docker image, we can run the container with:
```bash
docker compose up -d
``` 
to start the container in the background and leaves it running, which compose up read `docker-compose.yml` for container setup.

Note: Please assure `config.py`, `regexp.py` and directories `models/` for required NLP models and `data/`for temporary data storage must locate in your current working directory when you start the docker container.

Here is the `docker-compose.yml` that powers the whole setup.

```yaml
services:
  app:
    container_name: ldrs_analytics
    image: ldrs_analytics:latest
    volumes:
      - ${PWD}/config.py:/ldrs_analytics/config.py
      - ${PWD}/regexp.py:/ldrs_analytics/regexp.py
      - ${PWD}/TS_section.csv:/ldrs_analytics/TS_section.csv
      - ${PWD}/data:/ldrs_analytics/data
      - ${PWD}/models:/ldrs_analytics/models
    ports:
    - "<SERVER_IP_ADDRESS>:8000:8000"
```
You may need to specify the server address and port number to host the Django application.


After launching the docker container, we can monitor the service with:
```bash
docker logs ldrs_analytics -f -t
``` 

You may also create a backup that can then be used with docker load with:
```bash
docker save ldrs_analytics > ldrs_analytics.tar
```
and load the docker image with:
```bash
docker load < ldrs_analytics.tar
```
