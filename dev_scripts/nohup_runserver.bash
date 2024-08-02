kill -9 $(lsof -t -i:8000)
# calling conda source activate boc from bash script
eval "$(conda shell.bash hook)"
source activate
conda activate boc
cd /home/data/ldrs_analytics
rm /home/data/ldrs_analytics/data/log/syslog_port8000.log
nohup python manage.py runserver 0.0.0.0:8000 > data/log/syslog_port8000.log 2>&1 &