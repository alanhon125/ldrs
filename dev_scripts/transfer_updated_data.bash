REVIEW_ANTD_TS_VER="_v3.1"
TS_VER="_v4"
FA_VER="_v4"
SENT_PAIRS_VER="_v9"

# After perform docparsing in 10.6.55.126, use this to transfer docparse JSON and CSV from 10.6.55.126

# sshpass -p "SNDSdata" scp -r data@10.6.55.126:/home/data1/public/ResearchHub/esg_demo/esg-analytics/data/docparse_json/TS /home/data/ldrs_analytics/data/docparse_json
# sshpass -p "SNDSdata" scp -r data@10.6.55.126:/home/data1/public/ResearchHub/esg_demo/esg-analytics/data/docparse_csv/TS/* /home/data/ldrs_analytics/data/docparse_csv/TS${TS_VER}
# sshpass -p "SNDSdata" scp -r data@10.6.55.126:/home/data1/public/ResearchHub/esg_demo/esg-analytics/data/docparse_json/FA /home/data/ldrs_analytics/data/docparse_json
# sshpass -p "SNDSdata" scp -r data@10.6.55.126:/home/data1/public/ResearchHub/esg_demo/esg-analytics/data/docparse_csv/FA/* /home/data/ldrs_analytics/data/docparse_csv/FA${FA_VER}
# sshpass -p "SNDSdata" scp -r data@10.6.55.126:/home/data/ResearchHub/esg_demo/esg-analytics/data/pdf/BoC/annotated_ts /home/data/ldrs_analytics/data/pdf
# sshpass -p "SNDSdata" scp -r data@10.6.55.126:/home/data/ResearchHub/esg_demo/esg-analytics/data/pdf/BoC/unannotated_fa_ts /home/data/ldrs_analytics/data/pdf

# Update data from 10.6.55.12 to alan's local laptop 10.6.210.53

# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/data/pdf data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics/data
# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/data/docparse_json data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics/data
# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/data/docparse_csv data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics/data
# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/data/doc/TS data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics/data/doc
# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/data data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/models data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/ldrs_analytics.tar data@10.6.208.130:/Users/data/Library/CloudStorage/OneDrive-HongKongAppliedScienceandTechnologyResearchInstituteCompanyLimited/docker_images
# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/ldrs_analytics_model.tar.gz data@10.6.208.130:/Users/data/Library/CloudStorage/OneDrive-HongKongAppliedScienceandTechnologyResearchInstituteCompanyLimited/docker_images
# sshpass -p "1q2w3e4r" rsync -r data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics/ldrs_analytics.tar /home/data/ldrs_analytics/ldrs_analytics.tar

# Update codes from 10.6.55.12 to alan's local laptop 10.6.210.53

sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/boc data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/app data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/packages.txt data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/requirements.txt data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/environment.yml data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/db.sqlite3 data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/.gitignore data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/.dockerignore data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/Dockerfile data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/README.md data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/regexp.py data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/dev_scripts data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/TS_section.csv data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/manage.py data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics
sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/docker-compose.yml data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics

# sshpass -p "1q2w3e4r" rsync -r /home/data/ldrs_analytics/config.py data@10.6.208.130:/Users/data/Documents/GitHub/ldrs/analytics

# Update codes from 10.6.55.12 to alan's local desktop 172.16.24.143

# sshpass -p "4026" scp -r /home/data/ldrs_analytics/boc alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/app alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/packages.txt alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/requirements.txt alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/environment.yml alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/db.sqlite3 alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/.gitignore alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/.dockerignore alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/Dockerfile alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/README.md alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/config.py alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/regexp.py alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/dev_scripts alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/TS_section.csv alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/manage.py alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/docker-compose.yml alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics

# sshpass -p "4026" scp -r /home/data/ldrs_analytics/models alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics
# sshpass -p "4026" scp -r /home/data/ldrs_analytics/data alan@172.16.24.143:/Users/alan/Documents/GitHub/ldrs/analytics

# Transfer sentence-pair dataset from 10.6.55.12 to 10.6.55.2

# sshpass -p "1q2w3e4r" scp -r /home/data/ldrs_analytics/data/antd_ts_merge_fa/merge/*sentence*.csv data@10.6.55.2:/home/data/data/alan/sentence-transformers/examples/training/nli/datasets
# sshpass -p "1q2w3e4r" scp /home/data/ldrs_analytics/data/antd_ts_merge_fa/merge/all_merge_results.csv data@10.6.55.2:/home/data/data/alan/sentence-transformers/examples/training/nli/datasets
# sshpass -p "1q2w3e4r" scp /home/data/ldrs_analytics/data/antd_ts_merge_fa/merge${FA_VER}/sentence_pairs${SENT_PAIRS_VER}.csv data@10.6.55.2:/home/data/data/owen/boc_data
