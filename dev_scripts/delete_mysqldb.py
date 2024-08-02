import mysql.connector
from mysql.connector import Error
from datetime import datetime
from dateutil import tz
import pandas as pd

# METHOD 2: Auto-detect zones:
from_zone = tz.tzutc()
to_zone = tz.tzlocal()

df_facility_type = pd.read_csv('/home/data/ldrs_analytics/data/doc/filename_facility_type.csv')
TS_files = [
        "125_VF_SYN_TS_mkd_20231013.pdf",
        "126_VF_SYN_TS_mkd_20230601.pdf",
        "127_NBFI_SYN_TS_mkd_20230330.pdf",
        "128_NBFI_SYN_TS_mkd_20230224.pdf",
        "129_AW_SYN_TS_mkd_20230626.pdf",
        "130_BL_SYN_TS_mkd_20230710.pdf",
        "131_AF_SYN_TS_mkd_20230718.pdf",
        "132_PF_SYN_TS_mkd_20230106.pdf",
        "133_BL_PRJ_TS_mkd_20230926.pdf",
        "134_GF_PRJ_TS_mkd_20230829.pdf",
        "135_GL_PRJ_TS_mkd_20231004.pdf",
        "136_BL_SYN_TS_mkd_20230418.pdf",
        "137_GL_SYN_TS_mkd_20231121.pdf",
        "138_GF_SYN_TS_mkd_20230130.pdf",
        "139_NBFI_PRJ_TS_mkd_20231122.pdf",
        "140_NBFI_SYN_TS_mkd_20230512.pdf",
        "141_GL_SYN_TS_mkd_20231221.pdf",
        "142_PF_SYN_TS_mkd_20230810.pdf",
        "143_GF_SYN_TS_mkd_undated.pdf",
        "144_NBFI_SYN_TS_mkd_20231031.pdf",
        "145_GF_SYN_TS_mkd_20231031.pdf",
        "146_BL_PRJ_TS_mkd_20230629.pdf",
        "147_GL_PRJ_TS_mkd_20230817.pdf",
        "148_GF_PRJ_TS_mkd_20230919.pdf",
        "149_BL_PRJ_TS_mkd_20231102.pdf"
    ]


queries = [
    'delete from tbl_feedback',
    'ALTER TABLE tbl_feedback AUTO_INCREMENT = 1',
    'delete from tbl_term_matching_data',
    'ALTER TABLE tbl_term_matching_data AUTO_INCREMENT = 1',
    # 'delete from tbl_fa_data',
    # 'ALTER TABLE tbl_fa_data AUTO_INCREMENT = 1',
    # 'delete from tbl_ts_data',
    # 'ALTER TABLE tbl_ts_data AUTO_INCREMENT = 1',
    # 'delete from tbl_task',
    # 'ALTER TABLE tbl_task AUTO_INCREMENT = 1',
    # 'delete from tbl_document',
    # 'ALTER TABLE tbl_document AUTO_INCREMENT = 1',
    # 'UPDATE tbl_document SET file_status = "PROCESSED"',
    # 'UPDATE tbl_document SET file_status = "WAITING_FOR_PROCESSING" WHERE file_type = "FA" and file_name REGEXP "^[0-9]+"',
    # "SELECT id FROM tbl_document where file_name IN ('4_GF_SYN_FA_mkd_20221008.pdf','4_GF_SYN_TS_mkd_20221008.pdf')"
]

try:
    connection = mysql.connector.connect(host='10.6.55.12',
                                         database='boc',
                                         user='root',
                                         password='1q2w3e4r',
                                         port='30306')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        
    # for query in queries:
    for TS_file in TS_files:
        abbr = '_'.join(TS_file.split('_')[:3])
        facility_type = df_facility_type[df_facility_type['filename']==TS_file.replace('.pdf','')]['facility_type'].values[0]
        FA_file = TS_file.replace('TS','FA')
        query = f'SELECT id from tbl_document where file_name IN ("{FA_file}", "{TS_file}") and create_user = "alan"'
        cursor = connection.cursor(buffered=True)
        cursor.execute(query)
        ids = [i[0] for i in cursor.fetchall()]
        print(f"'{query}'operated successfully.")
        utc = datetime.utcnow()

        # Tell the datetime object that it's in UTC time zone since 
        # datetime objects are 'naive' by default
        utc = utc.replace(tzinfo=from_zone).astimezone(to_zone).strftime('%Y-%m-%d %H:%M:%S')
        insert_statement = f'INSERT INTO tbl_task (name, customer_no, borrower_name, caw_proposal_no, maker_group, checker_group, fa_file_id, ts_file_id, status, create_time, create_user, update_time, update_user, del_flag, system_status, facility_type) VALUES ("{abbr}", "alan", "alan", "alan", "maker_group_01", "checker_group_01", {ids[0]}, {ids[1]}, "CREATING", "{utc}", "alan", "{utc}", "alan", 0, "PROCESSED", "{facility_type}")'
        print(insert_statement)
        cursor.execute(insert_statement)
        
except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")