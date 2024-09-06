# coding=utf8
import os,pymysql

# 配置数据库
mysql_host = os.getenv('base_mysql_host')
mysql_port = int(os.getenv('base_mysql_port'))
mysql_user = os.getenv('base_mysql_user')
mysql_pwd = os.getenv('base_mysql_pwd')
mysql_db = os.getenv('base_qinglong_db')

mysql_config = {
    'host': mysql_host,
    'port': mysql_port,
    'user': mysql_user,
    'password': mysql_pwd,
    'database': mysql_db
}

def start_connection():
    
    return pymysql.connect(**mysql_config)

def close_connection(connection_ins):
    
    if connection_ins:
        connection_ins.close()

def write_many(connection, sql:str, data_list: list):
    
    try:
        with connection.cursor() as cursor:
            cursor.executemany(sql,data_list)
            connection.commit()

    except Exception as e:
        print(e)