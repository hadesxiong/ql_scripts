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

def start_custom_connection(mysql_config: dict):

    try:
        mysql_conf = {
            'host': mysql_config['host'],
            'port': mysql_config['port'],
            'user': mysql_config['user'],
            'password': mysql_config['password'],
            'database': mysql_config['database']
        }

        return pymysql.connect(**mysql_conf)
    
    except Exception as e:
        print(e)    

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

def fetch_all(connection, sql:str, data_list: list):

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql,data_list)
            columns = [col[0] for col in cursor.description]
            results = cursor.fetchall()
            structured_results = [dict(zip(columns,row)) for row in results]
            return structured_results
        
    except Exception as e:
        print(e)


def fetch_one(connection, sql:str, data_list: list):

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql,data_list)
            columns = [col[0] for col in cursor.description]
            results = cursor.fetchone()
            if results:
                structured_result = dict(zip(columns, results))
                return structured_result
            else:
                return None
        
    except Exception as e:
        print(e)