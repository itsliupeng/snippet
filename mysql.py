import json
import pymysql


"""
CREATE DATABASE bigdata;

USE bigdata;

CREATE TABLE wechat (
    hash_id BIGINT NOT NULL PRIMARY KEY,
    content mediumtext NOT NULL,
    title mediumtext NULL,
    cluster_id INT NOT NULL,
    source INT NOT NULL
);

CREATE TABLE falcon_1 (
    hash_id BIGINT NOT NULL PRIMARY KEY,
    content mediumtext NOT NULL,
    url mediumtext NULL,
    cluster_id INT NOT NULL,
    source INT NOT NULL
);




"""
# 数据库连接参数
db_params = {
    'host': 'localhost',
    'user': 'root',
    # 'password': 'mysql',
    'database': 'bigdata',
    'charset': 'utf8mb4'  # 使用这个字符集以支持所有的 Unicode 字符
}


# CREATE USER 'root'@'localhost' IDENTIFIED WITH authentication_plugin BY 'mysql';

# CREATE USER 'root'@'localhost' IDENTIFIED BY 'mysql';

# 连接到数据库
connection = pymysql.connect(**db_params)
cursor = connection.cursor()

# 从文件读取数据
total = 0
# for line in open('/lp/soft/mysql-files/cc_id_content.json', 'r'):
#     data = json.loads(line)
#     hash_id = data['hash_id']
#     content = data['content'].encode("utf-8")
#
#     # 插入数据到数据库
#     query = """
#     INSERT INTO mytable (hash_id, content) VALUES (%s, %s)
#     ON DUPLICATE KEY UPDATE content = VALUES(content)
#     """
#     cursor.execute(query, (hash_id, content))
#
#     total += 1
#
#     if total % 10000 == 0:
#         # 提交更改
#         connection.commit()
#         print(f"num: {total}")


# 关闭自动提交
connection.autocommit(False)

# 预编译SQL语句
query = """
INSERT INTO mytable (hash_id, content) VALUES (%s, %s)
ON DUPLICATE KEY UPDATE content = VALUES(content)
"""

# 批量插入数据
values = []

for line in open('/lp/soft/mysql-files/cc_id_content.json', 'r'):
    data = json.loads(line)
    hash_id = data['hash_id']
    content = data['content'].encode("utf-8")

    values.append((hash_id, content))

    total += 1

    if total % 10000 == 0:
        cursor.executemany(query, values)
        connection.commit()
        print(f"num: {total}")
        values = []

# 插入剩余的数据
if values:
    cursor.executemany(query, values)
    connection.commit()

# 重新开启自动提交（如果需要）
connection.autocommit(True)

cursor.close()
connection.close()


#######################################



import concurrent.futures
import pymysql
import json

def insert_data(chunk):
    connection = pymysql.connect(
        host='localhost',
        user='your_username',
        password='your_password',
        db='your_dbname',
        charset='utf8mb4'
    )
    cursor = connection.cursor()

    query = """
    INSERT INTO mytable (hash_id, content) VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE content = VALUES(content)
    """

    values = [(data['hash_id'], data['content'].encode("utf-8")) for data in chunk]
    cursor.executemany(query, values)
    connection.commit()

    cursor.close()
    connection.close()

chunks = []
chunk_size = 10000
with open('/lp/soft/mysql-files/cc_id_content.json', 'r') as f:
    chunk = []
    for line in f:
        if len(chunk) >= chunk_size:
            chunks.append(chunk)
            chunk = []
        data = json.loads(line)
        chunk.append(data)
    if chunk:
        chunks.append(chunk)

# 使用线程池并发插入数据
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(insert_data, chunks)

















# 查询

with connection.cursor() as cursor:
    sql = "SELECT * FROM mytable LIMIT 10;"
    cursor.execute(sql)
    results = cursor.fetchall()

for row in results:
    print(row)