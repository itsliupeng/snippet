import pymysql
import json
from multiprocessing import Process, Queue
import os
cpu_count = 32

CHUNK_NUM = 10000

db_params = {
    'host': 'localhost',
    'user': 'liupeng',
    'password': 'mysql',
    'database': 'bigdata',
    'charset': 'utf8mb4'  # 使用这个字符集以支持所有的 Unicode 字符
}

def producer(queue, file_list):
    for file_path in file_list:
        with open(file_path, 'r') as f:
            chunk = []
            for line in f:
                data = json.loads(line)
                chunk.append(data)
                if len(chunk) >= CHUNK_NUM:
                    queue.put(chunk)
                    chunk = []
            if chunk:
                queue.put(chunk)
    queue.put(None)  # Sentinel value to signal the end


def consumer(queue):
    connection = pymysql.connect(**db_params)
    cursor = connection.cursor()

    query = f"INSERT INTO falcon_1 (hash_id, content, url, cluster_id, source) VALUES (%s, %s, %s, %s, {SOURCE}) " \
            f"ON DUPLICATE KEY UPDATE content = VALUES(content)"
    count = 0
    while True:
        chunk = queue.get()
        if chunk is None:  # Check for sentinel value
            break
        values = [(data['hash_id'], data['content'].encode("utf-8"), data['url'], data['prediction']) for data in chunk]
        cursor.executemany(query, values)
        connection.commit()
        count += len(chunk)
        if count % 10000 == 0:
            print(f"Inserted {count} rows so far.", flush=True)
    cursor.close()
    connection.close()


if __name__ == '__main__':
    SOURCE = 2

    queue = Queue(maxsize=cpu_count * 4)  # Buffer up to 10 chunks

    # file_list = ['/lp/soft/mysql-files/cc_id_content.json']
    data_dir = '/ML-A100/data/train_data/falcon/falcon-refinedweb/id_content_cluster/3'
    file_list = os.listdir(data_dir)
    file_list = list(filter(lambda x: x.endswith(".json"), file_list))
    file_list = list(map(lambda x: os.path.join(data_dir, x), file_list))

    prod = Process(target=producer, args=(queue, file_list))
    cons = [Process(target=consumer, args=(queue,)) for _ in range(cpu_count)]

    prod.start()
    for c in cons:
        c.start()

    prod.join()
    for c in cons:
        queue.put(None)  # Add sentinel values for each consumer
        c.join()


