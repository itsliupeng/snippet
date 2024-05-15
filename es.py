from elasticsearch import Elasticsearch, helpers
from multiprocessing import Process, Queue
import json
import os
from elasticsearch import ConnectionError, ConnectionTimeout
from time import sleep
import farmhash

cpu_count = 4
CHUNK_NUM = 500


def producer(queue, file_list):
    for file_path in file_list:
        print(f"begin to read {file_path}", flush=True)
        with open(file_path, 'r') as f:
            chunk = []
            for line in f:
                data = json.loads(line)

                action = {
                    "_op_type": "index",
                    "_index": INDEX_NAME,
                    # "_id": str(data["hash_id"]),
                    "_id": str(farmhash.hash64(data['content'])),
                    "_source": data
                }
                chunk.append(action)

                if len(chunk) >= CHUNK_NUM:
                    queue.put(chunk)
                    chunk = []
            if chunk:
                queue.put(chunk)
    queue.put(None)  # Sentinel value to signal the end

def consumer(queue):
    es = None
    while es is None:  # Ensure that the initial connection is established
        try:
            es = Elasticsearch([ES_HOST], request_timeout=60, max_retries=10, retry_on_timeout=True)
        except ConnectionError:
            print("Connection error occurred. Retrying in 5 seconds...", flush=True)
            sleep(5)

    count = 0
    while True:
        chunk = queue.get()
        if chunk is None:  # Check for sentinel value
            break
        try:
            helpers.bulk(es, chunk, max_retries=10, initial_backoff=30)
        except (ConnectionTimeout, ConnectionError, Exception) as e:
            print(f"Error occurred: {e}. Waiting for 5 seconds before retrying...", flush=True)
            es = None
            while es is None:  # Keep trying to re-establish the connection
                try:
                    es = Elasticsearch([ES_HOST], request_timeout=60, max_retries=10, retry_on_timeout=True)
                except (ConnectionError, ConnectionTimeout, Exception) as e:
                    print("Connection error occurred. Retrying in 5 seconds...", flush=True)
                    sleep(5)
            # 完成此次 insert
            helpers.bulk(es, chunk, max_retries=10, initial_backoff=5)

        count += len(chunk)
        if count % 10000 == 0:
            print(f"Inserted {count} rows so far.", flush=True)

if __name__ == '__main__':
    # Elasticsearch Configuration
    ES_HOST = "http://localhost:9200"
    INDEX_NAME = "cc_d0822"

    queue = Queue(maxsize=cpu_count * 4)

    data_dir = '/mnt/vepfs/data/ccnet/processed/rule_ppl_cls_dedup_0822/class/final'
    file_list = os.listdir(data_dir)
    file_list = list(filter(lambda x: x.endswith(".json"), file_list))
    file_list = list(map(lambda x: os.path.join(data_dir, x), file_list))
    file_list = sorted(file_list)

    prod = Process(target=producer, args=(queue, file_list))
    cons = [Process(target=consumer, args=(queue,)) for _ in range(cpu_count)]

    prod.start()
    for c in cons:
        c.start()

    prod.join()
    for c in cons:
        queue.put(None)  # Add sentinel values for each consumer
        c.join()
