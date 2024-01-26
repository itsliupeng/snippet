import json
from multiprocessing import Pool, Process, Queue, cpu_count


# def producer(filename, queue):
#     with open(filename, 'r') as f:
#         for line in f:
#             queue.put(line)
#     # Signal the consumers that the producer is done
#     for _ in range(cpu_count()):
#         queue.put(None)
#
#
# def consumer(queue, output):
#     while True:
#         line = queue.get()
#         if line is None:  # Signal to terminate the consumer
#             break
#         json_str = line.strip().split("\t")[-1]
#         j = json.loads(json_str)
#         output.put(j)
#
#
# def read_json(filename):
#     queue = Queue(maxsize=1000)  # Buffer up to 1000 lines
#     output = Queue()
#
#     # Start producer
#     prod = Process(target=producer, args=(filename, queue))
#     prod.start()
#
#     # Start consumers
#     consumers = []
#     for _ in range(cpu_count()):
#         cons = Process(target=consumer, args=(queue, output))
#         cons.start()
#         consumers.append(cons)
#
#     j_list = []
#     finished_consumers = 0
#     while finished_consumers < cpu_count():
#         item = output.get()
#         if item is None:
#             finished_consumers += 1
#         else:
#             j_list.append(item)
#
#     # Wait for processes to finish
#     prod.join()
#     for cons in consumers:
#         cons.join()
#
#     return j_list


def read_json(filename):
    j_list = []
    for line in open(filename):
        json_str = line.strip().split("\t")[-1]
        j = json.loads(json_str)
        j_list.append(j)
    return j_list


# {'Question': 59049, 'Answer': 207960, 'Post': 60394}


def parse_answer(item):
    # todo: filter by behaviors
    if 'json_raw' in item:
        try:
            if item['json_raw'] is None:
                return None, None
            json_raw = json.loads(item['json_raw'])
        except Exception as e:
            print(f"Answer: {item}: {e}")
            return None, None
    else:
        json_raw = item
    try:
        # Extract the title
        title_data = json.loads(json_raw['question']['title'])
        title_values = [[span['text'] for span in section['spans']] for section in title_data['sections']]

        # Extract the content
        content_data = json.loads(json_raw['content'])
        text_values = [[span['text'] for span in section['spans']] for section in content_data['sections']]

        return title_values, text_values

    except Exception as e:
        print(f"Answer: {json_raw}: {e}")
        return None, None


def parse_post(item):
    # todo: filter by behaviors
    if 'json_raw' in item:
        try:
            if item['json_raw'] is None:
                return None, None
            json_raw = json.loads(item['json_raw'])
        except Exception as e:
            print(f"Post {item}: {e}")
            return None, None
    else:
        json_raw = item
    try:
        doc = json_raw['contentQtextDocument']
        if 'contentEmbedSection' in doc and doc['contentEmbedSection'] is not None:
            qtext = doc['contentEmbedSection']['content']
            # Extract the title
            if 'question' in qtext and 'title' in qtext['question'] and qtext['question']['title'] is not None:
                title_data = json.loads(qtext['question']['title'])
            elif 'title' in qtext:
                if qtext['title'] is None:
                    return None, None
                title_data = json.loads(qtext['title'])
            else:
                raise ValueError(f"Post: no valid title data: {qtext}")
            # Extract the content
            if 'content' in qtext and qtext['content'] is not None:
                content_data = json.loads(qtext['content'])
            else:
                return None, None
        else:
            title_data = json.loads(json_raw['title'])
            content_data = json.loads(json_raw['content'])

        title_values = [[span['text'] for span in section['spans']] for section in title_data['sections']]
        text_values = [[span['text'] for span in section['spans']] for section in content_data['sections']]
        return title_values, text_values
    except Exception as e:
        print(f"Post: {json_raw}: {e}")
        return None, None


def parse_json_list(l):
    question_list = []
    post_list = []
    for x in l:
        if x['type_name'] == 'Answer':
            question_list.append(x)
        elif x['type_name'] == 'Post':
            post_list.append(x)
        else:
            pass
    return question_list, post_list


def make_qa(item):
    q = item[0]
    a = item[1]
    q_s = "\n".join([" ".join(inner_list) for inner_list in q])
    a_s = "\n".join([" ".join(inner_list) for inner_list in a])
    return q_s, a_s


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str)

    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    print(f"parse from {in_file} to {out_file}")

    j_list = read_json(in_file)

    question_list, post_list = parse_json_list(j_list)

    # Initialize a pool of workers
    num_workers = 128  # Change this based on your CPU cores
    with Pool(num_workers) as pool:
        question_parsed_list = pool.map(parse_answer, question_list)
        post_parsed_list = pool.map(parse_post, post_list)

    # question_parsed_list = list(map(parse_answer, question_list))
    question_parsed_list = list(filter(lambda x: x[0] is not None and x[1] is not None, question_parsed_list))
    qa_list = list(map(make_qa, question_parsed_list))

    # post_parsed_list = list(map(parse_post, post_list))
    post_parsed_list = list(filter(lambda x: x[0] is not None and x[1] is not None, post_parsed_list))
    post_qa_list = list(map(make_qa, post_parsed_list))

    all_qa_list = qa_list + post_qa_list
    out_j_list = []
    for item in all_qa_list:
        out_j_list.append(json.dumps({'question': item[0], 'answer': item[1]}, ensure_ascii=False))
    with open(out_file, 'w') as of:
        for item in out_j_list:
            of.write(f"{item}\n")
