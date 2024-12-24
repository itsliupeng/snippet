import os
import json

def trans(original_json):
    transformed_json_list = []
    # 确保输入格式正确
    messages = original_json.get("messages", [])
    messages = list(filter(lambda msg: msg['role'] != "system", messages))
    if not messages or len(messages) % 2 != 0:
        print(f"The 'messages' list not even for {original_json}")
        return ""

    for i in range(0, len(messages), 2):
        # 每两条消息构成一组 question 和 answer
        question = messages[i]
        answer = messages[i + 1]

        # 确保每组的结构符合要求
        if question['role'] == 'user' and answer['role'] == 'assistant':
            transformed_json = {
                "id": original_json["id"],
                "message_id": i // 2,  # 按组编号
                "question": question["content"],
                "answer": answer["content"]
            }
            transformed_json_list.append(transformed_json)
        else:
            print(f"Message roles are not in the expected order (user -> assistant).\n\n {question} \n\n {question}")
            return ""

    return transformed_json_list

input_dir = "/gpfs/public/align/caoguo/datasets/posttrain/versions/chatmls/moyi_sft_data_0927/moyi_sft_data_0914_v3_all_latest/jsonls_202404_fixed_length_with_4o"
output_dir = "/lp/pretrain_audio_data/instruct_data"
files = os.listdir(input_dir)
for x in files:
    out_file = os.path.join(output_dir, x)
    with open(out_file, "w") as of:
        for line in open(os.path.join(input_dir, x)):
            j = json.loads(line.strip())
            new_j = trans(j)
            if new_j != "":
                of.write(f"{json.dumps(new_j)}\n")
    print(f"finished writing {x}")
