import os
import json

whisper_text_file = "all_whisper_refined.jsonl"
whisper_text_list = []
for line in open(whisper_text_file):
    whisper_text_list.append(json.loads(line.strip()))

wav_text_dict = {}
for item in whisper_text_list:
    key = item['wav'].split("/")[-1]
    text = item['whisper']
    refined = item['refined']
    refined_len = len(refined.split())
    text_len = len(text.split())
    if 0.9 * text_len < refined_len < 1.1 * text_len:
        text = refined

    if key in wav_text_dict:
        print(f'duplicate keys for {key} : {text}')
    wav_text_dict[key] = text



meta_file = "all_meta.jsonl"
raw_j_list = []
for line in open(meta_file):
    raw_j_list.append(json.loads(line.strip()))


d_map = {}
for item in raw_j_list:
    if item['question_audio'] in d_map:
        continue
    d_map[item['question_audio']] = item

j_list = list(d_map.values())

for item in j_list:
    if item['question_audio'] in wav_text_dict:
        item['question_refined'] = wav_text_dict[item['question_audio']]
    if item['answer_audio'] in wav_text_dict:
        item['answer_refined'] = wav_text_dict[item['answer_audio']]



question_invalid_list = []
answer_invalid_list = []
question_not_exist_list = []
answer_not_exist_list = []
valid_list = []
for item in j_list:
    q_ok = False
    ans_ok = False

    if item['question_audio'] not in wav_text_dict:
        # print(f"question_audio {item['question_audio']} not in wav_text_dict")
        question_not_exist_list.append(item)
    else:
        a = item['question']
        b = wav_text_dict[item['question_audio']]
        if len(a.split()) > len(b.split()) * 1.2:
            question_invalid_list.append(item)
        else:
            q_ok = True

    if item['answer_audio'] not in wav_text_dict:
        # print(f"answer_audio {item['answer_audio']} not in wav_text_dict")
        answer_not_exist_list.append(item)
    else:
        a = item['answer']
        b = wav_text_dict[item['answer_audio']]
        if len(a.split()) > len(b.split()) * 1.2:
            answer_invalid_list.append(item)
        else:
            ans_ok = True

    if q_ok and ans_ok:
        valid_list.append(item)

print(f"j_list: {len(j_list)}, valid_list: {len(valid_list)}, question_invalid_list: {len(question_invalid_list)}, answer_invalid_list: {len(answer_invalid_list)}")
print('Done.')

with open("valid_meta.jsonl", "w") as of:
    for item in valid_list:
        of.write(f"{json.dumps(item)}\n")
