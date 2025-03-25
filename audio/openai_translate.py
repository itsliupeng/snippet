import json
import argparse
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="pretrain-data",
    base_url="https://api.01ww.xyz/v1",
)
model = 'gpt-4o'


def get_text(text, n=2):
    prompt = f"请将下面的英文句子翻译为中文，要求尽可能地口语化，符合一个中国北方本地市民的说话方式, 只给出翻译后的中文，不要附带其他内容：{text}"
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            n=n,
            stream=False
        )
        if completion:
            return [x.message.content for x in completion.choices]
        else:
            print(completion)
            return []
    except Exception as e:
        print(f"Error during translation: {e}, flush=True")
        return []


def main(input_file, output_file):
    with open(output_file, "w", encoding="utf-8") as of:
        count = 0
        for line in open(input_file, encoding="utf-8"):
            try:
                j = json.loads(line.strip())

                # 数据完整性检查
                if 'question' not in j or 'answer' not in j:
                    print("Skipping line due to missing keys:", j, flush=True)
                    continue

                question = j['question']
                answer = j['answer']

                new_questions = get_text(question)
                new_answers = get_text(answer)

                if len(new_questions) == len(new_answers):
                    for idx, (q, a) in enumerate(zip(new_questions, new_answers)):
                        new_j = {
                            "question": q,
                            "answer": a,
                            "repeat_id": idx,
                            "index": j.get('index'),
                            "round": j.get('round'),
                            "split_name": j.get("split_name"),
                            "raw_question": question,
                            "raw_answer": answer

                        }
                        of.write(json.dumps(new_j, ensure_ascii=False) + "\n")
                else:
                    print(f"Mismatch in translation count for question: {question}", flush=True)

            except Exception as e:
                print(f"Error processing line: {e}", flush=True)

            count += 1
            if count % 3 == 0:
                of.flush()
                print(f"Processed {count} jsons", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate JSONL questions and answers into Chinese.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output JSONL file."
    )

    args = parser.parse_args()
    main(args.input_file, args.output_file)
