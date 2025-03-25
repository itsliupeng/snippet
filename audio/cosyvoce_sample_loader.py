import json
import torch
import numpy as np
def sample_loader(raw: dict) -> dict:
    data = raw
    question = data['question']
    answer = data['answer']
    question_audio_token = torch.from_numpy(np.frombuffer(raw['question_audio'], dtype=np.int32).view(1, -1))
    answer_audio_token = torch.from_numpy(np.frombuffer(raw['answer_audio'], dtype=np.int32).view(1, -1))
    return dict(
        __key__=raw["__key__"],
        question=question,
        answer=answer,
        question_audio_token=question_audio_token,
        answer_audio_token=answer_audio_token
    )

def part_filter(part: str) -> bool:
    return True