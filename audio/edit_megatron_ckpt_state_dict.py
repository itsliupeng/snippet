
import torch
import torch.nn.functional as F

m = torch.load("model_optim_rng.pt", "cpu")
m['model'].keys()


yi_model = torch.load("/lp/models/Yi-6B_raw_tp1/iter_0000001/mp_rank_00/model_optim_rng.pt", "cpu")
yi_model['model'].keys()

a = m['model']['language_model.embedding.word_embeddings.weight']
b = yi_model['model']['embedding.word_embeddings.weight']

for key in yi_model['model'].keys():
    if '_extra_state' in key:
        continue

    language_key = f"language_model.{key}"
    a = m['model'][language_key]
    b = yi_model['model'][key]

    if len(a.shape) == 1:
        # Normalize the embeddings along the last dimension
        a_normalized = F.normalize(a.view(-1), p=2, dim=0)  # Normalize each row of a
        b_normalized = F.normalize(b.view(-1), p=2, dim=0)  # Normalize each row of b

        # Compute cosine similarity row-wise
        cosine_similarity = torch.sum(a_normalized * b_normalized, dim=0)
    else:
        # Normalize the embeddings along the last dimension
        a_normalized = F.normalize(a, p=2, dim=1)  # Normalize each row of a
        b_normalized = F.normalize(b, p=2, dim=1)  # Normalize each row of b

        # Compute cosine similarity row-wise
        cosine_similarity = torch.sum(a_normalized * b_normalized, dim=1)
        # Average cosine similarity for all embeddings
        cosine_similarity = torch.mean(cosine_similarity)
    if cosine_similarity != 1.0:
        print(f"{key}: {a.shape}, cosine {cosine_similarity}")





for key in ['embedding.word_embeddings.weight', 'output_layer.weight']:
    language_key = f"language_model.{key}"
    m['model'][language_key] = yi_model['model'][key]
    print(f"updating {key}")


################################################################

torch.save(m, "model_optim_rng.pt")