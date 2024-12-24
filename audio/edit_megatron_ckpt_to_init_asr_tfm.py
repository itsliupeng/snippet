
import torch
import torch.nn.functional as F

m = torch.load("model_optim_rng.pt", "cpu")
m['model'].keys()

# copy
for key in m['model'].keys():
    if '_extra_state' in key:
        continue

    if key.startswith("asr_adapter"):
        copy_key = key.replace("asr_adapter", "language_model.decoder")
        print(f"replace weight of {key} with {copy_key}")
        m['model'][key] = m['model'][copy_key]


# check
for key in m['model'].keys():
    if '_extra_state' in key:
        continue

    if key.startswith("asr_adapter"):
        copy_key = key.replace("asr_adapter", "language_model.decoder")

        a = m['model'][key]
        b = m['model'][copy_key]

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
        # if cosine_similarity != 1.0:
            print(f"{key}: {a.shape}, cosine {cosine_similarity}")


################################################################

torch.save(m, "model_optim_rng.pt")