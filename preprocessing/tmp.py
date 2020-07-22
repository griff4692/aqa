from scipy.spatial.distance import cdist

from sentence_transformers import SentenceTransformer

x = 'dog'
y = 'dogs'

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

x_embed = embedder.encode(x)  # Batch size 1
y_embed = embedder.encode(y)

sim = cdist(x_embed, y_embed)
print(sim)
