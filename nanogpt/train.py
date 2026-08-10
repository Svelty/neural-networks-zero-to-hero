import torch
import torch.nn as nn
from torch.nn import functional as F


#hyperparams
batch_size = 64 # parallel independant sequences processed at once
block_size = 256 # max context length for predictions
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2



# --- explicit constraints implied by the code below ---

# Shape constraints
# Block splits the embedding across heads: head_size = n_embd // n_head (line 129),
# and MultiHeadAttention concatenates the n_head outputs back to n_head*head_size
# before feeding nn.Linear(n_embd, n_embd) (line 99) and the residual add (line 136).
# So the division must be exact, otherwise the concat is narrower than n_embd.
assert n_embd % n_head == 0, "n_embd must be divisible by n_head (head_size = n_embd // n_head)"
assert n_head >= 1
assert n_embd >= n_head, "head_size = n_embd // n_head must be >= 1"

# Context-length constraints
# block_size sizes both the causal mask buffer (line 76) and the position embedding
# table (line 146), so every sequence fed to the model must have T <= block_size.
# get_batch always produces exactly T = block_size and generate crops to the last
# block_size tokens (line 176), so this holds as long as block_size >= 1.
assert block_size >= 1
assert batch_size >= 1

# LayerNorm/Linear need a non-degenerate feature dim; FeedForward widens to 4*n_embd.
assert n_embd >= 1
assert n_layer >= 1, "nn.Sequential of zero blocks would make the model a plain bigram"

# Training-loop constraints
assert max_iters >= 1
assert eval_interval >= 1, "used as a modulus in `iter % eval_interval` (line 197)"
assert eval_iters >= 1, "losses.mean() over a zero-length tensor is NaN (line 63)"
assert eval_interval <= max_iters, "otherwise estimate_loss only ever runs at iter 0"

# Numeric-range constraints
assert 0.0 <= dropout < 1.0, "nn.Dropout requires p in [0, 1); p == 1 zeroes everything"
assert learning_rate > 0

assert device in ('cuda', 'cpu')

# Constraints that also exist but cannot be checked here (they depend on the data,
# which is loaded further down) — see the asserts after the train/val split:
#   len(train_data) > block_size  and  len(val_data) > block_size
#   vocab_size >= 1




torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # input string output list of ints
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]



# get_batch calls torch.randint(len(data) - block_size, ...), which requires a
# strictly positive upper bound, and slices data[i : i+block_size+1] for the targets,
# so each split needs strictly more than block_size tokens.
assert len(train_data) > block_size, "train split shorter than block_size"
assert len(val_data) > block_size, "val split shorter than block_size"
assert vocab_size >= 1, "nn.Embedding(vocab_size, n_embd) needs a non-empty vocab"



def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    # one head of self attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,C) ----> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    # multiple heads of attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    # a simple layer followed by a non linearity

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    #transformer block, communication followed by computation

    def __init__(self, n_embd, n_head):
        #n_embd; number of embedding dimensions, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa =  MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directaly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        #idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the latest block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))