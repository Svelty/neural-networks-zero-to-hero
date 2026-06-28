import torch
import torch.nn.functional as F



words = open('names.txt', 'r').read().splitlines()



chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}




xs, ys = [], []

for w in words:
    chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)


xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)


g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)


for k in range(10):
    #forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    print(xenc.shape)
    print(W.shape)
    logits = xenc @ W # log counts
    print(logits.shape)
    counts = logits.exp() #equivalent to N from prev example
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean() # this last term is "regularization" - equivalent to smoothing above
    print(loss.item())

    #backpass
    W.grad = None #set to 0
    loss.backward()

    #update
    W.data += -50 * W.grad


g = torch.Generator().manual_seed(2147483647)


for i in range(5):
    out = []
    ix = 0;
    while True:

        # BEFORE
        # p = P[ix]

        #NOW
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # log counts
        counts = logits.exp() #equivalent to N from prev example
        p = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break;
    print(''.join(out))