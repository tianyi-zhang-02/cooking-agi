"""Watch the vanilla Transformer work: shape trace -> train -> cross-attention map.

Task: reverse a sequence. src = [a b c d], tgt = [d c b a]. It is a seq2seq task
(so the encoder-decoder shape is justified) and the correct cross-attention is
known in advance -- an anti-diagonal -- so you can literally look at the attention
matrix and check the model learned to align the way it should.

Run: python vanilla_demo.py
"""

import torch

from vanilla import VanillaTransformer, causal_mask, loss_fn, make_optimizer, pad_mask

PAD, BOS, EOS = 0, 1, 2
N_SYM, MAX_LEN = 16, 10
VOCAB = 3 + N_SYM


def make_batch(bs, device, fixed_len=None):
    src = torch.zeros(bs, MAX_LEN, dtype=torch.long)
    tgt_in = torch.zeros(bs, MAX_LEN + 1, dtype=torch.long)
    tgt_out = torch.zeros(bs, MAX_LEN + 1, dtype=torch.long)
    lens = (torch.full((bs,), fixed_len) if fixed_len
            else torch.randint(MAX_LEN // 2, MAX_LEN + 1, (bs,)))
    for b, l in enumerate(lens.tolist()):
        syms = torch.randint(3, VOCAB, (l,))
        src[b, :l] = syms
        tgt_in[b, 0], tgt_in[b, 1:l + 1] = BOS, syms.flip(0)
        tgt_out[b, :l], tgt_out[b, l] = syms.flip(0), EOS
    return src.to(device), tgt_in.to(device), tgt_out.to(device), lens


# --------------------------------------------------------------------------- #
def shape_trace(model, src, tgt_in):
    """One forward pass, printing what each stage produces."""
    print("shape trace (B=batch, S=src len, T=tgt len, d=d_model, h=heads)\n")
    b, s, t = src.shape[0], src.shape[1], tgt_in.shape[1]
    d, h = model.generator.in_features, model.encoder.layers[0].self_attn.h
    print(f"  src token ids                          {tuple(src.shape)}")

    src_m = pad_mask(src, PAD)
    tgt_m = pad_mask(tgt_in, PAD) | causal_mask(t, src.device)
    print(f"  src_mask   (True = blocked)            {tuple(src_m.shape)}  "
          f"-> broadcasts to (B,{h},S,S)")
    print(f"  tgt_mask   (pad OR causal)             {tuple(tgt_m.shape)}")

    x = model.src_embed(src)
    print(f"  embed * sqrt(d) + positional encoding  {tuple(x.shape)}")

    enc = model.encoder.layers[0]
    q = enc.self_attn.w_q(x).view(b, s, h, -1).transpose(1, 2)
    print(f"  Q/K/V after split into heads           {tuple(q.shape)}   d_k = d/h = {d // h}")
    print(f"  attention weights QK^T/sqrt(d_k)       {(b, h, s, s)}  <- rows sum to 1")

    memory = model.encode(src, src_m)
    print(f"  encoder output ('memory')              {tuple(memory.shape)}")

    y = model.tgt_embed(tgt_in)
    print(f"  decoder input embeddings               {tuple(y.shape)}")
    print(f"  cross-attn: Q from decoder {tuple(y.shape)}, K/V from memory {tuple(memory.shape)}")
    print(f"              -> weights                 {(b, h, t, s)}  <- NOT square")

    out = model.decode(memory, src_m, tgt_in, tgt_m)
    print(f"  decoder output                         {tuple(out.shape)}")
    print(f"  logits = generator(out)                {tuple(model.generator(out).shape)}")


def ascii_attention(mat, row_lbl, col_lbl, title):
    """mat: (T, S) attention weights, rows = decoder steps, cols = source positions."""
    ramp = " .:-=+*#%@"
    print(f"\n{title}")
    print("        " + "".join(f"{c:>3}" for c in col_lbl) + "   <- source (encoder)")
    for i, row in enumerate(mat):
        cells = "".join(f"  {ramp[min(int(v * len(ramp)), len(ramp) - 1)]}" for v in row.tolist())
        print(f"  {row_lbl[i]:>5} {cells}")
    print("  ^ decoder step")


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    device = "cpu"  # model is tiny; cpu beats the mps dispatch overhead here

    model = VanillaTransformer(VOCAB, VOCAB, n=2, d_model=64, d_ff=128, h=4,
                               dropout=0.1, pad_idx=PAD).to(device)
    opt, sched = make_optimizer(model, d_model=64, warmup=400)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}"
          f"   (paper's base model: 65M)\n")

    src, tgt_in, tgt_out, _ = make_batch(2, device, fixed_len=MAX_LEN)
    model.eval()
    shape_trace(model, src, tgt_in)

    print("\n" + "=" * 68 + "\ntraining on 'reverse the sequence'\n")
    model.train()
    for step in range(1, 3001):
        src, tgt_in, tgt_out, _ = make_batch(64, device)
        loss = loss_fn(model(src, tgt_in), tgt_out, PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e}")

    # ------------------------------------------------------------- evaluate
    src, _, _, lens = make_batch(64, device)
    out = model.greedy_decode(src, MAX_LEN + 1, BOS)
    ok = sum(out[b, 1:l + 1].tolist() == src[b, :l].flip(0).tolist()
             for b, l in enumerate(lens.tolist()))
    print(f"\ngreedy decode: {ok}/64 sequences reversed exactly")
    l = lens[0].item()
    print(f"  src     {src[0, :l].tolist()}")
    print(f"  decoded {out[0, 1:l + 1].tolist()}")

    # ------------------------------------------- what did cross-attention learn
    src, tgt_in, _, _ = make_batch(1, device, fixed_len=8)
    model.eval()
    with torch.no_grad():
        model(src, tgt_in)
    # (B, h, T, S) -> average the heads of the last decoder layer's cross-attn
    attn = model.decoder.layers[-1].cross_attn.attn[0].mean(0)[:9, :8]
    ascii_attention(
        attn,
        row_lbl=["BOS"] + [str(i) for i in range(1, 9)],
        col_lbl=[str(i) for i in range(1, 9)],
        title="cross-attention, last decoder layer (heads averaged)",
    )
    print("\n  decoder step k should be reading source position 9-k -> anti-diagonal")


if __name__ == "__main__":
    main()
