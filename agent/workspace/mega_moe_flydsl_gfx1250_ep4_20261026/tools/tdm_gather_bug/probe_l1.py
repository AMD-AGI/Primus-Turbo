"""EP1 fused stage1 (dispatch + L1 gather GEMM), per pool row: is l1_out right for the token buffer it read?

python3 probe_l1.py [--shape 1024 2880 2880 32 4] [--reps 4] [--bwd]
Every rep: fresh x / routing, stage1 forward (and with --bwd the full staged fwd+bwd, which dispatches dy
and x into the same token buffer). Prints the rows whose l1_out is off from token_buffer[idx] @ w1[e].T, by
position in the 256-row tile and by 64-column k-chunk (the chunk whose removal explains the error best).
"""
import argparse, os, socket, sys, torch, torch.distributed as dist

sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe_stage1, fused_mega_moe_stage2

ap = argparse.ArgumentParser()
ap.add_argument("--shape", type=int, nargs=5, default=[1024, 2880, 2880, 32, 4])
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--bwd", action="store_true")
args = ap.parse_args()
T_, H, I, E, K = args.shape
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(s.getsockname()[1]))
dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda", 0))
g = dist.group.WORLD
torch.manual_seed(T_)
w1 = (torch.randn(E, 2 * I, H, device="cuda") * 5 / H**0.5).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda") * 2 / I**0.5).to(torch.bfloat16)
act = GLUActivation.swigluoai()
prev = None
for rep in range(args.reps):
    x = torch.randn(T_, H, device="cuda", dtype=torch.bfloat16)
    tw, ti = torch.randn(T_, E, device="cuda").softmax(-1).topk(K, dim=-1)
    lx, ltw, lw1, lw2 = (t.clone().requires_grad_() for t in (x, tw, w1, w2))
    l1_out, dw, handle = fused_mega_moe_stage1(lx, ti, ltw, lw1, g)
    torch.cuda.synchronize()
    buf = get_symm_buffer_for_mega_moe()
    a_rows = int(buf.num_max_recv_tokens)
    tok = buf.dispatch_token_buffer.view(-1)[: a_rows * H].view(a_rows, H).clone()
    idx = handle.pool_row_to_recv_token[: l1_out.shape[0]].long()
    nt = int(handle.num_tile_blocks[0])
    rows = nt * 256
    idx, out = idx[:rows], l1_out[:rows].float()
    e = handle.tile_to_expert[:nt].long().repeat_interleave(256)
    valid = (idx >= 0) & (idx < a_rows)
    ia = idx.clamp(0, a_rows - 1)
    want = torch.cat([tok[ia[t * 256:(t + 1) * 256]].float() @ w1[e[t * 256]].float().t() for t in range(nt)])
    err = (out - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-6)
    err[~valid] = 0
    bad = (err > 2e-2).nonzero().flatten()
    msg = f"rep {rep}: tiles {nt}, valid rows {int(valid.sum())}, bad rows {bad.numel()}"
    if bad.numel():
        pos = sorted(set((bad % 256).tolist()))
        # which 64-wide k-chunk was missed / replaced: remove one chunk's contribution and see which fits best
        r = bad[:64]
        diff = out[r] - want[r]
        fits = []
        for c in range(H // 64):
            ks = slice(c * 64, (c + 1) * 64)
            part = torch.stack([tok[ia[p], ks].float() @ w1[e[p], :, ks].float().t() for p in r.tolist()])
            fits.append(((diff + part).norm(dim=1) / diff.norm(dim=1)).cpu())
        fits = torch.stack(fits)  # chunk, row
        best = fits.min(0)
        msg += (f", worst {err.max().item():.2e}, tile positions {pos[:24]}{'...' if len(pos) > 24 else ''}; "
                f"tiles {sorted(set((bad // 256).tolist()))[:12]}; best single-chunk fit per row (chunk:resid) "
                + " ".join(f"{int(c)}:{v:.2f}" for c, v in zip(best.indices[:12], best.values[:12])))
        colerr = ((out[bad] - want[bad]).abs() > 0.05 * want[bad].abs().amax(1, keepdim=True))
        nblk = colerr.view(bad.numel(), -1, 16).any(2)  # row, 16-col block
        per_nb = nblk.view(bad.numel(), -1).sum(0)
        hot = per_nb.nonzero().flatten().tolist()
        msg += (f"\n  wrong 16-col blocks (block:rows) " + " ".join(f"{b}:{int(per_nb[b])}" for b in hot[:24])
                + f"; 256-col N blocks {sorted(set(b // 16 for b in hot))}")
        src = ia[bad]
        msg += f"\n  source rows: {int((src < 256).sum())}/{src.numel()} below 256, e.g. {src[:12].tolist()}"
        cols = torch.tensor(sorted(set(c for b in hot for c in range(b * 16, b * 16 + 16))), device="cuda")
        for p in bad[:8].tolist():
            wb = w1[e[p]][cols].float()
            got = out[p, cols]
            fit = []
            for nm, s_ in (("cur", tok), ("prev", prev if prev is not None else tok)):
                cand = s_.float() @ wb.t()  # rows, cols
                r_ = ((cand - got).norm(dim=1) / got.norm()).min(0)
                fit.append(f"{nm} r{int(r_.indices)} ({r_.values.item():.3f})")
            msg += f"\n  row {p} src {int(ia[p])}: wrong cols best explained by " + ", ".join(fit)
            if prev is not None:
                d = got - want[p, cols]
                res = []
                for c in range(H // 64):
                    ks = slice(c * 64, (c + 1) * 64)
                    for nm, alt in (("prev", prev[ia[p], ks].float()), ("zero", torch.zeros(64, device="cuda"))):
                        part = (alt - tok[ia[p], ks].float()) @ wb[:, ks].t()
                        res.append(((d - part).norm() / d.norm()).item())
                best = min(range(len(res)), key=res.__getitem__)
                msg += f"; best k-tile swap: kt {best // 2} <- {('prev', 'zero')[best % 2]} (resid {res[best]:.2f})"
                # 16 B chunks of the first two k-tiles, greedy, prev data
                contrib = torch.stack([(prev[ia[p], j * 64:(j + 1) * 64].float() - tok[ia[p], j * 64:(j + 1) * 64].float())
                                       @ wb[:, j * 64:(j + 1) * 64].t() for j in range(H // 64)])
                left, picked = d.clone(), []
                for _ in range(H // 64):
                    j = int((left[None] - contrib).norm(dim=1).argmin())
                    if (left - contrib[j]).norm() >= 0.98 * left.norm():
                        break
                    left -= contrib[j]
                    picked.append(j)
                msg += f"; prev k-tiles {sorted(picked)} -> resid {(left.norm() / d.norm()).item():.2f}"
                # k-tile kt of this row taken from another row r' (current or previous data)?
                for kt_ in (0, 1):
                    ks = slice(kt_ * 64, (kt_ + 1) * 64)
                    base = tok[ia[p], ks].float() @ wb[:, ks].t()
                    best_ = None
                    for nm, s_ in (("cur", tok), ("prev", prev)):
                        part = s_[:, ks].float() @ wb[:, ks].t() - base  # rows, cols
                        r2 = (d[None] - part).norm(dim=1) / d.norm()
                        v, i_ = r2.min(0)
                        if best_ is None or v < best_[0]:
                            best_ = (v.item(), nm, int(i_))
                    msg += f"; kt{kt_} from {best_[1]} r{best_[2]} (resid {best_[0]:.2f})"
        if prev is not None:
            # per 16-byte chunk: does swapping in the previous rep's data for some chunks explain the row?
            explained = []
            for p in bad[:16].tolist():
                d = out[p] - want[p]
                w = w1[e[p]].float()
                contrib = [(prev[ia[p], j * 8:(j + 1) * 8].float() - tok[ia[p], j * 8:(j + 1) * 8].float())
                           @ w[:, j * 8:(j + 1) * 8].t() for j in range(H // 8)]
                contrib = torch.stack(contrib)  # chunk, N
                left, picked = d.clone(), []
                for _ in range(6):
                    j = int((left[None] - contrib).norm(dim=1).argmin())
                    if (left - contrib[j]).norm() >= left.norm():
                        break
                    left -= contrib[j]
                    picked.append(j)
                explained.append(f"{(left.norm() / d.norm()).item():.2f}@{picked}")
            msg += "\n  stale-chunk fit (resid@16B chunks): " + " ".join(explained)
        # recover the A row each bad output used (out = a_used @ w1[e].T, 2I >> H) and trace its wrong 16 B chunks
        cands = {"cur": tok, "prev": prev if prev is not None else tok}
        for p in bad[:6].tolist():
            w = w1[e[p]].double()
            a_used = torch.linalg.lstsq(w, out[p].double()[:, None]).solution[:, 0].float()
            a_true = tok[ia[p]].float()
            ch = (a_used - a_true).view(-1, 8).norm(dim=1) / a_true.view(-1, 8).norm(dim=1).clamp_min(1e-3)
            wrong = (ch > 0.3).nonzero().flatten().tolist()
            where = []
            for j in wrong[:6]:
                piece = a_used[j * 8:(j + 1) * 8]
                best = None
                for nm, src in cands.items():
                    blk = src.float().view(src.shape[0], -1, 8)  # row, chunk, 8
                    dist_ = (blk - piece).norm(dim=2) / piece.norm().clamp_min(1e-3)
                    v, flat = dist_.flatten().min(0)
                    r_, c_ = divmod(int(flat), blk.shape[1])
                    if best is None or v < best[0]:
                        best = (float(v), nm, r_, c_)
                where.append(f"c{j}<-{best[1]} r{best[2]} c{best[3]} ({best[0]:.2f})")
            msg += (f"\n  row {p} (tile {p // 256} pos {p % 256}, src {int(ia[p])}): {len(wrong)} wrong 16B chunks "
                    f"of {H // 8}: " + "; ".join(where))
    prev = tok
    print(msg, flush=True)
    if args.bwd:
        y = fused_mega_moe_stage2(l1_out, dw, handle, ti, ltw, lw2, g, act, None, None)
        y.backward(torch.randn_like(y))
        torch.cuda.synchronize()
dist.destroy_process_group()
