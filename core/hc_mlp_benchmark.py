import gymnasium as gym
import numpy as np, torch, time
from scipy import stats

torch.set_num_threads(2)

# ── A2PO ──
def run_a2po(seed, ni=80, bs=2, el=40):
    np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make('HalfCheetah-v4'); od = 17; n = 6; lr = 0.003; sig = 0.3
    nets = [torch.nn.Sequential(torch.nn.Linear(od, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)) for _ in range(n)]
    for net in nets:
        for p in net.parameters(): torch.nn.init.normal_(p, 0, 0.01)
    np_ = [sum(p.numel() for p in net.parameters()) for net in nets]
    fi = [np.ones(np_[i]) * 0.01 for i in range(n)]
    yt = [np.zeros(np_[i]) for i in range(n)]
    pp = [np.zeros(np_[i]) for i in range(n)]
    rets = []
    for it in range(ni):
        br = []; bg = [[] for _ in range(n)]
        for b in range(bs):
            obs, _ = env.reset(); log_ps = [[] for _ in range(n)]; rews = []
            for t in range(el):
                acts = []
                for i in range(n):
                    o = torch.FloatTensor(obs); mu = nets[i](o)
                    a_val = mu.detach().numpy().item() + np.random.randn() * sig
                    log_ps[i].append(-((a_val - mu) ** 2) / (2 * sig ** 2))
                    acts.append(np.array([a_val]))
                obs, rew, te, tr, _ = env.step(np.concatenate(acts)); rews.append(rew)
                if te or tr: break
            br.append(sum(rews)); bl = np.mean(rews)
            for i in range(n):
                nets[i].zero_grad()
                loss = sum(-(rews[t] - bl) * log_ps[i][t] for t in range(len(rews))) / len(rews)
                loss.backward()
                g = torch.cat([p.grad.flatten() for p in nets[i].parameters()]).detach().numpy()
                bg[i].append(np.clip(g, -1, 1))
        rets.append(np.mean(br)); ag = [np.mean(bg[i], axis=0) for i in range(n)]
        for i in range(n):
            for g in bg[i]: fi[i] = 0.9 * fi[i] + 0.1 * g ** 2
            pc = ag[i] / (np.sqrt(fi[i]) + 0.01); yt[i] = yt[i] + pc - pp[i]; pp[i] = pc.copy()
            with torch.no_grad():
                off = 0
                for p in nets[i].parameters():
                    nn = p.numel(); p.add_(torch.FloatTensor(yt[i][off:off + nn].reshape(p.shape)) * lr); off += nn
    env.close()
    return float(np.mean(rets[-15:]))

# ── IPPO ──
def run_ippo(seed, ni=80, bs=2, el=40):
    np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make('HalfCheetah-v4'); od = 17; n = 6; lr = 0.003; sig = 0.3
    nets = [torch.nn.Sequential(torch.nn.Linear(od, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)) for _ in range(n)]
    for net in nets:
        for p in net.parameters(): torch.nn.init.normal_(p, 0, 0.01)
    rets = []
    for it in range(ni):
        br = []; bg = [[] for _ in range(n)]
        for b in range(bs):
            obs, _ = env.reset(); log_ps = [[] for _ in range(n)]; rews = []
            for t in range(el):
                acts = []
                for i in range(n):
                    o = torch.FloatTensor(obs); mu = nets[i](o)
                    a_val = mu.detach().numpy().item() + np.random.randn() * sig
                    log_ps[i].append(-((a_val - mu) ** 2) / (2 * sig ** 2))
                    acts.append(np.array([a_val]))
                obs, rew, te, tr, _ = env.step(np.concatenate(acts)); rews.append(rew)
                if te or tr: break
            br.append(sum(rews)); bl = np.mean(rews)
            for i in range(n):
                nets[i].zero_grad()
                loss = sum(-(rews[t] - bl) * log_ps[i][t] for t in range(len(rews))) / len(rews)
                loss.backward()
                g = torch.cat([p.grad.flatten() for p in nets[i].parameters()]).detach().numpy()
                bg[i].append(np.clip(g, -1, 1))
        rets.append(np.mean(br)); ag = [np.mean(bg[i], axis=0) for i in range(n)]
        for i in range(n):
            with torch.no_grad():
                off = 0
                for p in nets[i].parameters():
                    nn = p.numel(); p.add_(torch.FloatTensor(ag[i][off:off + nn].reshape(p.shape)) * lr); off += nn
    env.close()
    return float(np.mean(rets[-15:]))

# ── NPG ──
def run_npg(seed, ni=80, bs=2, el=40):
    np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make('HalfCheetah-v4'); od = 17; n = 6; lr = 0.003; sig = 0.3
    nets = [torch.nn.Sequential(torch.nn.Linear(od, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)) for _ in range(n)]
    for net in nets:
        for p in net.parameters(): torch.nn.init.normal_(p, 0, 0.01)
    rets = []
    for it in range(ni):
        br = []; bg = [[] for _ in range(n)]
        for b in range(bs):
            obs, _ = env.reset(); log_ps = [[] for _ in range(n)]; rews = []
            for t in range(el):
                acts = []
                for i in range(n):
                    o = torch.FloatTensor(obs); mu = nets[i](o)
                    a_val = mu.detach().numpy().item() + np.random.randn() * sig
                    log_ps[i].append(-((a_val - mu) ** 2) / (2 * sig ** 2))
                    acts.append(np.array([a_val]))
                obs, rew, te, tr, _ = env.step(np.concatenate(acts)); rews.append(rew)
                if te or tr: break
            br.append(sum(rews)); bl = np.mean(rews)
            for i in range(n):
                nets[i].zero_grad()
                loss = sum(-(rews[t] - bl) * log_ps[i][t] for t in range(len(rews))) / len(rews)
                loss.backward()
                g = torch.cat([p.grad.flatten() for p in nets[i].parameters()]).detach().numpy()
                bg[i].append(np.clip(g, -1, 1))
        rets.append(np.mean(br)); ag = [np.mean(bg[i], axis=0) for i in range(n)]
        for i in range(n):
            fg = max(np.mean(ag[i] ** 2), 1e-8)
            update = ag[i] / (np.sqrt(fg) + 0.01)
            with torch.no_grad():
                off = 0
                for p in nets[i].parameters():
                    nn = p.numel(); p.add_(torch.FloatTensor(update[off:off + nn].reshape(p.shape)) * lr); off += nn
    env.close()
    return float(np.mean(rets[-15:]))

# ── Run all ──
if __name__ == '__main__':
    seeds = range(5)

    print("Running A2PO...")
    aucs = [run_a2po(s) for s in seeds]
    ci = stats.t.interval(0.95, df=4, loc=np.mean(aucs), scale=stats.sem(aucs))
    print(f"HC-MLP A2PO: {np.mean(aucs):.1f}±{np.std(aucs, ddof=1):.1f} CI=[{ci[0]:.1f},{ci[1]:.1f}]")

    print("Running IPPO...")
    aucs = [run_ippo(s) for s in seeds]
    ci = stats.t.interval(0.95, df=4, loc=np.mean(aucs), scale=stats.sem(aucs))
    print(f"HC-MLP IPPO: {np.mean(aucs):.1f}±{np.std(aucs, ddof=1):.1f} CI=[{ci[0]:.1f},{ci[1]:.1f}]")

    print("Running NPG...")
    aucs = [run_npg(s) for s in seeds]
    ci = stats.t.interval(0.95, df=4, loc=np.mean(aucs), scale=stats.sem(aucs))
    print(f"HC-MLP NPG(lr=0.003): {np.mean(aucs):.1f}±{np.std(aucs, ddof=1):.1f} CI=[{ci[0]:.1f},{ci[1]:.1f}]")
