import numpy as np
from scipy import stats


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_mg(method, n_agents, het, seed, n_iters=800, lr=0.05):
    rng = np.random.RandomState(seed)
    asz = [3 + i % 3 for i in range(n_agents)]  # heterogeneous action sizes
    rs = [1.0 + het * i / (n_agents - 1) for i in range(n_agents)] if n_agents > 1 else [1.0]

    # Build reward tensor (use factored form for large n)
    pol = [rng.randn(asz[i]) * 0.01 for i in range(n_agents)]
    fd = [np.ones(asz[i]) * 0.01 for i in range(n_agents)]
    y = [np.zeros(asz[i]) for i in range(n_agents)]
    pg = [np.zeros(asz[i]) for i in range(n_agents)]
    rets = []
    bs = 16

    for it in range(n_iters):
        probs = [softmax(pol[i]) for i in range(n_agents)]
        br = []
        ba = []

        for _ in range(bs):
            acts = [rng.choice(asz[i], p=probs[i]) for i in range(n_agents)]
            # Factored reward: bonus for action 0 + cooperative bonus
            r = sum(rs[i] * (1.0 if acts[i] == 0 else 0.0) for i in range(n_agents))
            r += 5.0 * (1.0 if all(a == 0 for a in acts) else 0.0)
            r += rng.randn() * 0.1
            br.append(r)
            ba.append(acts)

        mr = np.mean(br)
        rets.append(mr)

        grads = []
        for i in range(n_agents):
            g = np.zeros(asz[i])
            for b in range(bs):
                gl = -probs[i].copy()
                gl[ba[b][i]] += 1.0
                g += (br[b] - mr) * gl
            grads.append(g / bs)

        if method == 'IPPO':
            for i in range(n_agents):
                pol[i] += lr * grads[i]

        elif method == 'NPG_Uniform':
            for i in range(n_agents):
                K = asz[i]
                F = np.eye(K) * 0.01
                for b in range(bs):
                    gl = -probs[i].copy()
                    gl[ba[b][i]] += 1.0
                    F += np.outer(gl, gl) / bs
                pol[i] += lr * np.linalg.solve(F, grads[i])

        elif method == 'A2PO_Diag':
            for i in range(n_agents):
                for b in range(bs):
                    gl = -probs[i].copy()
                    gl[ba[b][i]] += 1.0
                    fd[i] = 0.9 * fd[i] + 0.1 * gl ** 2
                pc = grads[i] / (fd[i] + 0.01)
                y[i] = y[i] + pc - pg[i]
                pg[i] = pc.copy()
                pol[i] += lr * y[i]

    mx = max(rets)
    thr = 0.9 * mx
    above = [j for j, v in enumerate(rets) if v > thr]
    return above[0] if above else n_iters


if __name__ == '__main__':
    seeds = list(range(5))

    for n_agents in [10, 20]:
        het = 2.0
        print(f"\n=== n={n_agents} agents, het={het} ===")
        for m in ['IPPO', 'NPG_Uniform', 'A2PO_Diag']:
            convs = [run_mg(m, n_agents, het, s) for s in seeds]
            ci = stats.t.interval(0.95, df=4, loc=np.mean(convs), scale=stats.sem(convs))
            print(f"  {m:15s}: conv={np.mean(convs):.1f}±{np.std(convs, ddof=1):.1f} CI=[{ci[0]:.1f},{ci[1]:.1f}]")
