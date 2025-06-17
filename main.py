from scipy.stats import f
import numpy as np


np.random.seed(1)
group_A = np.random.normal(50, 5, 20)
group_B = np.random.normal(52, 5, 20)
group_C = np.random.normal(55, 5, 20)
group_D = np.random.normal(58, 5, 20)

means = {g: np.mean(eval(f'group_{g}')) for g in ['A', 'B', 'C', 'D']}
n = {g: len(eval(f'group_{g}')) for g in ['A', 'B', 'C', 'D']}
k = 4
N = sum(n.values())
df_error = N - k
SSE = sum(np.sum((eval(f'group_{g}') - np.mean(eval(f'group_{g}')))**2) for g in ['A', 'B', 'C', 'D'])
MSE = SSE / df_error

contrasts = {
    'A - B': {'A': 1, 'B': -1, 'C': 0, 'D': 0},
    'A - (C + D)/2': {'A': 1, 'B': 0, 'C': -0.5, 'D': -0.5},
    'B - D': {'A': 0, 'B': 1, 'C': 0, 'D': -1},
    'C - mean(others)': {'A': -1/3, 'B': -1/3, 'C': 1, 'D': -1/3},
}

results = []
for name, coef in contrasts.items():
    L = sum([coef[g] * means[g] for g in coef])
    SE = np.sqrt(MSE * sum((coef[g]**2) / n[g] for g in coef))
    F_scheffe = (L ** 2) / (SE ** 2) / (k - 1)
    F_crit = f.ppf(0.95, dfn=(k - 1), dfd=df_error)
    p_value = 1 - f.cdf(F_scheffe * (k - 1), dfn=(k - 1), dfd=df_error)
    results.append({
        'Contrast': name,
        'L': L,
        'Standard Error': SE,
        'F (Scheffé)': F_scheffe,
        'F critical (α=0.05)': F_crit,
        'p-value': p_value,
        'Reject Null (α=0.05)': F_scheffe > F_crit
    })

print(pd.DataFrame(results))
