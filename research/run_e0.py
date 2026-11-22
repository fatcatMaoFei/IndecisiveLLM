"""Deterministic E0 sweep; numerical math sanity, not an LLM experiment."""
import json
import random
from pathlib import Path
from statistics import mean
from indecisive.core import Signal, Task, evaluate, choose


def run():
    r = random.Random(20261122)
    tasks = [Task(.5, (Signal('single', .8, .2, .1),), max_queries=1)]
    for k in range(250):
        signals = tuple(Signal(f'q{i}', r.uniform(.51, .99), r.uniform(.01, .49),
                               r.uniform(.005, .25)) for i in range(4))
        tasks.append(Task(r.uniform(.1, .9), signals,
                          gamma=r.uniform(.85, 1.0), max_queries=3))
    rows = []
    for k, task in enumerate(tasks):
        greedy, short, exact = (evaluate(task, mode) for mode in ('greedy', 'one_step', 'exact'))
        action, value = choose(task, task.prior, tuple(range(len(task.signals))),
                               task.max_queries)
        rows.append(dict(id=k, greedy=greedy, one_step=short, exact=exact,
                         gain_over_greedy=exact-greedy, gain_over_one_step=exact-short,
                         initial_action=action, horizon=task.max_queries,
                         planner_self_check_error=abs(value-exact)))
    summary = {
        'experiment': 'E0 exact binary latent-state planning (NOT LLM)',
        'seed': 20261122, 'n_tasks': len(tasks), 'generated_task_count': len(tasks)-1,
        'max_self_check_error': max(a['planner_self_check_error'] for a in rows),
        'average_greedy': mean(a['greedy'] for a in rows),
        'average_one_step': mean(a['one_step'] for a in rows),
        'average_exact': mean(a['exact'] for a in rows),
        'average_gain_exact_vs_greedy': mean(a['gain_over_greedy'] for a in rows),
        'average_gain_exact_vs_one_step': mean(a['gain_over_one_step'] for a in rows),
        'n_strict_gain_over_one_step': sum(a['gain_over_one_step'] > 1e-9 for a in rows),
        'n_strict_gain_over_greedy': sum(a['gain_over_greedy'] > 1e-9 for a in rows),
        'interpretation': 'Sanity check of an exact model on synthetic known dynamics; not evidence of improvement over LLM agents or unknown-world generalization.'
    }
    result={'summary': summary, 'rows': rows}
    p=Path(__file__).resolve().parent.parent/'results'/'e0_seed20261122.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    run()
