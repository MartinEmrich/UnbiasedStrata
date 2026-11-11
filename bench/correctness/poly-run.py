#!/usr/bin/env python3
"""Configuration B arms (CORRECTNESS-PLAN.md 3.3): a fresh strata-poly process per sequence, teacher-forced on the
reference answer, top-20 probabilities per step.  poly-run.py <file-tag> <gguf> <arm> [seq ...]
arms: base   = --tail-skip 0 --no-draft --lookup 0 --window 1          (arm 1)
      draft  = --tail-skip 0 (draft block, lookup 3, window 8 defaults) (arm 2; top.txt must equal base's bytes)
      kvq8   = base + --kv q8_0                                          (arm 3)
      k16v8  = base + --kv k16v8 (fp16 keys, Q8_0 values; T2 against kvq8 and base)
      pb512  = base + --prompt-batch 512                                 (3.3 last paragraph, step 0 only)
Writes poly/<file-tag>/<arm>/<seq>.top.txt, .ids.txt, .log, command.sh."""
import os, pathlib, subprocess, sys, time
# US2_CORR_ROOT / US2_POLY: another model's root and engine binary (DeepSeek, GLM: 2026-10-19)
ROOT = pathlib.Path(os.environ.get('US2_CORR_ROOT', pathlib.Path.home() / 'us2-results/correctness'))
EXE = pathlib.Path(os.environ.get('US2_POLY', pathlib.Path.home() / 'us2-build/build/bin/strata-poly'))
tag, gguf, arm = sys.argv[1], sys.argv[2], sys.argv[3]; only = set(sys.argv[4:])

def answer_path(p, seq, tag):
    """The scored answer: the fixed one (<seq>.answer.ids, the real-text set) when it exists, else the reference
    model's own greedy answer <seq>.answer.<tag>.ids (a different file is a different model, so a different answer)."""
    fixed = p.parent / f'{seq}.answer.ids'
    return fixed if fixed.exists() else p.parent / f'{seq}.answer.{tag}.ids'
ARMS = {'base': ['--tail-skip', '0', '--no-draft', '--lookup', '0', '--window', '1'],
        'draft': ['--tail-skip', '0'],
        'kvq8': ['--tail-skip', '0', '--no-draft', '--lookup', '0', '--window', '1', '--kv', 'q8_0'],
        'k16v8': ['--tail-skip', '0', '--no-draft', '--lookup', '0', '--window', '1', '--kv', 'k16v8'],
        'pb512': ['--tail-skip', '0', '--no-draft', '--lookup', '0', '--window', '1', '--prompt-batch', '512']}
OUT = ROOT / 'poly' / tag / arm; OUT.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, STRATA_FREEZE_EXPERTS='1')
for p in sorted(ROOT.glob('inputs/*/*.prompt.ids')):
    seq = p.name[:-len('.prompt.ids')]
    if only and seq not in only: continue
    ans_path = answer_path(p, seq, tag)
    if not ans_path.exists(): print(f'{seq}: no answer ids yet (reference GPU run missing)', flush=True); continue
    prompt = p.read_text().strip(); ans = ans_path.read_text().strip()
    n = len(ans.split(','))
    cmd = [str(EXE), '--model', gguf, '--max-context', '32768'] + ARMS[arm] + \
          ['--tokens', f'@{p}', '--follow', f'@{ans_path}', '--max-new', str(n), '--top-probs', '20']
    t0 = time.time()
    try:
        with open(OUT / f'{seq}.top.txt', 'w') as err, open(OUT / f'{seq}.ids.txt', 'w') as out:
            r = subprocess.run(cmd, env=env, stdout=out, stderr=err)
    except OSError as e:   # E2BIG: a 31K-id --tokens argument is over Linux's 128 KiB single-argument limit
        print(f'{p.parent.name}/{seq} {arm}: cannot exec ({e}); the id list needs a file option', flush=True)
        (OUT / f'{seq}.log').write_text(f'not run: {e}\n'); continue
    with open(OUT / 'command.sh', 'a') as f:
        f.write('STRATA_FREEZE_EXPERTS=1 ' + ' '.join(cmd) + '\n')
    # top lines go to stderr with the engine's log; keep the log lines apart from the top lines
    raw = (OUT / f'{seq}.top.txt').read_text().splitlines()
    (OUT / f'{seq}.log').write_text('\n'.join(l for l in raw if not l.startswith('top ')) + '\n')
    (OUT / f'{seq}.top.txt').write_text('\n'.join(l for l in raw if l.startswith('top ')) + '\n')
    steps = sum(1 for l in raw if l.startswith('top '))
    print(f'{p.parent.name}/{seq} {arm}: exit {r.returncode}, {steps} steps ({n} wanted), {time.time()-t0:.0f} s', flush=True)
