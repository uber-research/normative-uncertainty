# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import json
import os
import glob
import subprocess
import fire
import hashlib
import multiprocessing

def gethash(v):
    return hashlib.sha1(v.encode('utf8')).hexdigest()

def get_exp_suffix(exp):
    return '_'.join((f'{k}-{v}' if k != 'theories' else f'{k}-{gethash(v)}') for k, v in sorted(exp.items()))

def get_img_filename(n_on_track, n_credences):
    return f'ot-{n_on_track}_nc-{n_credences}'

def process_experiment(exp, cachedir, timesteps, n_on_track, n_credences):
    suffix = get_exp_suffix(exp)
    outdir = cachedir + '/' + suffix
    if glob.glob(outdir + '/final_net*') == []:
        subprocess.run([
            'python',
            'freeform_voter.py',
            'train_trolley',
            '--num_timesteps', str(timesteps),
            '--save_to', outdir
        ] + [
            f'--{k}={v}' for k, v in exp.items()
        ])

    filename = get_img_filename(n_on_track, n_credences)
    if not os.path.exists(f'{outdir}/{filename}.png'):
        subprocess.run([
            'python',
            'freeform_voter.py',
            'test_trolley',
            '--load_from', outdir,
            '--n_on_track', str(n_on_track),
            '--n_credences', str(n_credences),
            '--suffix_name', filename
        ])

def run(timesteps, n_on_track, n_credences, processes=1):
    suffix = f'ts-{timesteps}'
    cachedir = '_results/_cache/' + suffix
    resdir = f'_results/ts-{timesteps}/ot-{n_on_track}_nc-{n_credences}'
    os.makedirs(cachedir, exist_ok=True)
    experiments = [(f, json.dumps(json.load(open(f)), sort_keys=True)) for f in glob.glob('_experiments/*/*.json')]
    unique_experiments = [json.loads(a) for a in set([e for _, e in experiments])]
    if processes == 1:
        for exp in unique_experiments:
            process_experiment(exp, cachedir, timesteps, n_on_track, n_credences)
    else:
        pool = multiprocessing.Pool(processes)
        pool.starmap(process_experiment, [(exp, cachedir, timesteps, n_on_track, n_credences) for exp in unique_experiments])

    for f, exp in experiments:
        exp = json.loads(exp)
        f = f.replace('_experiments/', '').replace('.json', '')
        os.makedirs(resdir + '/' + f, exist_ok=True)
        base = f'{cachedir}/{get_exp_suffix(exp)}/'
        for file in glob.glob(base + '*.png') + glob.glob(base + '*.pdf'):
            outfile = f'{resdir}/{f}/{file.split("__")[-1]}'
            open(outfile, 'wb').write(open(file, 'rb').read())

if __name__ == '__main__':
    fire.Fire(run)
