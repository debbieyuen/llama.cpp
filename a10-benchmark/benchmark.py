#!/usr/bin/env python3
"""Streaming benchmark for llama-server's native API. See README.md."""
import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
SMOKE = [
    ('What is 17 + 25? Reply with only the integer.\nAnswer:', '42'),
    ('Return exactly this JSON with no spaces or explanation: {"ok":true}\nAnswer:', '{"ok":true}'),
    ('Record: name=Ada; city=Oslo; age=31. Return only the city.\nAnswer:', 'Oslo'),
    ('Follow this instruction: output only the word READY in uppercase.\nAnswer:', 'READY'),
]

def percentile(values, p):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    i = (len(values) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)

def summarize(rows, seconds):
    good = [r for r in rows if r['status'] == 'ok']
    total = sum(r['output_tokens'] for r in good)
    result = dict(requests=len(rows), successful=len(good), failed=len(rows)-len(good),
                  wall_seconds=seconds, output_tok_s=total / seconds if seconds else None,
                  correctness_pass=sum(r.get('correct') is True for r in good),
                  correctness_fail=sum(r.get('correct') is False for r in good))
    for key in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'server_decode_tok_s', 'server_prompt_tok_s'):
        for p in (0.5, 0.95, 0.99):
            result[f'{key}_p{int(p*100)}'] = percentile([r.get(key) for r in good], p)
    return result

async def post(client, endpoint, data):
    r = await client.post(endpoint, json=data)
    r.raise_for_status()
    return r.json()

async def tokens(client, text):
    return (await post(client, '/tokenize', {'content': text, 'add_special': False}))['tokens']

async def make_case(client, target, kind, position):
    # Token IDs give exact prompt sizes without a guessed words-to-tokens ratio.
    filler = await tokens(client, ' A routine archive entry records ordinary weather and inventory updates.')
    if not filler:
        raise ValueError('Tokenizer returned no filler tokens')
    expected = None
    if kind == 'retrieval':
        expected = 'ORCHID-739241'
        needle = await tokens(client, '\nThe secret archive code is ORCHID-739241.\n')
        suffix = await tokens(client, '\nWhat is the secret archive code? Reply with only the code.\nAnswer:')
        room = target - len(needle) - len(suffix)
        if room < 0:
            raise ValueError(f'{target} tokens is too short for retrieval instructions')
        pad = (filler * (room // len(filler) + 1))[:room]
        offset = round(room * position)
        prompt = pad[:offset] + needle + pad[offset:] + suffix
    else:
        suffix = await tokens(client, '\nContinue with a detailed numbered description of the archive:\n')
        if target < len(suffix):
            raise ValueError('Input target too short')
        room = target - len(suffix)
        prompt = (filler * (room // len(filler) + 1))[:room] + suffix
    return prompt, expected

async def request(client, prompt, output, expected, args, request_id, destination):
    start = time.perf_counter()
    first = last = None
    text, token_ids, final = '', [], None
    row = {'request_id': request_id, 'status': 'error', 'input_tokens_sent': len(prompt),
           'output_budget': output, 'output_tokens': 0, 'correct': None}
    payload = {'prompt': prompt, 'n_predict': output, 'temperature': 0, 'seed': 0,
               'stream': True, 'cache_prompt': args.cache_prompt,
               'ignore_eos': expected is None, 'return_tokens': True}
    try:
        async with asyncio.timeout(args.timeout):
            async with client.stream('POST', '/completion', json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == '[DONE]':
                        break
                    event = json.loads(raw)
                    if 'error' in event:
                        raise RuntimeError(str(event['error']))
                    chunk = event.get('content', '')
                    ids = event.get('tokens', [])
                    if chunk or ids:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        text += chunk
                        token_ids.extend(ids)
                    if event.get('stop'):
                        final = event
        if final is None:
            raise RuntimeError('Stream ended without final completion event')
        timing = final.get('timings', {})
        count = timing.get('predicted_n', final.get('tokens_predicted', len(token_ids)))
        if not isinstance(count, int) or count <= 0 or first is None:
            raise RuntimeError('No generated tokens or token count available')
        row.update(status='ok', output_tokens=count,
                   actual_prompt_tokens=timing.get('prompt_n'),
                   ttft_ms=(first-start)*1000,
                   tpot_ms=(last-first)*1000/(count-1) if count > 1 else None,
                   server_decode_tok_s=timing.get('predicted_per_second'),
                   server_prompt_tok_s=timing.get('prompt_per_second'),
                   truncated=final.get('truncated'), stop_type=final.get('stop_type'),
                   correct=(text.strip() == expected) if expected is not None else None)
        if final.get('truncated'):
            row.update(status='error', error='Server truncated the input context')
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['e2e_ms'] = (time.perf_counter()-start)*1000
    row['text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    (destination / f'{request_id}.json').write_text(json.dumps(
        {'metrics': row, 'text': text, 'token_ids': token_ids, 'final': final}, indent=2))
    return row

async def snapshot(client, folder, name):
    try:
        r = await client.get('/metrics')
        r.raise_for_status()
        (folder / f'metrics-{name}.txt').write_text(r.text)
    except Exception as exc:
        (folder / f'metrics-{name}.txt').write_text(f'Unavailable: {type(exc).__name__}')

async def run(args):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = Path(args.results) / stamp
    root.mkdir(parents=True)
    manifest = vars(args).copy()
    manifest.update(created_utc=stamp, client_platform=platform.platform(),
                    metric_scope='native completion API; raw token prompts; client streaming latency',
                    environment_labels='User-supplied labels, not verified deployment facts')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    headers = {'Authorization': f'Bearer {os.environ[args.api_key_env]}'} if os.getenv(args.api_key_env) else {}
    summaries = []
    async with httpx.AsyncClient(base_url=args.url.rstrip('/'), headers=headers,
                                 timeout=args.timeout, limits=httpx.Limits(max_connections=max(args.concurrency)+4)) as client:
        health = await client.get('/health')
        health.raise_for_status()
        for endpoint in ('props', 'slots'):
            try:
                r = await client.get('/'+endpoint)
                r.raise_for_status()
                (root / f'{endpoint}.json').write_text(json.dumps(r.json(), indent=2))
            except Exception:
                pass
        if args.suite == 'smoke':
            cases = [('smoke', 0, args.output[0], 1, i) for i in range(len(SMOKE))]
        elif args.suite == 'retrieval':
            cases = [('retrieval', n, args.output[0], 1, p) for n in args.input for p in args.positions]
        else:
            cases = [('performance', n, o, c, 0.5) for n, o, c in itertools.product(args.input, args.output, args.concurrency)]
        console.print(f'[bold cyan]{args.label}[/] | {len(cases)} cases | results: {root}')
        with Progress(SpinnerColumn(), TextColumn('{task.description}'), BarColumn(),
                      TextColumn('{task.completed}/{task.total}'), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task('Benchmark cases', total=len(cases))
            for idx, (kind, n, out, concurrency, pos) in enumerate(cases):
                if n + out > args.context_per_slot:
                    summaries.append(dict(case=idx, status='skipped', reason='Input + output exceeds context per slot', input=n, output=out, concurrency=concurrency))
                    progress.advance(task)
                    continue
                folder = root / f'case-{idx:04d}'
                folder.mkdir()
                expected = None
                if kind == 'smoke':
                    question, expected = SMOKE[int(pos)]
                    prompt = await tokens(client, question)
                else:
                    prompt, expected = await make_case(client, n, kind, pos)
                (folder / 'prompt.json').write_text(json.dumps({'tokens': prompt, 'expected': expected, 'needle_position_fraction_of_filler': pos}))
                if len(prompt) + out > args.context_per_slot:
                    raise ValueError('Smoke prompt plus output budget exceeds context per slot')
                for w in range(args.warmups):
                    warm = await request(client, prompt, out, expected, args, f'warmup-{w}', folder)
                    if warm['status'] != 'ok':
                        raise RuntimeError(f'Warmup failed; see {folder}: {warm.get("error")}')
                await snapshot(client, folder, 'before')
                started = time.perf_counter()
                counter = itertools.count()
                rows = []
                req_task = progress.add_task(f'Case {idx}: in={len(prompt)} out={out} C={concurrency}', total=None if args.suite == 'stability' else args.requests)
                async def worker():
                    while True:
                        number = next(counter)
                        if args.suite == 'stability':
                            if time.perf_counter()-started >= args.duration:
                                return
                        elif number >= args.requests:
                            return
                        row = await request(client, prompt, out, expected, args, f'request-{number:06d}', folder)
                        rows.append(row)
                        with (folder / 'requests.jsonl').open('a') as f:
                            f.write(json.dumps(row)+'\n')
                        progress.advance(req_task)
                await asyncio.gather(*(worker() for _ in range(concurrency)))
                elapsed = time.perf_counter()-started
                await snapshot(client, folder, 'after')
                summary = summarize(rows, elapsed)
                summary.update(case=idx, suite=args.suite, input=len(prompt), output=out,
                               concurrency=concurrency, needle_position=pos if kind=='retrieval' else None,
                               unique_texts=len({r['text_sha256'] for r in rows if r['status']=='ok'}))
                summary['repeatability_pass'] = summary['unique_texts'] == 1 and summary['failed'] == 0 if args.suite == 'repeatability' else None
                summaries.append(summary)
                (root / 'summary.json').write_text(json.dumps(summaries, indent=2))
                console.print(f'[green]Completed[/] case {idx}: {summary["successful"]}/{len(rows)} successful; {summary["output_tok_s"]:.2f} output tok/s')
                progress.advance(task)
    (root / 'summary.json').write_text(json.dumps(summaries, indent=2))
    keys = sorted({k for row in summaries for k in row})
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    table = Table(title='Results (warmups excluded)')
    for title in ('Case', 'Input', 'Output', 'C', 'OK / Failed', 'tok/s', 'TTFT p50 ms', 'Correct / Wrong'):
        table.add_column(title)
    for r in summaries:
        table.add_row(str(r['case']), str(r['input']), str(r['output']), str(r['concurrency']),
                      f'{r.get("successful",0)} / {r.get("failed",0)}',
                      f'{r.get("output_tok_s",0):.2f}', str(r.get('ttft_ms_p50', 'skipped')),
                      f'{r.get("correctness_pass",0)} / {r.get("correctness_fail",0)}')
    console.print(table)
    console.print(f'[bold cyan]Saved {root}[/]')
    return 1 if any(r.get('failed',0) or r.get('correctness_fail',0) or r.get('repeatability_pass') is False for r in summaries) else 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--label', default='kimi-k3-a10')
    p.add_argument('--suite', choices=['smoke','performance','retrieval','stability','repeatability'], default='smoke')
    p.add_argument('--input', nargs='+', type=int, default=[64,1024,4096,8192,32768,65536,98304,130000])
    p.add_argument('--output', nargs='+', type=int, default=[256,512,2048])
    p.add_argument('--concurrency', nargs='+', type=int, default=[1])
    p.add_argument('--positions', nargs='+', type=float, default=[0.1,0.5,0.9])
    p.add_argument('--requests', type=int, default=5)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--duration', type=float, default=600)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--context-per-slot', type=int, required=True)
    p.add_argument('--cache-prompt', action='store_true')
    p.add_argument('--results', default='benchmark-results')
    p.add_argument('--api-key-env', default='LLAMA_API_KEY')
    args = p.parse_args()
    if min(args.input+args.output+args.concurrency+[args.requests,args.context_per_slot]) <= 0 or args.warmups < 0 or args.timeout <= 0 or args.duration <= 0 or any(not 0<=v<=1 for v in args.positions):
        p.error('Counts and durations must be positive; positions must be between 0 and 1')
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        console.print('[yellow]Interrupted; completed request files remain on disk.[/]')
        return 130
    except Exception as exc:
        console.print(f'{type(exc).__name__}: {exc}', style='red', markup=False)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
