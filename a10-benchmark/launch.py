#!/usr/bin/env python3
"""Launch llama-server and retain its configuration and logs."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--model', default='models/kimi-k3/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf')
    p.add_argument('--store', default='models/kimi-k3/kimi-k3.experts')
    p.add_argument('--cache-mib', type=int, default=102400)
    p.add_argument('--gpu-layers', type=int, default=20)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--context-per-slot', type=int, default=2048)
    p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--second-chance', type=int, default=0)
    p.add_argument('--reader', choices=['uring','pread'], default='uring')
    p.add_argument('--startup-timeout', type=int, default=1800)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if min(args.cache_mib,args.threads,args.context_per_slot,args.parallel,args.startup_timeout) <= 0 or args.gpu_layers < 0:
        p.error('Memory, thread, context and time limits must be positive')
    repo = args.repo.resolve()
    executable = repo/'build/bin/llama-server'
    model, store = repo/args.model, repo/args.store
    command = [str(executable), '-m', str(model), '-ngl', str(args.gpu_layers),
               '--n-cpu-moe','93','--moe-cache-mib',str(args.cache_mib),
               '--load-mode','mmap','--fit','off','--no-warmup','--no-context-shift',
               '-c',str(args.context_per_slot*args.parallel),'--parallel',str(args.parallel),
               '-b','32','-ub','32','--threads',str(args.threads),
               '--host','127.0.0.1','--port',str(args.port),'--metrics','-lv','4']
    overrides = {'LLAMA_EXPERT_STORE': str(store), 'LLAMA_EXPERT_STORE_KEEP_PAGE_CACHE':'1',
                 'LLAMA_EXPERT_STORE_MLOCK':'0','LLAMA_EXPERT_STORE_GPU':'0',
                 'LLAMA_EXPERT_STORE_SECOND_CHANCE':str(args.second_chance),
                 'LLAMA_EXPERT_STORE_URING':'1' if args.reader=='uring' else '0'}
    config = {'command':command,'environment':overrides,'context_per_slot':args.context_per_slot}
    if args.dry_run:
        console.print_json(data=config)
        return 0
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            p.error(f'Port {args.port} is already in use; choose another port')
    for path in (executable,model,store):
        if not path.is_file():
            p.error(f'Missing file: {path}')
    # Do not inherit unrelated experimental flags from the shell.
    env = {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_EXPERT_STORE','GGML_CUDA_FORCE_HOST_BANK'))}
    env.update(overrides)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder = repo/'benchmark-results'/f'server-{stamp}'
    folder.mkdir(parents=True)
    config['git_revision'] = subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip()
    (folder/'launch.json').write_text(json.dumps(config,indent=2))
    console.print(f'[cyan]Starting server. Log: {folder / "server.log"}[/]')
    process = None
    try:
        with (folder/'server.log').open('w') as log, (folder/'resources.jsonl').open('w') as telemetry:
            process = subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            began = time.monotonic()
            ready = False
            with httpx.Client(timeout=2,trust_env=False) as client, Progress(SpinnerColumn(),TextColumn('{task.description}'),TimeElapsedColumn(),console=console) as progress:
                task = progress.add_task('Loading Kimi K3 (no percentage available)',total=None)
                while process.poll() is None:
                    if not ready:
                        try:
                            response = client.get(f'http://127.0.0.1:{args.port}/health')
                            if response.status_code == 200:
                                ready = True
                                progress.update(task,description=f'Server ready on port {args.port}; Ctrl+C stops it')
                                console.print('[green]Ready. Run benchmark.py in a second terminal.[/]')
                        except httpx.HTTPError:
                            pass
                        if not ready and time.monotonic()-began > args.startup_timeout:
                            raise TimeoutError('Server startup timed out; inspect server.log')
                    sample = {'unix_time':time.time(),'server_pid':process.pid}
                    for name,path in [('process_status',f'/proc/{process.pid}/status'),('process_io',f'/proc/{process.pid}/io'),('host_memory','/proc/meminfo')]:
                        try:
                            sample[name] = Path(path).read_text()
                        except OSError:
                            pass
                    try:
                        gpu = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                        sample['gpu_csv'] = gpu.stdout.strip()
                    except (OSError,subprocess.TimeoutExpired):
                        pass
                    telemetry.write(json.dumps(sample)+'\n')
                    telemetry.flush()
                    time.sleep(2)
            return process.returncode
    except KeyboardInterrupt:
        console.print('[yellow]Stopping server.[/]')
        return 130
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()

if __name__=='__main__':
    raise SystemExit(main())
