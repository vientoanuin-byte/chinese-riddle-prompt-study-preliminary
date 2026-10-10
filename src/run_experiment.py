"""Sequential, rate-limited A/B experiment using a chat-completions-compatible API."""
import argparse
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import random
import time
import requests
from common import ROOT, SYSTEM, GUIDANCE, CONTROL, load_data, digest, extract_answer, build_messages, normalize_type, balanced_schedule

FIELDS = ['run_id', 'riddle_id', 'group', 'position', 'group_order', 'question', 'answer', 'type', 'source', 'split', 'component_count',
          'radical_count', 'structure_source', 'response', 'predicted_answer', 'correct',
          'status', 'attempts', 'http_status', 'latency_seconds', 'finish_reason',
          'model_requested', 'model_returned', 'prompt_tokens', 'completion_tokens', 'timestamp_utc']

class ChatClient:
    def __init__(self, base_url, key, rpm, retries, timeout):
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        self.interval, self.retries, self.timeout = 60 / rpm, retries, timeout
        self.last = 0.0
        self.session = requests.Session()

    def call(self, payload):
        start, status = time.monotonic(), ''
        for attempt in range(1, self.retries + 2):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            delay = 0
            try:
                response = self.session.post(self.url, headers=self.headers, json=payload, timeout=self.timeout, allow_redirects=False)
                status = response.status_code
                if status == 200:
                    obj = response.json()
                    choice = obj['choices'][0]
                    content = choice['message'].get('content')
                    if not isinstance(content, str) or not content.strip():
                        return dict(status='empty_response', attempts=attempt, http_status=status)
                    return dict(status='ok' if choice.get('finish_reason') == 'stop' else 'incomplete_response',
                                response=content, attempts=attempt, http_status=status,
                                latency_seconds=round(time.monotonic() - start, 3),
                                finish_reason=choice.get('finish_reason', ''),
                                model_returned=obj.get('model', ''),
                                prompt_tokens=obj.get('usage', {}).get('prompt_tokens', ''),
                                completion_tokens=obj.get('usage', {}).get('completion_tokens', ''))
                if status not in {408, 429, 500, 502, 503, 504}:
                    # 保留服务商返回的简短错误正文；这对火山方舟的 404
                    # （模型名与推理接入点 ID 不一致）尤其重要。
                    body = response.text[:1000]
                    return dict(status='http_error', attempts=attempt, http_status=status,
                                error_body=body)
                retry_after = response.headers.get('Retry-After', '0')
                try:
                    delay = max(0, float(retry_after))
                except ValueError:
                    try:
                        delay = max(0, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError):
                        delay = 0
            except (requests.Timeout, requests.ConnectionError):
                status = 'network_error'
            except requests.RequestException:
                return dict(status='request_error', attempts=attempt, http_status='request_error')
            except (ValueError, KeyError, IndexError, TypeError):
                return dict(status='invalid_response', attempts=attempt, http_status=status)
            if attempt <= self.retries:
                time.sleep(max(delay, min(60, 2 ** (attempt - 1)) + random.random()))
        return dict(status='api_error', attempts=attempt, http_status=status)

def make_payload(model, messages, temperature=0, max_tokens=256, token_parameter='max_tokens', extra_body=None):
    extra = extra_body or {}
    if not isinstance(extra, dict) or set(extra) - {'thinking', 'enable_thinking'}:
        raise ValueError('Only thinking and enable_thinking are allowed in extra_body')
    if token_parameter not in {'max_tokens', 'max_completion_tokens'}:
        raise ValueError('Invalid token parameter')
    return dict(model=model, messages=messages, temperature=temperature,
                **{token_parameter: max_tokens}, **extra)

def main(argv=None, api_key=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/test.json')
    p.add_argument('--structure', type=Path, default=ROOT / 'data/structure.json')
    p.add_argument('--annotations', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'output/experiment_results.csv')
    p.add_argument('--base-url', default=os.getenv('LLM_BASE_URL', ''))
    p.add_argument('--model', default=os.getenv('LLM_MODEL', ''))
    p.add_argument('--rpm', type=float, default=20)
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--timeout', type=float, default=90)
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--token-parameter', choices=['max_tokens', 'max_completion_tokens'], default='max_tokens')
    p.add_argument('--extra-body', default='{}', help='JSON with provider thinking settings (no secrets)')
    p.add_argument('--seed', type=int, default=20261009, help='Local ordering seed; not an API determinism guarantee')
    p.add_argument('--dry-run', action='store_true', help='Save prompt preview only; never fabricate responses')
    args = p.parse_args(argv)
    extra_body = json.loads(args.extra_body)
    make_payload(args.model, [], token_parameter=args.token_parameter, extra_body=extra_body)
    if args.rpm <= 0 or args.retries < 0 or args.timeout <= 0 or args.max_tokens <= 0:
        p.error('Invalid rate/retry/timeout/token parameter')
    data = load_data(args.data)
    if args.annotations:
        with args.annotations.open(encoding='utf-8-sig', newline='') as f:
            annotations = list(csv.DictReader(f))
        labels = {str(r['riddle_id']): r['type'].strip() for r in annotations}
        if len(labels) != len(annotations) or set(labels) != {r['riddle_id'] for r in data}:
            p.error('Annotation IDs must match the dataset exactly, without duplicates')
        for r in data:
            label = labels[r['riddle_id']]
            r['type'] = normalize_type(label)
    structure = json.loads(args.structure.read_text(encoding='utf-8-sig')) if args.structure.exists() else {}
    schedule = balanced_schedule(data, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        preview = args.output.with_suffix('.prompts.json')
        preview.write_text(json.dumps([{'riddle_id': r['riddle_id'], 'group': g,
            'messages': build_messages(r['question'], g)} for r, g, _, _ in schedule], ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Preview only: {preview}; {len(schedule)} calls planned, none sent.')
        return
    key = api_key or os.getenv('LLM_API_KEY')
    if not key or not args.model or not args.base_url:
        p.error('Set LLM_API_KEY, LLM_MODEL, and LLM_BASE_URL (base URL includes /v1 when required)')
    if not args.base_url.startswith('https://') and not args.base_url.startswith(('http://localhost:', 'http://127.0.0.1:')):
        p.error('Use HTTPS, or an explicitly configured local HTTP endpoint')
    config = dict(data=data, structure=structure, base_url=args.base_url, model=args.model,
                  temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                  system=SYSTEM, guidance=GUIDANCE, control=CONTROL, protocol_version=3,
                  extra_body=extra_body, token_parameter=args.token_parameter)
    run_id = digest(config)
    manifest_path = args.output.with_suffix('.manifest.json')
    done = set()
    if args.output.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding='utf-8'))['run_id'] != run_id:
            p.error('Existing output uses another configuration or lacks its manifest; choose a new output file')
        with args.output.open(encoding='utf-8-sig', newline='') as f:
            for old in csv.DictReader(f):
                pair = (old['riddle_id'], old['group'])
                if old['run_id'] != run_id or pair in done:
                    p.error('Mixed run IDs or duplicate rows in output')
                done.add(pair)
    else:
        manifest_path.write_text(json.dumps({'run_id': run_id, 'config': config,
            'rpm': args.rpm, 'retries': args.retries, 'timeout': args.timeout,
            'created_utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2), encoding='utf-8')
    client = ChatClient(args.base_url, key, args.rpm, args.retries, args.timeout)
    new_file = not args.output.exists()
    with args.output.open('a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for row, group, position, group_order in schedule:
            if (row['riddle_id'], group) in done:
                continue
            result = client.call(make_payload(args.model, build_messages(row['question'], group),
                args.temperature, args.max_tokens, args.token_parameter, extra_body))
            pred = extract_answer(result.get('response', ''))
            status = result['status']
            if status == 'ok' and not pred:
                result['status'] = 'no_answer'
            features = structure.get(row['answer'], {})
            record = {**row, **{k: features.get(k, '') for k in ['component_count', 'radical_count', 'structure_source']},
                      **result, 'run_id': run_id, 'group': group, 'model_requested': args.model,
                      'position': position, 'group_order': group_order,
                      'predicted_answer': pred,
                      'correct': int(pred == row['answer']) if result['status'] in {'ok', 'no_answer'} else '',
                      'timestamp_utc': datetime.now(timezone.utc).isoformat()}
            writer.writerow(record)
            f.flush()
            os.fsync(f.fileno())
            print(f"{row['riddle_id']} {group}: {result['status']}", flush=True)
            if result.get('http_status') in {400, 401, 403, 404, 422}:
                raise SystemExit('Permanent API configuration/authentication error; stopped. Fix config and use a new output file.')
    print(f'Saved {args.output}. Recorded failures remain auditable and are not silently retried on resume.')

if __name__ == '__main__':
    main()
