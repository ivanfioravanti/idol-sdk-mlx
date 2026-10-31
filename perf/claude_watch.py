import json, sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
ev = []
for line in open('perf/claude_run.log'):
    try: d = json.loads(line)
    except Exception: continue
    if d.get('type') == 'assistant':
        for c in d['message'].get('content', []):
            if c.get('type') == 'text': ev.append('TEXT: ' + c['text'][:300].replace('\n', ' '))
            elif c.get('type') == 'tool_use':
                i = c['input']; ev.append(f"{c['name']}: " + str(i.get('command') or i.get('file_path') or i)[:200].replace('\n', ' '))
    elif d.get('type') == 'result':
        ev.append(f"RESULT: {d.get('subtype')} turns={d.get('num_turns')} cost={d.get('total_cost_usd')} models={list((d.get('modelUsage') or {}).keys())}\n" + str(d.get('result'))[:3000])
    elif d.get('type') == 'system' and d.get('subtype') == 'init': ev.append(f"INIT model={d.get('model')}")
print('\n'.join(ev[:1] + ev[-n:]))
