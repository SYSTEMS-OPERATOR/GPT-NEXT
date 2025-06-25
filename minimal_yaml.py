import ast
import re


def safe_load(stream):
    """Minimal YAML loader supporting a subset of YAML used in config files."""
    text = stream.read() if hasattr(stream, "read") else stream
    anchors = {}
    processed = []
    for line in text.splitlines():
        # Strip comments
        line = line.split('#')[0]
        if not line.strip():
            continue
        anchor_def = re.match(r'^(\s*)([^:]+):\s*&([^\s]+)\s+(.*)$', line)
        alias_def = re.match(r'^(\s*)([^:]+):\s*\*([^\s]+)\s*$', line)
        if anchor_def:
            indent, key, anc, val = anchor_def.groups()
            anchors[anc] = ast.literal_eval(val)
            line = f"{indent}{key}: {val}"
        elif alias_def:
            indent, key, anc = alias_def.groups()
            val = anchors.get(anc)
            val_repr = repr(val) if isinstance(val, str) else str(val)
            line = f"{indent}{key}: {val_repr}"
        processed.append(line)

    lines = processed
    idx = 0

    def parse_block(exp_indent=0):
        nonlocal idx
        obj = {}
        while idx < len(lines):
            line = lines[idx]
            indent = len(line) - len(line.lstrip())
            if indent < exp_indent:
                break
            if indent > exp_indent:
                idx += 1
                continue
            stripped = line.strip()
            if ':' not in stripped:
                idx += 1
                continue
            key, rest = stripped.split(':', 1)
            key = key.strip()
            rest = rest.strip()
            idx += 1
            if rest == '':
                if idx < len(lines):
                    next_indent = len(lines[idx]) - len(lines[idx].lstrip())
                else:
                    next_indent = exp_indent + 2
                obj[key] = parse_block(next_indent)
            else:
                obj[key] = ast.literal_eval(rest)
        return obj

    return parse_block(0)
