"""Simplistic YAML loader to avoid external dependencies."""

from typing import Any, Dict


def load(path: str) -> Dict[str, Any]:
    """Load a subset of YAML syntax from ``path``.

    This parser supports mappings, simple lists, anchors and references.
    It falls back to the full PyYAML parser if available.
    """
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ModuleNotFoundError:
        return _load_without_yaml(path)


def _load_without_yaml(path: str) -> Dict[str, Any]:
    anchors: Dict[str, Any] = {}
    root: Dict[str, Any] = {}
    stack = [(0, root)]  # list of (indent, container)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            while stack and indent < stack[-1][0]:
                stack.pop()
            container = stack[-1][1]
            if ":" not in line:
                continue
            key, value = [p.strip() for p in line.lstrip().split(":", 1)]
            if value == "":
                new_dict: Dict[str, Any] = {}
                container[key] = new_dict
                stack.append((indent + 2, new_dict))
                continue
            if value.startswith("*"):
                container[key] = anchors.get(value[1:], value)
                continue
            if "&" in value:
                val, anchor = value.split("&", 1)
                val = val.strip()
                parsed = _parse_value(val)
                anchors[anchor.strip()] = parsed
                container[key] = parsed
                continue
            container[key] = _parse_value(value)
    return root


def _parse_value(text: str) -> Any:
    text = text.strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.lower() == "null":
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.isdigit():
        return int(text)
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("[") and text.endswith("]"):
        items = [i.strip() for i in text[1:-1].split(",") if i.strip()]
        return [_parse_value(i) for i in items]
    return text

