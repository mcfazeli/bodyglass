"""Every `${...}` that app.js writes into HTML must go through escH() or be a listed safe site.

HTML sinks are `x.innerHTML = ...` (and `+=`), `insertAdjacentHTML(pos, ...)` and the first argument of
`toast(...)`, which shows its message as HTML. Inside a sink the check looks at template-literal
interpolations (nested templates are checked too) and at a bare value such as `toast(r.error)`.
It reads the source as text: no browser, Node, network or extra dependencies.
shortcut: markup first built into a variable (e.g. `prog`) is checked only where it is built if that is
itself a sink; list such variables in SAFE after reading where they come from. Upgrade to a real JS parser
if app.js grows many of them.
"""
import pathlib
import re

APP = pathlib.Path(__file__).resolve().parents[1] / "bodyglass/web/app.js"

# Interpolations that are safe without escH(), with the reason. Keys are the exact text inside `${...}`.
# Never list DICOM-derived text (descriptions, kinds, UIDs), error or exception messages, or file names here.
SAFE = {
    "msg": "toast()'s own parameter; every toast() call is checked at its call site",
    "title": "model-menu label; the caller escapes the model name",
    "sub": "model-menu subtitle; the caller escapes the e-mail address",
    "fmtAnswer(ans)": "AI answer: fmtAnswer escapes it with escH first, then adds its own <b>/<i>",
    "L.def.color": "layer colour from the built-in layer table",
    "L.def.name": "layer name from the built-in layer table",
    "L.st.color": "structure colour from the built-in structure table",
    "sts[0].color": "structure colour from the built-in structure table",
    "s.id": "numeric structure label id",
    "nm[0].toUpperCase() + nm.slice(1)": "structure group name from the built-in structure table",
    "role": "series role key chosen in app.js",
    "ROLE[role][0]": "series role label from the built-in ROLE table",
    "ROLE[role][1]": "series role label from the built-in ROLE table",
    "k": "scan chip key from the built-in chip list",
    "q.kind": "sequence key from the built-in sequence table",
    "q.name": "sequence name from the built-in sequence table",
    "x.n": "command name from the built-in command palette",
    "x.k || ''": "keyboard shortcut from the built-in command palette",
    "x.name": "structure name from the built-in search index",
    "x.sub": "structure group from the built-in search index",
    "n": "preset or chip name from a built-in table",
    "l": "preset layer key from the built-in preset table",
    "t": "progress step text from the built-in STEPS table",
    "c.icon": "chip icon emoji chosen in app.js",
    "c.image": "data: URL of a picture app.js drew or the user picked; it holds no quote",
    "legend.replace('Blend: ', '')": "colour legend from the built-in sequence table",
    "p.name": "fold name from the built-in atlas table",
    "p.text": "fold explanation from the built-in atlas table",
    "label": "focus label: a structure name from the built-in structure table",
    "sel.name": "structure name from the built-in structure table",
    "ex.info.name": "example-brain credit shipped with the app",
    "ex.info.license": "example-brain licence shipped with the app",
    "ex.info.credit": "example-brain credit shipped with the app",
    "info.credit": "example-volunteer credit shipped with the app",
    "info.license": "example-volunteer licence shipped with the app",
    "prog": "setup progress markup renderWelcome builds from numbers, fixed text and escH(s.error)",
    "btn": "setup button markup chosen in renderWelcome",
    "a.download": "export name app.js builds from a structure name and the date",
    "what": "model capability text chosen in app.js",
    "FIX[m.fix]": "button label from the built-in FIX table",
    "s.variant": "model variant key ('4' or '8') from the built-in model list",
    "kind": "toast style class chosen by the caller",
}
SAFE_PATTERNS = [
    (r"(fmtMl|fmtDate|fmtMin|gb|Math\.round)\(.*\)", "number or date formatted by app.js"),
    (r".+\.toFixed\(.*\)", "number formatted by toFixed"),
    (r".+\.length", "a count"),
    (r"new Date\(.*\)\.toLocale(Date)?String\(\)", "date formatted by the browser"),
    (r"x|y|z|b|i|pct|tA|tB|info\.bodies|p\.downloadGB|p\.diskGB|p\.ramGB|s\.pct", "number computed by app.js"),
]

QUOTES = "'\""
REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^") | {""}


def scan_string(src, i):
    """Index just past the string or regex literal that starts at i."""
    q = src[i]
    i += 1
    while src[i] != q:
        i += 2 if src[i] == "\\" else 1
        if q == "/" and src[i - 1] == "[":   # regex character class may hold a slash
            while src[i] != "]":
                i += 2 if src[i] == "\\" else 1
    return i + 1


def scan_template(src, i, found):
    """Parse the template literal starting at src[i] == '`'; record (offset, text) of each interpolation."""
    i += 1
    while src[i] != "`":
        if src[i] == "\\":
            i += 2
        elif src.startswith("${", i):
            start = i + 2
            i = scan_code(src, start, found, stop="}")
            found.append((start, src[start:i]))
            i += 1
        else:
            i += 1
    return i + 1


def skip_literal(src, i, prev, found=None):
    """If a string, regex or template literal starts at i, the index just past it; else None."""
    c = src[i]
    if c in QUOTES or (c == "/" and prev in REGEX_BEFORE and src[i + 1:i + 2] not in ("/", "*")):
        return scan_string(src, i)
    if c == "`":
        return scan_template(src, i, [] if found is None else found)
    return None


def scan_code(src, i, found, stop):
    """Walk code from i to the first unnested character in `stop`; collect interpolations of templates seen."""
    depth, prev = 0, ""
    while i < len(src):
        c = src[i]
        if depth == 0 and c in stop:
            if c != "\n" or not continues(src, i, prev):
                return i
        end = skip_literal(src, i, prev, found)
        if end:
            i, prev = end, "x"
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return i
            depth -= 1
        if not c.isspace():
            prev = c if not (c.isalnum() or c in "_$") else "x"
        i += 1
    return i


def continues(src, i, prev):
    """A newline inside an expression: does the statement continue on the next line?"""
    nxt = src[i:].lstrip()[:1]
    return prev in set("+?:|&,(") or nxt in set("+?:|&.")


def strip_parens(text):
    text = text.strip()
    while text.startswith("(") and text.endswith(")") and balanced(text[1:-1]):
        text = text[1:-1].strip()
    return text


def balanced(text):
    depth = 0
    for c in re.sub(r"'[^']*'|\"[^\"]*\"", "", text):
        depth += {"(": 1, ")": -1}.get(c, 0)
        if depth < 0:
            return False
    return depth == 0


def top_level_split(text, seps):
    """Split code at operators in `seps` that sit outside brackets, strings, regexes and templates."""
    parts, depth, start, i, prev = [], 0, 0, 0, ""
    while i < len(text):
        c = text[i]
        end = skip_literal(text, i, prev)
        if end:
            i, prev = end, "x"
            continue
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(c, 0)
        if depth == 0 and c in seps:
            parts.append(text[start:i])
            start = i + 1
        if not c.isspace():
            prev = c if not (c.isalnum() or c in "_$") else "x"
        i += 1
    return parts + [text[start:]]


def is_literal(text):
    text = strip_parens(text)
    return bool(re.fullmatch(r"'(\\.|[^'\\])*'|\"(\\.|[^\"\\])*\"", text)) or (
        text.startswith("`") and text.endswith("`") and scan_template(text, 0, []) == len(text))


def safe(text):
    """True if this interpolated or bare value cannot carry outside text into HTML."""
    text = strip_parens(text)
    if text in SAFE or is_literal(text) or (text.startswith("escH(") and text.endswith(")") and balanced(text[5:-1])):
        return True
    # cond ? a : b, a || b, a + b: every value part must be safe (a condition is never shown).
    branches = top_level_split(text, "?")
    if len(branches) > 1:
        values = top_level_split("?".join(branches[1:]), ":")
        return len(values) > 1 and safe(values[0]) and safe(":".join(values[1:]))
    for sep in "+|":
        parts = [p for p in top_level_split(text, sep) if p.strip()]
        if len(parts) > 1:
            return all(safe(p) for p in parts)
    # .map(...).join(...) of markup built here: the templates inside are checked on their own.
    if re.fullmatch(r"[\w.\[\]]+(\.filter\(.*\))?\.map\(.*\)\.join\(.*\)", text, re.S):
        return True
    return any(re.fullmatch(p, text, re.S) for p, _ in SAFE_PATTERNS)


SINK = re.compile(r"\.innerHTML\s*\+?=(?!=)|\binsertAdjacentHTML\(\s*'[a-z]+'\s*,|(?<![\w.])toast\(")


def problems(src):
    """[(line, text)] for every un-escaped value written into an HTML sink in this source."""
    out = []
    for m in SINK.finditer(src):
        if src[m.start() - 9:m.start()] == "function ":   # the definition of toast itself
            continue
        start = m.end()
        stop = ";)\n" if "innerHTML" in m.group() else ",)"
        found = []
        end = scan_code(src, start, found, stop)
        value = src[start:end].strip()
        if not value:
            continue
        for offset, text in found:
            if not safe(text):
                out.append((src.count("\n", 0, offset) + 1, "${" + text + "}"))
        if not found and not safe(value):   # a bare value such as toast(r.error) or innerHTML = text
            out.append((src.count("\n", 0, start) + 1, value))
    return out


def test_app_js_writes_no_unescaped_text_into_html():
    found = problems(APP.read_text())
    assert not found, "Un-escaped text in an HTML sink (wrap it in escH(), or list a truly safe site in " \
        "SAFE with a reason):\n" + "\n".join(f"  app.js:{line}: {text}" for line, text in found)


def test_checker_flags_raw_description_and_accepts_escaped():
    raw = "el.innerHTML = `<div>${s.description}</div>`;\n"
    assert problems(raw) == [(1, "${s.description}")]
    assert problems(raw.replace("${s.description}", "${escH(s.description)}")) == []


def test_checker_covers_every_sink_kind_and_reports_line_numbers():
    src = "\n".join([
        "toast(r.error, 'err');",
        "toast(`${N.msg} (${N.error})`, 'err', 9000);",
        "box.insertAdjacentHTML('beforebegin', `<b>${e.kind}</b>`);",
        "box.innerHTML += `<i>${e.description || ''}</i>`;",
        "x.innerHTML = rows.map((e) => `<p>${e.description}</p>`).join('');",
        "x.innerHTML = text;",
        "toast(ok ? 'Done' : `Failed: ${j.error}`);",
        "toast('Fixed text', 'err'); x.innerHTML = '';",
        "toast(`Adding “${escH(j.neck)}”…`);",
        "x.innerHTML = `<b>${fmtMl(v)}</b>`;",
    ])
    assert problems(src) == [
        (1, "r.error"), (2, "${N.msg}"), (2, "${N.error}"), (3, "${e.kind}"),
        (4, "${e.description || ''}"), (5, "${e.description}"), (6, "text"), (7, "${j.error}"),
    ]


def test_safe_list_has_a_reason_for_every_entry_and_no_dicom_or_error_text():
    source = APP.read_text()
    for key in SAFE:
        assert "${" + key + "}" in source or f"innerHTML = {key};" in source or f"`toast glass ${{{key}}}`" in source, \
            f"stale SAFE entry: {key}"
    for key, reason in list(SAFE.items()) + SAFE_PATTERNS:
        assert reason.strip(), key
        assert key not in {"e.kind", "e.description"}, key
        assert not re.search(r"description|uid|error|\.msg\b|exception|file", key, re.I), key
