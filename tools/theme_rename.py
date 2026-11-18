"""Move the window's code from the old fixed colours to the look in use (light or dark).

    python tools/theme_rename.py [files...]      (no files: every .rs under app/src/ui and app/src/app.rs)

The colours were constants (theme::BRIGHT, DIM, ...), which cannot follow a look chosen while the program
runs. They are now fields of the palette in use, `theme::pal().bright` and so on. This rewrites every use
of an old name into the new form, keeping each file's line endings. It is safe to run again: what it has
rewritten no longer matches. The old names still compile (as the dark look's colours, with a deprecation
warning), so code written against them elsewhere keeps building until this is run over it.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAMES = ['TEXT', 'DIM', 'BRIGHT', 'FAINT', 'HINT', 'BUTTON', 'INK', 'ACCENT_MINI', 'ACCENT', 'HANDLE_TO', 'ERROR_TEXT', 'ERROR_FILL', 'MENU_FILL', 'PAST', 'COSTLY']
PATTERN = re.compile(r'(?<![\w.:])(theme::)?(' + '|'.join(NAMES) + r')\b(?!\s*[:=]\s*Color32)')


def rewrite(text):
    out, n, i = [], 0, 0
    # leave string literals and comments alone
    token = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|' + PATTERN.pattern)
    for m in token.finditer(text):
        s = m.group(0)
        if s.startswith('"') or s.startswith('//'):
            continue
        out.append(text[i:m.start()])
        prefix = 'theme::' if s.startswith('theme::') else ''
        name = s[len(prefix):]
        out.append(prefix + 'pal().' + name.lower())
        i = m.end()
        n += 1
    out.append(text[i:])
    return ''.join(out), n


def main():
    files = sys.argv[1:]
    if not files:
        ui = os.path.join(ROOT, 'app', 'src', 'ui')
        files = [os.path.join(ui, f) for f in sorted(os.listdir(ui)) if f.endswith('.rs') and f != 'theme.rs']
        files.append(os.path.join(ROOT, 'app', 'src', 'app.rs'))
    total = 0
    for f in files:
        raw = open(f, 'rb').read().decode('utf-8')
        new, n = rewrite(raw)
        if n:
            open(f, 'wb').write(new.encode('utf-8'))
            print(f'{os.path.relpath(f, ROOT)}: {n}')
            total += n
    print(f'{total} uses rewritten')


if __name__ == '__main__':
    main()
