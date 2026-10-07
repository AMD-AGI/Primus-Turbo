#!/usr/bin/env python3
"""Minimal Markdown -> self-contained HTML converter for SUMMARY.md.

Handles: ATX headings, GFM pipe tables (with alignment and inline <ul>/<li>
in cells), nested ordered/unordered lists, fenced code blocks, horizontal
rules, paragraphs, and inline code / **bold** / ~~strike~~ / [links](url).
Single '*' is deliberately NOT treated as emphasis (the report uses a
trailing '*' to mark derived TF/s values).

Usage: python3 -I md2html.py IN.md OUT.html
"""
import html
import re
import sys

SEP = re.compile(r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$')
LIST = re.compile(r'^(\s*)([-*+]|\d+[.)])\s+(.*)$')
HEAD = re.compile(r'^(#{1,6})\s+(.*?)\s*#*\s*$')
HR = re.compile(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$')
FENCE = re.compile(r'^\s*```')
ALLOWED_TAGS = re.compile(r'</?(?:ul|ol|li|br|sub|sup)\s*/?>', re.I)


def is_cjk(ch):
    return ('⺀' <= ch <= '鿿') or ('豈' <= ch <= '﫿') or ('＀' <= ch <= '￯') or ('　' <= ch <= '〿')


def join_lines(parts):
    s = ''
    for p in parts:
        if not s:
            s = p
        elif is_cjk(s[-1]) and p and is_cjk(p[0]):
            s += p
        else:
            s += ' ' + p
    return s


def inline(text):
    ph = []

    def stash(s):
        ph.append(s)
        return '\x00%d\x00' % (len(ph) - 1)

    def code_repl(m):
        content = m.group(2)
        if len(content) >= 2 and content[0] == ' ' and content[-1] == ' ' and content.strip():
            content = content[1:-1]
        return stash('<code>' + html.escape(content, quote=False) + '</code>')

    text = re.sub(r'(`+)(.+?)\1', code_repl, text)
    text = ALLOWED_TAGS.sub(lambda m: stash(m.group(0).lower()), text)
    text = html.escape(text, quote=False)
    text = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)',
                  lambda m: '<a href="%s">%s</a>' % (html.escape(m.group(2), quote=True), m.group(1)), text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'~~(.+?)~~', r'<del>\1</del>', text)
    # placeholders may be nested inside each other's output only via stash order; restore repeatedly
    for _ in range(3):
        if '\x00' not in text:
            break
        text = re.sub(r'\x00(\d+)\x00', lambda m: ph[int(m.group(1))], text)
    return text


def split_row(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|') and not s.endswith('\\|'):
        s = s[:-1]
    return [c.strip().replace('\\|', '|') for c in re.split(r'(?<!\\)\|', s)]


def is_block_start(lines, i):
    line = lines[i]
    if not line.strip():
        return True
    if FENCE.match(line) or HEAD.match(line) or HR.match(line) or LIST.match(line):
        return True
    if '|' in line and i + 1 < len(lines) and SEP.match(lines[i + 1]):
        return True
    return False


def render_table(header, aligns, rows):
    ncol = len(header)

    def al(k):
        a = aligns[k] if k < len(aligns) else ''
        return ' class="%s"' % a if a else ''

    minw = min(ncol * 6.5, 100)
    out = ['<div class="tw"><table style="min-width:%.1fem">' % minw, '<thead><tr>']
    out += ['<th%s>%s</th>' % (al(k), inline(h)) for k, h in enumerate(header)]
    out.append('</tr></thead><tbody>')
    for row in rows:
        cells = (row + [''] * ncol)[:ncol]
        if ncol > 1 and re.fullmatch(r'\*\*.+\*\*', cells[0]) and not any(cells[1:]):
            out.append('<tr class="grp"><td colspan="%d">%s</td></tr>' % (ncol, inline(cells[0])))
        else:
            out.append('<tr>' + ''.join('<td%s>%s</td>' % (al(k), inline(c)) for k, c in enumerate(cells)) + '</tr>')
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def parse_list(lines, i):
    items = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            k = i + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and LIST.match(lines[k]):
                i = k
                continue
            break
        m = LIST.match(line)
        if m:
            indent = len(m.group(1).expandtabs(4))
            marker = m.group(2)
            ordered = marker[0].isdigit()
            num = int(marker[:-1]) if ordered else None
            items.append([indent, ordered, num, m.group(3).strip()])
            i += 1
        elif items and line[:1] == ' ' and not FENCE.match(line) and not ('|' in line and line.strip().startswith('|')):
            items[-1][3] = join_lines([items[-1][3], line.strip()])
            i += 1
        else:
            break
    return items, i


def render_list(items):
    out = []
    stack = []  # (indent, tag)

    def open_list(indent, tag, num):
        start = ' start="%d"' % num if tag == 'ol' and num not in (None, 1) else ''
        out.append('<%s%s>' % (tag, start))
        stack.append((indent, tag))

    for indent, ordered, num, text in items:
        tag = 'ol' if ordered else 'ul'
        while stack and indent < stack[-1][0]:
            out.append('</li></%s>' % stack[-1][1])
            stack.pop()
        if stack and indent == stack[-1][0]:
            if stack[-1][1] != tag:
                out.append('</li></%s>' % stack[-1][1])
                stack.pop()
                open_list(indent, tag, num)
            else:
                out.append('</li>')
        else:
            open_list(indent, tag, num)
        out.append('<li>' + inline(text))
    while stack:
        out.append('</li></%s>' % stack[-1][1])
        stack.pop()
    return '\n'.join(out)


def convert(md):
    lines = md.split('\n')
    out = []
    toc = []
    hid = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if FENCE.match(line):
            lang = s[3:].strip()
            j = i + 1
            buf = []
            while j < len(lines) and not FENCE.match(lines[j]):
                buf.append(lines[j])
                j += 1
            cls = ' class="language-%s"' % html.escape(lang) if lang else ''
            out.append('<pre><code%s>%s</code></pre>' % (cls, html.escape('\n'.join(buf), quote=False)))
            i = j + 1
            continue
        m = HEAD.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            hid += 1
            anchor = 's%d' % hid
            out.append('<h%d id="%s">%s</h%d>' % (level, anchor, inline(text), level))
            toc.append((level, anchor, text))
            i += 1
            continue
        if HR.match(line):
            out.append('<hr>')
            i += 1
            continue
        if '|' in line and i + 1 < len(lines) and SEP.match(lines[i + 1]):
            header = split_row(line)
            aligns = []
            for c in split_row(lines[i + 1]):
                c = c.strip()
                if c.startswith(':') and c.endswith(':'):
                    aligns.append('c')
                elif c.endswith(':'):
                    aligns.append('r')
                else:
                    aligns.append('')
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith('|'):
                rows.append(split_row(lines[j]))
                j += 1
            out.append(render_table(header, aligns, rows))
            i = j
            continue
        if LIST.match(line):
            items, j = parse_list(lines, i)
            out.append(render_list(items))
            i = j
            continue
        buf = [s]
        j = i + 1
        while j < len(lines) and not is_block_start(lines, j):
            buf.append(lines[j].strip())
            j += 1
        out.append('<p>%s</p>' % inline(join_lines(buf)))
        i = j
    return '\n'.join(out), toc


def strip_tags(s):
    return re.sub(r'<[^>]+>', '', s)


def build_toc(toc):
    out = ['<nav class="toc" aria-label="目录"><details open><summary>目录</summary>']
    cur = None
    for level, anchor, text in toc:
        if level not in (2, 3):
            continue
        label = strip_tags(inline(text))  # inline() already escapes text
        if level == 2:
            if cur == 3:
                out.append('</ul></li>')
            elif cur == 2:
                out.append('</li>')
            if cur is None:
                out.append('<ul>')
            out.append('<li><a href="#%s">%s</a>' % (anchor, label))
            cur = 2
        else:
            if cur == 2:
                out.append('<ul>')
            elif cur is None:
                out.append('<ul><li><ul>')
            out.append('<li><a href="#%s">%s</a></li>' % (anchor, label))
            cur = 3
    if cur == 3:
        out.append('</ul></li></ul>')
    elif cur == 2:
        out.append('</li></ul>')
    out.append('</details></nav>')
    return '\n'.join(out)


CSS = """
:root {
  color-scheme: light;
  --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d0d7de;
  --th-bg: #eaeef2; --zebra: #f6f8fa; --grp-bg: #ddf4ff; --grp-fg: #0a3069;
  --code-bg: #eff1f3; --pre-bg: #f6f8fa; --accent: #0969da; --del: #8c959f;
  --toc-bg: #f6f8fa; --hover: #fff8c5;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d;
    --th-bg: #1f2630; --zebra: #151b23; --grp-bg: #102a46; --grp-fg: #a5d6ff;
    --code-bg: #262c36; --pre-bg: #151b23; --accent: #4493f8; --del: #6e7681;
    --toc-bg: #151b23; --hover: #2d2a12;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d;
  --th-bg: #1f2630; --zebra: #151b23; --grp-bg: #102a46; --grp-fg: #a5d6ff;
  --code-bg: #262c36; --pre-bg: #151b23; --accent: #4493f8; --del: #6e7681;
  --toc-bg: #151b23; --hover: #2d2a12;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB",
        "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
}
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 72px; overflow-wrap: anywhere; }
h1 { font-size: 1.65em; line-height: 1.35; margin: 0.3em 0 0.7em; }
h2 { font-size: 1.38em; line-height: 1.4; margin: 2.4em 0 0.7em; padding-bottom: 0.3em; border-bottom: 1px solid var(--border); }
h3 { font-size: 1.16em; margin: 1.9em 0 0.5em; }
h4 { font-size: 1.03em; margin: 1.5em 0 0.45em; }
h2, h3, h4 { scroll-margin-top: 12px; }
p { margin: 0.6em 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
strong { font-weight: 650; }
code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.86em; background: var(--code-bg); padding: 0.08em 0.35em; border-radius: 4px;
}
pre { background: var(--pre-bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;
      overflow-x: auto; font-size: 13px; line-height: 1.55; margin: 0.8em 0 1.2em; }
pre code { background: none; padding: 0; font-size: inherit; white-space: pre; overflow-wrap: normal; }
del { color: var(--del); }
hr { border: 0; border-top: 1px solid var(--border); margin: 2.2em 0; }
ul, ol { padding-left: 1.6em; margin: 0.5em 0; }
li { margin: 0.22em 0; }
li > ul, li > ol { margin: 0.2em 0; }
.tw { overflow: auto; max-height: 82vh; margin: 0.8em 0 1.5em; border: 1px solid var(--border);
      border-radius: 6px; -webkit-overflow-scrolling: touch; overscroll-behavior-x: contain; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; line-height: 1.55; }
th, td { padding: 6px 9px; text-align: left; vertical-align: top;
         border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); }
th:last-child, td:last-child { border-right: 0; }
tbody tr:last-child td { border-bottom: 0; }
thead th { position: sticky; top: 0; z-index: 2; background: var(--th-bg); font-weight: 650;
           box-shadow: 0 1px 0 var(--border); }
tbody tr:nth-child(even) td { background: var(--zebra); }
tbody tr:hover td { background: var(--hover); }
tr.grp td { background: var(--grp-bg); color: var(--grp-fg); font-weight: 650; }
td ul, td ol { margin: 0; padding-left: 1.15em; }
td li { margin: 0.1em 0; }
th.r, td.r { text-align: right; font-variant-numeric: tabular-nums; }
th.c, td.c { text-align: center; }
nav.toc { background: var(--toc-bg); border: 1px solid var(--border); border-radius: 6px;
          padding: 8px 16px; margin: 1em 0 1.6em; font-size: 0.94em; }
nav.toc summary { cursor: pointer; font-weight: 650; }
nav.toc ul { margin: 0.35em 0; padding-left: 1.3em; }
nav.toc li { margin: 0.12em 0; }
nav.toc ul ul { font-size: 0.95em; }
footer { color: var(--muted); font-size: 0.85em; margin-top: 3em; border-top: 1px solid var(--border); padding-top: 1em; }
@media (max-width: 640px) {
  body { font-size: 14.5px; }
  h1 { font-size: 1.4em; }
  table { font-size: 12.5px; }
}
"""


def main():
    src, dst = sys.argv[1], sys.argv[2]
    md = open(src, encoding='utf-8').read()
    body, toc = convert(md)
    title = 'Llama-3.1-8B Attention 优化进展'
    toc_html = build_toc(toc)
    idx = body.find('</h1>')
    if idx >= 0:
        body = body[:idx + 5] + '\n' + toc_html + body[idx + 5:]
    else:
        body = toc_html + body
    doc = ('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           '<meta name="color-scheme" content="light dark">\n'
           '<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n<main>\n%s\n'
           '<footer>由 <code>SUMMARY.md</code> 生成（output/1007__summary）。本报告只整理已有记录，未运行任何 GPU 任务。</footer>\n'
           '</main>\n</body>\n</html>\n') % (title, CSS, body)
    open(dst, 'w', encoding='utf-8').write(doc)


if __name__ == '__main__':
    main()
