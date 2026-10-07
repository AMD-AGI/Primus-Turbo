#!/usr/bin/env python3
"""Compare a Chinese markdown chunk with its English translation (CPU only).

Checks: numeric tokens (multiset), CJK left in the English text, headings per
level, markdown tables (count, rows per table, cells per row).
Usage: python3 -I check_i18n.py ZH.md EN.md
Exit 0 when everything matches, 1 otherwise.
"""
import collections
import re
import sys

NUM = re.compile(r'\d+(?:\.\d+)?')
CJK = re.compile(r'[㐀-鿿　-〿！-～]')
PIPE = re.compile(r'(?<!\\)\|')


def tables(lines):
    out, cur, fence = [], None, False
    for l in lines:
        if l.strip().startswith('```'):
            fence = not fence
        if not fence and l.startswith('|'):
            cells = len(PIPE.findall(l)) - 1
            if cur is None:
                cur = []
                out.append(cur)
            cur.append(cells)
        else:
            cur = None
    return out


def main():
    zh = open(sys.argv[1], encoding='utf-8').read()
    en = open(sys.argv[2], encoding='utf-8').read()
    ok = True
    a, b = collections.Counter(NUM.findall(zh)), collections.Counter(NUM.findall(en))
    miss, extra = a - b, b - a
    if miss or extra:
        ok = False
        print('NUMBERS missing in EN:', dict(miss))
        print('NUMBERS extra in EN  :', dict(extra))
    left = [(i + 1, l) for i, l in enumerate(en.split('\n')) if CJK.search(l)]
    if left:
        ok = False
        print(f'CJK left in EN: {len(left)} lines')
        for i, l in left[:15]:
            m = CJK.search(l)
            print(f'  L{i}: ...{l[max(0, m.start() - 40):m.start() + 40]}...')
    hz = collections.Counter(re.findall(r'(?m)^(#{1,6}) ', zh))
    he = collections.Counter(re.findall(r'(?m)^(#{1,6}) ', en))
    if hz != he:
        ok = False
        print('HEADINGS differ:', dict(hz), 'vs', dict(he))
    tz, te = tables(zh.split('\n')), tables(en.split('\n'))
    if [len(t) for t in tz] != [len(t) for t in te]:
        ok = False
        print('TABLE rows differ:', [len(t) for t in tz], 'vs', [len(t) for t in te])
    for k, t in enumerate(te):
        if len(set(t)) > 1:
            ok = False
            print(f'TABLE {k} in EN has rows with different cell counts: {sorted(set(t))}')
    for k, (x, y) in enumerate(zip(tz, te)):
        if x and y and x[0] != y[0]:
            ok = False
            print(f'TABLE {k}: {x[0]} cells in ZH vs {y[0]} in EN')
    print('OK' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
