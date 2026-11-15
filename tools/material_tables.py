"""Writes core/src/material_data.rs from the gathered optical data (docs/materials_optical_data.json).

usage: python tools/material_tables.py <optical_data.json> <material_data.rs>

Run again only when the data file changes; the Rust file it writes is checked in. Every table is on the
same grid: 400, 410, ... 700 nm (31 values).
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src, encoding='utf-8'))

# (id, name shown, key in the data file)
METALS = [
    ('silver', 'silver', 'silver'),
    ('aluminium', 'aluminium', 'aluminium'),
    ('gold', 'gold', 'gold'),
    ('copper', 'copper', 'copper'),
    ('chromium', 'chrome', 'chromium'),
    ('nickel', 'nickel', 'nickel'),
    ('platinum', 'platinum', 'platinum'),
    ('titanium', 'titanium', 'titanium'),
    ('iron', 'iron', 'iron'),
    ('tungsten', 'tungsten', 'tungsten'),
]

# (id, name shown, key in the data file, which array, what kind of number it is)
SPECTRA = [
    ('ruby', 'ruby', 'ruby', 'alpha'),
    ('emerald', 'emerald', 'emerald', 'alpha'),
    ('sapphire', 'blue sapphire', 'blue_sapphire', 'alpha'),
    ('amethyst', 'amethyst', 'amethyst', 'alpha'),
    ('citrine', 'citrine', 'citrine', 'alpha'),
    ('smoky', 'smoky quartz', 'smoky_quartz', 'alpha'),
    ('topaz', 'brown topaz', 'brown_topaz', 'alpha'),
    ('cobalt', 'cobalt blue glass', 'cobalt_blue_glass', 'alpha'),
    ('didymium', 'didymium glass', 'didymium_glass', 'alpha'),
    ('aqua', 'copper aqua glass', 'copper_blue_green_glass', 'alpha'),
    ('signal', 'selenium ruby glass', 'selenium_ruby_glass', 'alpha'),
    ('yellow', 'cadmium yellow glass', 'cadmium_yellow_glass', 'alpha'),
    ('amber', 'amber bottle glass', 'amber_glass', 'alpha'),
    ('manganese', 'manganese purple glass', 'manganese_purple_glass', 'alpha'),
    ('chromegreen', 'chromium green glass', 'chromium_green_glass', 'alpha'),
    ('cranberry', 'gold ruby (cranberry) glass', 'gold_ruby_glass', 'alpha'),
    ('uranium', 'uranium glass', 'uranium_glass', 'alpha'),
    ('bronze', 'bronze glass', 'bronze_float_glass', 'alpha'),
    ('grey', 'grey glass', 'grey_float_glass', 'alpha'),
    ('water', 'water', 'water', 'alpha'),
]

KIND = {'tabulated': 'Measured', 'digitised': 'Digitised', 'approximate': 'Approximate'}


def rust_str(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def arr(v, fmt='%.5g'):
    return '[' + ', '.join((fmt % x) if '.' in (fmt % x) or 'e' in (fmt % x) else (fmt % x) + '.0' for x in v) + ']'


out = []
out.append('// Written by tools/material_tables.py from the gathered optical data: do not edit by hand.')
out.append('// Every table holds 31 values, at 400, 410, ... 700 nm.')
out.append('')
out.append('use super::{Metal, Spectrum, SpectrumKind};')
out.append('')
out.append('pub static METALS: &[Metal] = &[')
for mid, name, key in METALS:
    m = d['metals'][key]
    assert m['nm'] == list(range(400, 701, 10)) and len(m['n']) == 31 and len(m['k']) == 31
    source = '%s; refractiveindex.info database, %s' % (m.get('dataset', m['source']), m.get('file', '').split(' ')[0])
    out.append('    Metal {')
    out.append('        id: %s, name: %s,' % (rust_str(mid), rust_str(name)))
    out.append('        source: %s,' % rust_str(source))
    out.append('        n: %s,' % arr(m['n']))
    out.append('        k: %s,' % arr(m['k']))
    out.append('    },')
out.append('];')
out.append('')
out.append('pub static SPECTRA: &[Spectrum] = &[')
for sid, name, key, which in SPECTRA:
    s = d['absorbers'][key]
    a = s[which]
    assert s['nm'] == list(range(400, 701, 10)) and len(a) == 31
    peak = max(a)
    shape = [max(x, 0.0) / peak for x in a]
    units = s.get('units', '')
    per_cm = peak if units.startswith('cm^-1') and 'per' not in units else 0.0
    source = s['source']
    if len(source) > 240:
        source = source[:237].rsplit(' ', 1)[0] + ' ...'
    out.append('    Spectrum {')
    out.append('        id: %s, name: %s, kind: SpectrumKind::%s,' % (rust_str(sid), rust_str(name), KIND[s['kind']]))
    out.append('        source: %s,' % rust_str(source))
    out.append('        peak_per_cm: %.5g,' % per_cm if '.' in ('%.5g' % per_cm) else '        peak_per_cm: %.5g.0,' % per_cm)
    out.append('        shape: %s,' % arr(shape, '%.4f'))
    out.append('    },')
out.append('];')
out.append('')
open(dst, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
print('wrote', dst, ':', len(METALS), 'metals,', len(SPECTRA), 'spectra')
