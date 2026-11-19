"""Compile observed linear devices and run the installed ngspice binary.

This compiler shares literal/unit validation with the exact calculator, but
never reads its solved state or encodes a calculated answer as a source.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess
import time

import sympy as sp

from .io import save
from .linear import Circuit, CalculationError, expression, identifier, quantity
from .reasoning import normalize_plan, project

NGSPICE = Path('/opt/homebrew/bin/ngspice')


def _number(value):
    if value.free_symbols or value.is_real is not True:
        raise CalculationError('SPICE requires a finite real numeric device coefficient')
    result = float(sp.N(value, 17))
    if not math.isfinite(result):
        raise CalculationError('SPICE device coefficient is nonfinite')
    return format(result, '.17g')


def _complex(value, unit):
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise CalculationError('ngspice returned a nonfinite quantity')
    return {'real': result.real, 'imaginary': result.imag, 'unit': unit}


def _phasor(value):
    value = complex(sp.N(value, 17))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise CalculationError('Nonfinite excitation phasor')
    return format(abs(value), '.17g') + ' ' + format(math.degrees(math.atan2(value.imag, value.real)), '.17g')


class _Compiler:
    def __init__(self, circuit, question):
        self.facts = project(circuit)
        if self.facts['electrical_issues']:
            raise CalculationError('Unresolved electrical interpretation prevents SPICE compilation')
        if 'analysis_context' not in circuit:
            raise CalculationError('Reader did not provide analysis conditions')
        self.context = normalize_plan(circuit['analysis_context'], question)
        # Constructor validates givens, dimensions and explicit ideal assumptions;
        # Circuit.system/prepare and all SymPy solving routines are never called.
        self.circuit = Circuit(self.facts, self.context, question)
        self.mode = self.circuit.mode
        if self.mode not in ('dc', 'ac'):
            raise CalculationError('SPICE integration supports DC or single-frequency AC only')
        self.lines = ['Observed circuit compiled by K-Bot; ngspice independent numerical run']
        self.mapping = {'nodes': {}, 'component_probes': {}, 'gauges': [], 'ideal_opamps': {}}
        self.symbols = {}; self.symbol_text = {}; self.orientations = {}; self.references = {}
        self._nodes()
        self._branches()
        self._references()
        omega = self.circuit.omega
        if self.mode == 'ac' and omega is None:
            for c in self.circuit.parts.values():
                if c['type'] in ('capacitor', 'inductor') and self.circuit.value(c)[1] != 'ohm':
                    raise CalculationError('Physical L/C values require an explicit AC frequency')
            omega = sp.S.One
            basis = 'normalized_single_frequency_impedances'
        else:
            basis = 'physical_question_or_source' if self.mode == 'ac' else None
        self.omega = omega
        self.analysis = {'mode': self.mode, 'frequency_hz': float(sp.N(omega/(2*sp.pi))) if omega is not None else None,
            'angular_frequency_rad_s': float(sp.N(omega)) if omega is not None else None,
            'frequency_basis': basis, 'phasor_convention': self.circuit.convention}
        if basis == 'normalized_single_frequency_impedances':
            self.analysis['frequency_note'] = 'Computational omega=1 rad/s represents supplied complex impedances; it is not a claimed physical circuit frequency.'

    def _nodes(self):
        parents = {n: n for n in self.circuit.nodes}
        def find(n):
            while parents[n] != n:
                n = parents[n]
            return n
        def join(nodes):
            for n in nodes[1:]:
                parents[find(n)] = find(nodes[0])
        for c in self.circuit.parts.values():
            join(list(c['terminals'].values()))
        for _, p, n, _ in self.circuit.drives:
            join([p, n])
        islands = {}
        for n in self.circuit.nodes:
            islands.setdefault(find(n), []).append(n)
        references = set()
        for nodes in islands.values():
            grounds = sorted(set(nodes) & set(self.facts['ground_nets']))
            if len(grounds) > 1:
                raise CalculationError('Distinct ground domains lack an explicit electrical reference relationship')
            negatives = [c['terminals']['negative'] for c in self.circuit.parts.values()
                if c['type'] == 'voltage_source' and c['terminals'].get('negative') in nodes]
            reference = grounds[0] if grounds else min(negatives or nodes)
            references.add(reference)
            self.mapping['gauges'].append({'net_id': reference, 'kind': 'physical_ground' if grounds else 'mathematical_gauge',
                'island_net_ids': sorted(nodes), 'note': 'SPICE node zero is a voltage reference; no new observed wire or ground symbol is asserted.'})
            self.lines.append('* Reference ' + reference + ' -> 0: ' + ('observed physical ground' if grounds else 'mathematical gauge only'))
        for i, net in enumerate(self.circuit.nodes, 1):
            self.mapping['nodes'][net] = '0' if net in references else 'n' + str(i)
            symbol = sp.Symbol('nv' + str(i), real=True)
            self.symbols['V:' + net] = symbol
            self.symbol_text[symbol] = 'v(' + self.mapping['nodes'][net] + ')'

    def _branches(self):
        for i, (cid, c) in enumerate(self.circuit.parts.items(), 1):
            terminals = c['terminals']; kind = c['type']
            if kind == 'op_amp':
                if cid not in self.circuit.ideal or set(terminals) != {'inverting', 'noninverting', 'output'}:
                    raise CalculationError('SPICE requires an explicitly ideal, linear three-terminal op-amp')
                keys = ['output']
            else:
                if kind not in {'resistor', 'impedance_block', 'capacitor', 'inductor', 'voltage_source', 'current_source', 'dependent_voltage_source', 'dependent_current_source'}:
                    raise CalculationError('Unsupported SPICE device model: ' + kind)
                if len(terminals) != 2:
                    raise CalculationError('SPICE two-terminal device has wrong arity')
                keys = ['positive', 'negative'] if 'positive' in terminals else ['from', 'to'] if 'from' in terminals else list(terminals)
                if any(k not in terminals for k in keys):
                    raise CalculationError('SPICE source polarity is not bound to owned terminals')
            token = 'd' + str(i); probe = 'vprobe_' + token
            self.orientations[cid] = keys
            self.mapping['component_probes'][cid] = {'probe': probe, 'primary_terminal': keys[0],
                'secondary_terminal': keys[1] if len(keys) > 1 else None,
                'ngspice_current_variable': 'i(' + probe + ')', 'element_token': token}
            symbol = sp.Symbol('bi' + str(i), real=True)
            self.symbols['I:' + cid] = symbol
            self.symbol_text[symbol] = 'i(' + probe + ')'

    def V(self, net):
        try:
            return self.symbols['V:' + net]
        except KeyError as error:
            raise CalculationError('Unknown observed voltage net in a control') from error

    def I(self, cid, terminal=None):
        if cid not in self.orientations:
            raise CalculationError('Unknown observed component in a current control')
        keys = self.orientations[cid]
        if terminal is None or terminal == keys[0]:
            return self.symbols['I:' + cid]
        if len(keys) == 2 and terminal == keys[1]:
            return -self.symbols['I:' + cid]
        if self.circuit.parts[cid]['type'] == 'op_amp' and terminal in ('inverting', 'noninverting'):
            return sp.S.Zero
        raise CalculationError('Current control uses an unowned terminal')

    def _references(self):
        def register(label, value):
            key = identifier(label)
            if key in self.references and sp.simplify(self.references[key]-value) != 0:
                raise CalculationError('Conflicting physical control reference')
            self.references[key] = value
        for row in self.facts['current_references']:
            if 'expression' not in row:
                raise CalculationError('Unresolved observed current reference')
            register(row['label'], expression(row['expression'], {}, self.V, self.I))
        for row in self.facts['voltage_references']:
            if row['positive_net_id'] is not None and row['negative_net_id'] is not None:
                register(row['label'], self.V(row['positive_net_id'])-self.V(row['negative_net_id']))
        for cid, c in self.circuit.parts.items():
            if c['type'] not in ('voltage_source', 'current_source'):
                continue
            aliases = list(c.get('names', []))
            if re.fullmatch(r'[A-Za-zΑ-ω][A-Za-z0-9_{}Α-ω]*', str(c.get('value', ''))):
                aliases.append(c['value'])
            for alias in aliases:
                register(alias, self.V(c['terminals']['positive'])-self.V(c['terminals']['negative']) if c['type'] == 'voltage_source' else self.I(cid))

    def _source(self, letter, name, a, b, value):
        if self.mode == 'dc':
            return f'{letter}{name} {a} {b} DC {_number(value)}'
        return f'{letter}{name} {a} {b} DC 0 AC {_phasor(value)}'

    def _linear_control(self, value):
        symbols = sorted(value.free_symbols, key=str)
        if not set(symbols) <= set(self.symbol_text):
            raise CalculationError('Unbound symbol in SPICE control expression')
        constant = sp.simplify(value.subs({s: 0 for s in symbols}))
        terms = []; remainder = value-constant
        for symbol in symbols:
            coefficient = sp.simplify(sp.diff(value, symbol))
            if coefficient.free_symbols:
                raise CalculationError('SPICE integration supports linear controls only')
            terms.append('(' + _number(coefficient) + ')*' + self.symbol_text[symbol])
            remainder -= coefficient*symbol
        if sp.simplify(remainder) != 0:
            raise CalculationError('SPICE integration supports affine controls only')
        return '+'.join(terms) or '0', constant

    def compile(self):
        for cid, c in self.circuit.parts.items():
            info = self.mapping['component_probes'][cid]; token = info['element_token']
            keys = self.orientations[cid]; t = c['terminals']; a = self.mapping['nodes'][t[keys[0]]]
            x = 'x_' + token; b = self.mapping['nodes'][t[keys[1]]] if len(keys) == 2 else '0'
            self.lines += ['* ' + cid + ' ' + c['type'], f"{info['probe']} {a} {x} 0"]
            kind = c['type']
            if kind == 'op_amp':
                positive = self.mapping['nodes'][t['noninverting']]; negative = self.mapping['nodes'][t['inverting']]
                # The own-output terms cancel algebraically: V+ = V-, with
                # unconstrained output current. No finite-gain approximation.
                self.lines.append(f'B{token} {x} 0 V=v({x})+v({positive})-v({negative})')
                self.mapping['ideal_opamps'][cid] = {'model': 'algebraic_nullor', 'constraint': 'V(noninverting)=V(inverting)',
                    'input_currents': 'zero', 'finite_gain_assumption': False}
                continue
            if kind.startswith('dependent_'):
                value, dim = quantity(c.get('value') or '', c.get('unit'), {**self.circuit.constants, **self.references})
                expected = 'V' if kind == 'dependent_voltage_source' else 'A'
                if dim is not None and dim != expected:
                    raise CalculationError('Dependent-source output unit is incompatible')
                control, constant = self._linear_control(value)
                if kind == 'dependent_voltage_source':
                    middle = 'o_' + token
                    self.lines += [f'B{token} {x} {middle} V={control}', self._source('V', 'offset_' + token, middle, b, constant)]
                else:
                    self.lines += [f'B{token} {x} {b} I={control}', self._source('I', 'offset_' + token, x, b, constant)]
                continue
            value, dim = self.circuit.value(c)
            if kind in ('voltage_source', 'current_source'):
                if dim is not None and dim != ('V' if kind == 'voltage_source' else 'A'):
                    raise CalculationError('Independent-source unit is incompatible')
                if self.mode == 'ac' and self.circuit.source_waves and cid not in self.circuit.source_waves:
                    if self.circuit.input_convention == 'unspecified':
                        raise CalculationError('Mixed waveform/phasor sources require an explicit convention')
                    if self.circuit.input_convention == 'rms':
                        value *= sp.sqrt(2)
                self.lines.append(self._source('V' if kind == 'voltage_source' else 'I', token, x, b, value))
                continue
            if kind in ('resistor', 'impedance_block') and dim == 'S':
                if kind == 'resistor' and sp.im(value) != 0:
                    raise CalculationError('A resistor cannot have complex conductance')
                if value == 0:
                    self.lines.append(f'Iopen_{token} {x} {b} DC 0 AC 0')
                    continue
                value, dim = 1/value, 'ohm'
            expected = {'resistor': 'ohm', 'impedance_block': 'ohm', 'capacitor': 'F', 'inductor': 'H'}[kind]
            if dim is not None and dim not in (expected, 'ohm'):
                raise CalculationError('Incompatible passive-device unit')
            if kind in ('resistor', 'impedance_block') or dim == 'ohm':
                if kind == 'resistor' and sp.im(value) != 0:
                    raise CalculationError('A resistor cannot have complex resistance')
                real, imaginary = sp.simplify(sp.re(value)), sp.simplify(sp.im(value))
                if self.mode == 'dc' and imaginary != 0:
                    raise CalculationError('Complex impedance requires AC analysis')
                middle = 'z_' + token if real != 0 and imaginary != 0 else b
                if real != 0:
                    self.lines.append(f'R{token} {x} {middle} {_number(real)}')
                if imaginary != 0:
                    start = middle if real != 0 else x
                    if imaginary > 0:
                        self.lines.append(f'L{token} {start} {b} {_number(imaginary/self.omega)}')
                    elif imaginary < 0:
                        self.lines.append(f'C{token} {start} {b} {_number(-1/(self.omega*imaginary))}')
                    else:
                        raise CalculationError('Complex impedance reactance is not identifiable')
                if real == 0 and imaginary == 0:
                    self.lines.append(f'Vshort_{token} {x} {b} 0')
            else:
                if value.is_real is not True or value <= 0:
                    raise CalculationError('Physical L/C values must be positive real quantities')
                if self.mode == 'dc':
                    self.lines.append(f'Iopen_{token} {x} {b} DC 0' if kind == 'capacitor' else f'Vshort_{token} {x} {b} 0')
                else:
                    self.lines.append(f"{'C' if kind == 'capacitor' else 'L'}{token} {x} {b} {_number(value)}")
        for cid, p, n, value in self.circuit.drives:
            if self.mode == 'ac' and self.circuit.source_waves:
                if self.circuit.input_convention == 'unspecified':
                    raise CalculationError('Boundary drives require an explicit amplitude convention')
                if self.circuit.input_convention == 'rms':
                    value *= sp.sqrt(2)
            self.lines.append(self._source('V', cid, self.mapping['nodes'][p], self.mapping['nodes'][n], value))
        command = 'op' if self.mode == 'dc' else 'ac lin 1 ' + format(self.analysis['frequency_hz'], '.17g') + ' ' + format(self.analysis['frequency_hz'], '.17g')
        self.lines += ['.options numdgt=15 gmin=0 gminsteps=0 srcsteps=0', '.control', 'set noaskquit', 'set filetype=ascii', command,
            'write spice.raw all', 'quit', '.endc', '.end']
        return '\n'.join(self.lines) + '\n'

    def results(self, raw):
        node_values = {net: 0j if name == '0' else raw['v(' + name + ')'] for net, name in self.mapping['nodes'].items()}
        branch_values = {cid: raw[info['ngspice_current_variable']] for cid, info in self.mapping['component_probes'].items()}
        terminals = {}
        for cid, c in self.circuit.parts.items():
            keys = self.orientations[cid]
            for key in c['terminals']:
                terminals[cid + '.' + key] = _complex(branch_values[cid] if key == keys[0] else -branch_values[cid] if len(keys) == 2 else 0j, 'A')
        substitutions = {self.symbols['V:' + n]: v for n, v in node_values.items()}
        substitutions.update({self.symbols['I:' + cid]: v for cid, v in branch_values.items()})
        currents = {r['label']: _complex(complex(sp.N(self.references[identifier(r['label'])].subs(substitutions))), 'A') for r in self.facts['current_references']}
        voltages = {r['label']: _complex(node_values[r['positive_net_id']]-node_values[r['negative_net_id']], 'V')
            for r in self.facts['voltage_references'] if r['positive_net_id'] is not None and r['negative_net_id'] is not None}
        return {'net_voltages': {n: _complex(v, 'V') for n, v in node_values.items()}, 'terminal_currents': terminals,
            'current_references': currents, 'voltage_references': voltages}


def _raw_values(path):
    text = path.read_text(); header, values = text.split('Values:\n', 1)
    count = re.search(r'No\. Points:\s*(\d+)', header)
    if not count or int(count.group(1)) != 1:
        raise CalculationError('ngspice did not produce exactly one operating point/frequency')
    variables = header.split('Variables:\n', 1)[1].strip().splitlines()
    names = [line.split()[1].lower() for line in variables]
    lines = [line.strip() for line in values.splitlines() if line.strip()]
    if len(lines) != len(names):
        raise CalculationError('ngspice raw result has incomplete variables')
    output = {}
    for index, (name, line) in enumerate(zip(names, lines)):
        scalar = line.split(None, 1)[1] if index == 0 else line
        pieces = scalar.split(',')
        value = complex(float(pieces[0]), float(pieces[1]) if len(pieces) == 2 else 0)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise CalculationError('ngspice result contains nonfinite values')
        output[name] = value
    return output


def simulate(circuit, question, output):
    """Run real local SPICE and save a separate numerical record, never an answer."""
    started = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    record = {'status': 'unsupported', 'engine': {'name': 'ngspice', 'executable': str(NGSPICE), 'version': None},
        'netlist_path': str(output/'circuit.spice'), 'log_path': str(output/'spice.log'),
        'net_voltages': {}, 'terminal_currents': {}, 'current_references': {}, 'voltage_references': {}}
    compiler = None
    try:
        compiler = _Compiler(circuit, question)
        netlist = compiler.compile(); (output/'circuit.spice').write_text(netlist)
        record.update(analysis=compiler.analysis, mapping=compiler.mapping)
        if not NGSPICE.is_file():
            raise RuntimeError('Installed ngspice executable is unavailable')
        version = subprocess.run([str(NGSPICE), '--version'], capture_output=True, text=True, timeout=5, check=True)
        match = re.search(r'ngspice-([^\s:]+)', version.stdout + version.stderr)
        record['engine']['version'] = match.group(1) if match else 'unreported'
        raw_path = output/'spice.raw'
        previous_raw_stamp = raw_path.stat().st_mtime_ns if raw_path.is_file() else None
        result = subprocess.run([str(NGSPICE), '-b', 'circuit.spice'], cwd=output, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=15)
        log = result.stdout + '\n' + result.stderr; (output/'spice.log').write_text(log)
        record['exit_code'] = result.returncode
        if result.returncode != 0 or not raw_path.is_file() or raw_path.stat().st_mtime_ns == previous_raw_stamp or re.search(r'(?im)^\s*(?:fatal error|error:|warning: singular matrix)', log):
            raise RuntimeError('ngspice did not complete a valid converged analysis; see spice.log')
        record.update(compiler.results(_raw_values(output/'spice.raw')), status='completed')
    except CalculationError as error:
        record.update(status='unsupported', error=str(error))
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        record.update(status='failed', error=str(error))
    if record['status'] != 'completed' and not (output/'spice.log').exists():
        (output/'spice.log').write_text(record.get('error', 'SPICE did not run') + '\n')
    record['seconds'] = time.monotonic()-started
    save(output/'spice-result.json', record)
    return record
