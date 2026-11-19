"""Exact linear circuit calculation from observed incidence and a Qwen plan.

One tableau formulation and one linear-algebra algorithm. Free variables stay
free: only identifiable requested quantities are reported. No model-generated
equations, Python execution, guessed device values or numerical backup path.
"""
from __future__ import annotations

import ast
import math
import re
import sympy as sp
from sympy.solvers.solveset import NonlinearError


class CalculationError(ValueError):
    pass


PAIRED = {'resistor', 'capacitor', 'inductor', 'impedance_block',
          'voltage_source', 'current_source',
          'dependent_voltage_source', 'dependent_current_source'}
PREFIX = {'': 1, 'p': sp.Rational(1, 10**12), 'n': sp.Rational(1, 10**9),
          'u': sp.Rational(1, 10**6), 'm': sp.Rational(1, 1000),
          'k': 1000, 'M': 10**6, 'G': 10**9}
UNITS = {'ohm': 'ohm', 'V': 'V', 'A': 'A', 'F': 'F', 'H': 'H',
         'Hz': 'Hz', 'rad/s': 'rad/s', 'S': 'S'}


def identifier(value):
    text = str(value).translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    return re.sub(r'[_{}\\]', '', re.sub(r'([A-Za-zΑ-ω][A-Za-z0-9_{}Α-ω]*)\(t\)',r'\1',text))


def clean(value):
    text = str(value).strip().replace('−', '-').replace('µ', 'u').replace('μ', 'u')
    text = text.replace('Ω', 'ohm').replace('Ω', 'ohm')
    text = re.sub(r'(?i)ohms?', 'ohm', text)
    text = text.replace('\\angle', '∠').replace('\\circ', '°')
    text = text.replace('\\cdot', '*').replace('×', '*').replace('\\,', '')
    text = re.sub(r'\\(?:mathrm|text)\{([^{}]*)\}', r'\1', text)
    return text.strip('$ ').replace(' ', '')


def numeric_grouping(text):
    """Normalize only conventional comma-grouped numeric tokens.

    Function-reference argument separators remain intact. A comma occurring
    inside a numeric token must have complete three-digit integer groups;
    it is never interpreted as a decimal separator or a list of coefficients.
    """
    number=re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d*)?(?:[eE][+-]?\d+)?')
    def replace(match):
        token=match.group()
        if ',' not in token:return token
        integer=re.split(r'[.eE]',token,maxsplit=1)[0]
        if not re.fullmatch(r'[1-9]\d{0,2}(?:,\d{3})+',integer):
            raise CalculationError('Malformed thousands grouping in numeric literal')
        return token.replace(',','')
    return number.sub(replace,text)


def expression(value, names, voltage=None, current=None):
    """Bounded arithmetic grammar; names resolve only to supplied constants/facts."""
    text = numeric_grouping(identifier(clean(value)).replace('^', '**'))
    token_pattern=re.compile(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-zΑ-ω][A-Za-z0-9Α-ω]*|\*\*|[()+*/,-]')
    tokens=[];position=0
    while position<len(text):
        match=token_pattern.match(text,position)
        if match is None:raise CalculationError('Unsupported arithmetic character')
        token=match.group();previous=tokens[-1] if tokens else ''
        number=lambda t:bool(t and (t[0].isdigit() or t[0]=='.'))
        name=lambda t:bool(t and (t[0].isalpha()))
        if (number(previous) and (name(token) or token=='(')) or (previous==')' and (name(token) or number(token) or token=='(')) or (name(previous) and previous not in ('V','I') and token=='('):tokens.append('*')
        tokens.append(token);position=match.end()
    text=''.join(tokens)
    if not text or len(text) > 512:
        raise CalculationError('Empty or excessive arithmetic expression')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as error:
        raise CalculationError(f'Unsupported expression: {value}') from error
    if len(list(ast.walk(tree))) > 128:
        raise CalculationError('Arithmetic expression exceeds the bounded grammar')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(ast.get_source_segment(text, node))
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id == 'j':return sp.I
            if node.id == 'pi':return sp.pi
            raise CalculationError(f'Unbound symbol: {node.id}')
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div):
                if b == 0: raise CalculationError('Division by zero in a circuit value')
                return a/b
            if isinstance(node.op, ast.Pow) and b.is_Integer and abs(b) <= 12: return a**b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [a.id for a in node.args if isinstance(a, ast.Name)]
            if len(args) == len(node.args):
                if node.func.id == 'V' and voltage is not None and len(args) == 1:
                    return voltage(args[0])
                if node.func.id == 'I' and current is not None and len(args) in (1, 2):
                    return current(*args)
        raise CalculationError('Only arithmetic and observed V(net)/I(device,terminal) references are allowed')
    result = visit(tree.body)
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise CalculationError('Nonfinite circuit value')
    return result


def quantity(value, unit=None, names=None, dimensions=None):
    """Parse a physical literal/expression once; prefixes are never multiplied twice."""
    text = clean(value); annotated = clean(unit or '')
    found = re.search(r'(p|n|u|m|k|M|G)?(ohm|rad/s|Hz|V|A|F|H|S)$', text)
    last_identifiers=re.findall(r'[A-Za-zΑ-ω][A-Za-z0-9Α-ω]*',identifier(text))
    if last_identifiers and last_identifiers[-1] in (names or {}):found=None
    suffix = (found.group(1) or '')+found.group(2) if found else annotated
    body = text[:found.start()] if found else text
    if suffix:
        match = re.fullmatch(r'(p|n|u|m|k|M|G)?(ohm|rad/s|Hz|V|A|F|H|S)', suffix)
        if not match: raise CalculationError(f'Unsupported unit: {suffix}')
        prefix, dimension = match.group(1) or '', match.group(2)
        multiplier = PREFIX[prefix]
        if found and annotated:
            other = re.fullmatch(r'(p|n|u|m|k|M|G)?(ohm|rad/s|Hz|V|A|F|H|S)', annotated)
            if not other or other.group(2) != dimension or (other.group(1) and other.group(1) != prefix):
                raise CalculationError('Value and unit annotations conflict')
    else:
        multiplier, dimension = 1, None
    if '∠' in body:
        magnitude, phase = body.split('∠')
        radians = phase.endswith('rad'); phase = phase.removesuffix('rad').removesuffix('°')
        angle = expression(phase, names or {}) * (1 if radians else sp.pi/180)
        number = expression(magnitude, names or {}) * (sp.cos(angle)+sp.I*sp.sin(angle))
    else:
        body = re.sub(r'j(?=\d)', 'j*', body)
        number = expression(body, names or {})
    if dimensions and any(identifier(name) in re.findall(r'[A-Za-zΑ-ω][A-Za-z0-9Α-ω]*',identifier(body)) for name in dimensions):
        from sympy.physics.units import ohm,volt,ampere,farad,henry,siemens,hertz,radian,second,convert_to,Quantity
        from sympy.physics.units.systems.si import SI
        from sympy.physics.units.util import check_dimensions
        units={'ohm':ohm,'V':volt,'A':ampere,'F':farad,'H':henry,'S':siemens,'Hz':hertz,'rad/s':radian/second}
        typed={name:value*units[dimensions[name]] if dimensions.get(name) in units else value for name,value in (names or {}).items()}
        if '∠' not in body:
            try:
                physical=check_dimensions(expression(body,typed));dimensional=SI.get_dimensional_expr(physical)
            except ValueError as error:raise CalculationError('Incompatible dimensions in symbolic component value') from error
            if dimensional!=1:
                choices=[key for key,u in units.items() if SI.get_dimension_system().equivalent_dims(dimensional,SI.get_dimensional_expr(u))]
                if not choices:raise CalculationError('Unsupported compound component dimension')
                inferred=choices[0]
                if dimension and not SI.get_dimension_system().equivalent_dims(dimensional,SI.get_dimensional_expr(units[dimension])):raise CalculationError('Symbolic value and annotated unit dimensions conflict')
                scalar=sp.simplify(convert_to(physical,units[inferred])/units[inferred])
                if scalar.has(Quantity):raise CalculationError('Symbolic component value did not reduce to a coherent SI unit')
                return scalar,inferred
    return sp.simplify(number*multiplier), dimension


def sinusoidal(value,unit,names):
    """A*cos(omega*t+phase) / A*sin(...), converted to cosine-reference phasors."""
    raw=re.sub(r'([A-Za-z0-9_{}])\s+(?=cos|sin)',r'\1*',str(value))
    text=clean(raw)
    suffix=re.search(r'(p|n|u|m|k|M|G)?(V|A)$',text)
    declared=suffix.group() if suffix else ''
    body=text[:suffix.start()] if suffix else text
    match=re.fullmatch(r'(.*?)(cos|sin)(.+)',body)
    if not match:return None
    amplitude,kind,argument=match.groups()
    if amplitude and amplitude[-1] not in '0123456789*)':return None
    amplitude=amplitude.rstrip('*') or '1'
    amplitude,dim=quantity(amplitude+declared,unit,names)
    if amplitude.is_real is not True:raise CalculationError('Time-domain sinusoidal amplitude must be real')
    argument=numeric_grouping(argument)
    # A sign belongs to the surrounding sum, not to the degree literal.
    # Consuming '+' would turn omega*t+phase into omega*t*(phase).
    degree_number=r'(?<![\w.])((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)°'
    argument=re.sub(degree_number,r'(\1*pi/180)',argument)
    time=sp.Symbol('t',real=True)
    phase=expression(argument,{**names,'t':time})
    omega=sp.simplify(sp.diff(phase,time));offset=sp.simplify(phase-omega*time)
    if omega.free_symbols or offset.free_symbols or omega.is_positive is not True:
        raise CalculationError('Source must be one sinusoid at a positive, known angular frequency')
    if kind=='sin':offset-=sp.pi/2
    return sp.simplify(amplitude*(sp.cos(offset)+sp.I*sp.sin(offset))),dim,omega


def question_assignments(question):
    """Read explicit physical name=value declarations, never infer a given from a query."""
    number=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    units=r'(?:[pnumkMGµμ]?\s*(?:rad/s|Hz|ohms?|Ω|Ω|V|A|F|H|S))'
    value=r'(?:'+number+r'\s*∠\s*'+number+r'\s*(?:°|rad)?|'+number+r')\s*'+units
    pattern=r'(?<![A-Za-z0-9_])([A-Za-zΑ-ω][A-Za-z0-9_{}Α-ω]*)\s*=\s*('+value+r')(?![A-Za-z0-9_])'
    return [{'symbol':identifier(m.group(1)),'value':m.group(2),'evidence_quote':m.group()} for m in re.finditer(pattern,question)]


def quoted(quote, question):
    """Accept lossless whitespace differences in an otherwise exact source span.

    PDF line wrapping and JSON transcription may change whitespace. Words,
    case, punctuation and their order must stay unchanged; a partial word is
    never evidence for a model assertion.
    """
    if not isinstance(quote, str) or not isinstance(question, str):return False
    span=' '.join(quote.split());source=' '.join(question.split())
    if not span:return False
    before=r'(?<!\w)' if re.match(r'\w',span[0]) else ''
    after=r'(?!\w)' if re.match(r'\w',span[-1]) else ''
    return re.search(before+re.escape(span)+after,source) is not None

def literal_quoted(value,quote):
    # Preserve word boundaries in prose. Removing all spaces turns a literal
    # followed by "and" into an apparent longer identifier (e.g. V-and).
    normalized=' '.join(clean(part) for part in re.split(r'\s+',quote or ''))
    tokens=re.findall(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-zΑ-ω_]+|[^\s]',clean(value))
    if not tokens:return False
    pattern=r'(?<![\w.])'+r'\s*'.join(re.escape(token) for token in tokens)+r'(?!\w|\.\d)'
    return bool(re.search(pattern,normalized))



class Circuit:
    def __init__(self, facts, plan, question):
        self.facts, self.plan = facts, plan
        self.parts = {c['id']: c for c in facts['components']}
        nodes = {n for c in self.parts.values() for n in c['terminals'].values()}
        if not nodes or None in nodes: raise CalculationError('Missing electrical net')
        self.nodes = sorted(nodes)
        if len(self.parts) > 32: raise CalculationError('Linear calculation supports at most 32 devices')
        self.values = {}; self.overrides = {}
        permitted = set(self.parts)
        for c in self.parts.values():
            permitted.update(identifier(n) for n in c.get('names', []))
            permitted.update(re.findall(r'[A-Za-zΑ-ω][A-Za-z0-9Α-ω]*', identifier(str(c.get('value', '')))))
        for row in plan['assignments']:
            name = identifier(row['symbol'])
            if name not in permitted or not quoted(row['evidence_quote'], question):
                raise CalculationError('Question assignment is not anchored to a known symbol and exact question quote')
            if not literal_quoted(row['value'],row['evidence_quote']):
                raise CalculationError('Assignments must copy a supplied literal, without Qwen arithmetic')
            parsed = quantity(row['value'])
            physical=[c for c in self.parts.values() if c['id']==name or name in [identifier(n) for n in c.get('names',[])]]
            if physical and parsed[1] is None:raise CalculationError('A physical question assignment must retain its given unit')
            if name in self.values and self.values[name] != parsed: raise CalculationError('Conflicting question assignments')
            self.values[name] = parsed
        self.constants = {k: v[0] for k, v in self.values.items()}
        self.omega = None
        analysis = plan['analysis']; self.mode = analysis['mode']
        self.input_convention=analysis.get('phasor_convention','unspecified')
        if self.mode=='ac' and self.input_convention!='unspecified':
            quote=analysis.get('convention_evidence')
            if not quoted(quote,question) or self.input_convention not in quote.lower():raise CalculationError('Input phasor convention lacks explicit question evidence')
        self.convention=self.input_convention
        if analysis['frequency'] is not None:
            sources = [question]+[str(c.get('value', '')) for c in self.parts.values()]
            if not any(quoted(analysis['frequency_evidence'], source) for source in sources) or not literal_quoted(analysis['frequency'],analysis['frequency_evidence']):
                raise CalculationError('Frequency is not anchored to source text')
            f, dim = quantity(analysis['frequency'])
            if f.is_real is not True or f <= 0 or dim not in ('Hz', 'rad/s'):
                raise CalculationError('AC frequency must be positive Hz or rad/s')
            self.omega = f*(2*sp.pi if dim == 'Hz' else 1)
        self.source_waves={}
        for cid,c in self.parts.items():
            if c['type'] not in ('voltage_source','current_source') or not c.get('value'):continue
            frequency_names={'omega':self.omega,'ω':self.omega,'f':self.omega/(2*sp.pi)} if self.omega is not None else {}
            wave=sinusoidal(c['value'],c.get('unit'),{**frequency_names,**self.constants})
            if wave is None:continue
            if self.mode!='ac':raise CalculationError('Sinusoidal source needs AC steady-state analysis')
            if self.omega is not None and sp.simplify(self.omega-wave[2])!=0:raise CalculationError('Source and analysis frequencies disagree')
            self.omega=wave[2];self.source_waves[cid]=wave[:2];self.convention='peak'
        self.ideal = set()
        for row in plan['ideal_op_amps']:
            c = self.parts.get(row['component_id'])
            if not c or c['type'] != 'op_amp' or not quoted(row['evidence_quote'], question):
                raise CalculationError('Ideal linear op-amp model lacks supplied evidence')
            self.ideal.add(c['id'])
        self.drives = []
        ports = {p['net_id'] for p in facts['external_ports']}
        for i, row in enumerate(plan['voltage_drives']):
            p, n = row['positive_net_id'], row['negative_net_id']
            if p not in ports or n not in ports | set(facts['ground_nets']) or not quoted(row['evidence_quote'], question):
                raise CalculationError('A boundary voltage drive needs observed terminals and an explicit question assignment')
            if not literal_quoted(row['value'],row['evidence_quote']): raise CalculationError('Boundary drive must copy its given literal')
            value, unit = quantity(row['value'])
            if unit != 'V': raise CalculationError('Boundary voltage drive needs voltage units')
            self.drives.append((f'drive{i+1}', p, n, value))

    def value(self, c):
        for name in [c['id']]+[identifier(n) for n in c.get('names', [])]:
            if name in self.values: return self.values[name]
        if c['id'] in self.source_waves:return self.source_waves[c['id']]
        raw = c.get('value')
        if raw is None:
            raise CalculationError(f"Missing value or question assignment for {c['id']}")
        token = identifier(clean(raw))
        if token in self.values: return self.values[token]
        return quantity(raw,c.get('unit'),self.constants,{name:unit for name,(_,unit) in self.values.items() if unit})

    def system(self, *, zero_sources=False, transfer=None, probe=None, port_transfer=None):
        volts = {n: sp.Symbol('V_'+n) for n in self.nodes}
        currents = {}; terminal_currents = {}; kcl = {n: sp.S.Zero for n in self.nodes}
        equations = []; unknowns = list(volts.values()); parents = {n: n for n in self.nodes}
        def find(n):
            while parents[n] != n: n = parents[n]
            return n
        def connect(nodes):
            for n in nodes[1:]: parents[find(n)] = find(nodes[0])
        def V(n):
            if n not in volts: raise CalculationError(f'Unknown voltage node {n}')
            return volts[n]
        def I(cid, terminal=None):
            if terminal is not None:
                if (cid, terminal) not in terminal_currents: raise CalculationError('Unknown component terminal-current reference')
                return terminal_currents[cid, terminal]
            if cid not in currents: raise CalculationError('Unknown component branch-current reference')
            return currents[cid]
        for cid, c in self.parts.items():
            kind = c['type']; terminals = c['terminals']; connect(list(terminals.values()))
            if kind in PAIRED:
                if len(terminals) != 2: raise CalculationError('Linear two-terminal device has wrong arity')
                keys = ['positive', 'negative'] if 'positive' in terminals else ['from', 'to'] if 'from' in terminals else list(terminals)
                a, b = keys; current = sp.Symbol('I_'+cid); currents[cid] = current; unknowns.append(current)
                terminal_currents[cid, a] = current; terminal_currents[cid, b] = -current
                kcl[terminals[a]] += current; kcl[terminals[b]] -= current
            elif kind == 'op_amp':
                if cid not in self.ideal or set(terminals) != {'inverting', 'noninverting', 'output'}:
                    raise CalculationError('Only explicitly ideal, linear three-terminal op-amps are supported')
                current = sp.Symbol('I_'+cid+'_output'); unknowns.append(current)
                for key in ('inverting', 'noninverting'): terminal_currents[cid, key] = sp.S.Zero
                terminal_currents[cid, 'output'] = current; kcl[terminals['output']] += current
                equations.append(V(terminals['inverting'])-V(terminals['noninverting']))
            else:
                raise CalculationError(f'Unsupported linear device model: {kind}')
        references = {}
        def register(label, value):
            key = identifier(label)
            if not re.fullmatch(r'[A-Za-zΑ-ω][A-Za-z0-9Α-ω]*', key): return
            if key in references and sp.simplify(references[key]-value) != 0: raise CalculationError('Conflicting measured reference definitions')
            references[key] = value
        for row in self.facts['current_references']:
            if 'expression' not in row: raise CalculationError('Unresolved current direction')
            register(row['label'], expression(row['expression'], {}, V, I))
        for row in self.facts['voltage_references']:
            if row['positive_net_id'] is not None and row['negative_net_id'] is not None:
                register(row['label'], V(row['positive_net_id'])-V(row['negative_net_id']))
        for cid, c in self.parts.items():
            if c['type'] not in ('voltage_source', 'current_source'): continue
            aliases = list(c.get('names', []))
            if re.fullmatch(r'[A-Za-zΑ-ω][A-Za-z0-9_{}Α-ω]*', str(c.get('value', ''))): aliases.append(c['value'])
            for alias in aliases:
                if c['type'] == 'voltage_source':
                    register(alias, V(c['terminals']['positive'])-V(c['terminals']['negative']))
                else: register(alias, I(cid))
        for cid, c in self.parts.items():
            kind = c['type']; terminals = c['terminals']
            if kind == 'op_amp': continue
            positive = next(key for key in terminals if terminal_currents[cid, key] == currents[cid])
            negative = next(key for key in terminals if key != positive)
            voltage = V(terminals[positive])-V(terminals[negative]); current = I(cid)
            if kind.startswith('dependent_'):
                value,dim = quantity(c.get('value') or '',c.get('unit'),{**self.constants, **references})
                expected='V' if kind=='dependent_voltage_source' else 'A'
                if dim is not None and dim!=expected:raise CalculationError('Dependent-source output unit is incompatible')
                equations.append(voltage-value if kind == 'dependent_voltage_source' else current-value)
            elif kind in ('voltage_source', 'current_source'):
                if transfer and cid==transfer[0]:value=transfer[1]
                elif zero_sources:value=sp.S.Zero
                else:
                    value,dim=self.value(c)
                    if dim is not None and dim!=('V' if kind=='voltage_source' else 'A'):raise CalculationError('Source value has an incompatible unit')
                if self.mode=='ac' and self.source_waves and cid not in self.source_waves and not zero_sources and transfer is None:
                    if self.input_convention=='unspecified':raise CalculationError('Mixed waveform/phasor sources need an explicit phasor convention')
                    if self.input_convention=='rms':value*=sp.sqrt(2)
                if self.mode == 'dc' and sp.im(value) != 0: raise CalculationError('Complex excitation needs AC analysis')
                equations.append(voltage-value if kind == 'voltage_source' else current-value)
            else:
                value, dim = self.value(c)
                if kind in ('resistor','impedance_block') and dim=='S':
                    if kind=='resistor' and sp.im(value)!=0:raise CalculationError('A resistor cannot have complex conductance')
                    if self.mode=='dc' and sp.im(value)!=0:raise CalculationError('Complex admittance needs AC analysis')
                    equations.append(current-value*voltage)
                    continue
                expected = {'resistor': 'ohm', 'impedance_block': 'ohm', 'capacitor': 'F', 'inductor': 'H'}[kind]
                if dim is not None and dim not in (expected, 'ohm'): raise CalculationError(f'Incompatible unit on {cid}: {dim}')
                if kind == 'resistor' and sp.im(value) != 0: raise CalculationError('A resistor cannot have complex resistance')
                if kind in ('resistor', 'impedance_block') or dim == 'ohm':
                    if self.mode == 'dc' and sp.im(value) != 0:raise CalculationError('Complex impedance needs AC analysis')
                    equations.append(voltage-value*current)
                elif self.mode == 'dc':
                    equations.append(current if kind == 'capacitor' else voltage)
                else:
                    if self.omega is None: raise CalculationError('Physical L/C values need an explicit AC frequency')
                    if value.is_real is not True or value <= 0: raise CalculationError('Physical L/C values must be positive real quantities')
                    equations.append(current-sp.I*self.omega*value*voltage if kind == 'capacitor' else voltage-sp.I*self.omega*value*current)
        def branch(cid, p, n, value):
            current = sp.Symbol('I_'+cid); unknowns.append(current); currents[cid] = current
            kcl[p] += current; kcl[n] -= current; connect([p, n]); equations.append(V(p)-V(n)-value)
            return current
        for cid, p, n, value in self.drives:
            if port_transfer and {p,n}==set(port_transfer[:2]):continue
            if self.mode=='ac' and self.source_waves:
                if self.input_convention=='unspecified':raise CalculationError('AC boundary drives need a declared amplitude convention')
                if self.input_convention=='rms':value*=sp.sqrt(2)
            branch(cid, p, n, 0 if zero_sources else value)
        if port_transfer:
            p,n,value=port_transfer;ports={p['net_id'] for p in self.facts['external_ports']}
            if p not in ports or n not in ports|set(self.facts['ground_nets']) or p==n:raise CalculationError('Transfer input needs distinct observed external terminals or a ground reference')
            branch('transfer_input',p,n,value)
        probe_current = None
        if probe:
            probe_voltage = sp.Symbol('port_excitation'); unknowns.append(probe_voltage)
            probe_current = branch('port_probe', *probe, probe_voltage)
        islands = {}
        for n in self.nodes: islands.setdefault(find(n), []).append(n)
        gauges = []
        for nodes in islands.values():
            grounds = sorted(set(nodes) & set(self.facts['ground_nets']))
            if len(grounds) > 1: raise CalculationError('Distinct ground domains need an explicit electrical reference relationship')
            negatives = [c['terminals']['negative'] for c in self.parts.values() if c['type'] == 'voltage_source' and c['terminals']['negative'] in nodes]
            reference = grounds[0] if grounds else min(negatives or nodes)
            gauges.append({'net_id': reference, 'kind': 'physical_ground' if grounds else 'mathematical_gauge'})
            equations.append(V(reference))
            equations.extend(kcl[n] for n in nodes if n != reference)
        try:
            matrix, rhs = sp.linear_eq_to_matrix(equations, unknowns)
            solutions = sp.linsolve((matrix, rhs), unknowns)
        except NonlinearError as error:
            raise CalculationError('Device/control relation is nonlinear') from error
        if solutions is sp.EmptySet or not solutions: raise CalculationError('Linear circuit equations are inconsistent')
        solution = next(iter(solutions)); mapping = dict(zip(unknowns, solution))
        residuals = [sp.simplify(e.xreplace(mapping)) for e in equations]
        if any(r != 0 for r in residuals): raise CalculationError('Linear result failed exact circuit-equation residual checks')
        return LinearResult(unknowns, mapping, volts, terminal_currents, references, probe_current,
            {'variables': [str(x) for x in unknowns], 'matrix': [[str(x) for x in row] for row in matrix.tolist()],
             'rhs': [str(x) for x in rhs], 'solution_family': {str(k): str(v) for k, v in mapping.items()},
             'gauges': gauges, 'equations': [str(e)+' = 0' for e in equations], 'exact_residuals_zero': True})


class LinearResult:
    def __init__(self, unknowns, mapping, volts, currents, references, probe_current, evidence):
        self.unknowns, self.mapping, self.volts = set(unknowns), mapping, volts
        self.currents, self.references, self.probe_current, self.evidence = currents, references, probe_current, evidence

    def evaluate(self, value): return sp.simplify(value.xreplace(self.mapping))

    def unique(self, value):
        result = self.evaluate(value)
        if result.free_symbols & self.unknowns: raise CalculationError('Requested quantity is not identifiable from the linear constraints')
        return result

    def voltage(self, p, n):
        if p not in self.volts or n not in self.volts: raise CalculationError('Requested voltage uses an unknown net')
        return self.volts[p]-self.volts[n]


def record(target, value, unit, direction):
    value = sp.simplify(value)
    result = {'id': target['id'], 'name': target['name'], 'unit': unit, 'reference_direction': direction}
    if value in (sp.oo, -sp.oo):
        return {**result, 'kind': 'symbolic', 'real': None, 'imaginary': None, 'expression': str(value)}
    if value.free_symbols: raise CalculationError('Requested numeric result still has unbound symbols')
    number = complex(sp.N(value, 17))
    if not (math.isfinite(number.real) and math.isfinite(number.imag)): raise CalculationError('Nonfinite numerical result')
    real = float(sp.N(sp.re(value), 17)); imaginary = float(sp.N(sp.im(value), 17))
    return {**result, 'kind': 'scalar' if sp.im(value) == 0 else 'phasor', 'real': real, 'imaginary': imaginary, 'expression': None}


def calculate(facts, plan, question, *, _prepared=None):
    circuit = _prepared.circuit if _prepared else Circuit(facts, plan, question)
    results = []; workings = []; evidence = []; base = _prepared.base if _prepared else None
    ids = set();selectors=set()
    for target in plan['targets']:
        if not target['id'] or target['id'] in ids: raise CalculationError('Requested quantity IDs must be unique')
        ids.add(target['id']); kind = target['kind']
        reference_kinds={identifier(r['label']):'reference_current' for r in facts['current_references']}
        reference_kinds.update({identifier(r['label']):'reference_voltage' for r in facts['voltage_references'] if r['positive_net_id'] is not None and r['negative_net_id'] is not None})
        named_references={identifier(label) for label in (target['id'],target['name']) if identifier(label) in reference_kinds}
        if kind=='voltage':
            from .measurements import voltage_symbols
            requested={identifier(label) for label in voltage_symbols(question)}
            claimed={identifier(label) for label in (target['id'],target['name'])}
            if (requested&claimed)-set(reference_kinds):raise CalculationError('Named voltage target has no observed physical binding')
        selector=tuple(sorted((k,str(v)) for k,v in target.items() if k not in ('id','name')))
        if selector in selectors:raise CalculationError('Different requested quantities repeat the same physical measurement binding')
        selectors.add(selector)
        if kind in ('resistance', 'impedance'):
            if kind == 'resistance' and circuit.mode != 'dc': raise CalculationError('Use impedance for AC port analysis')
            system = circuit.system(zero_sources=True, probe=(target['positive_net_id'], target['negative_net_id']))
            voltage = system.evaluate(system.voltage(target['positive_net_id'], target['negative_net_id']))
            incoming = system.evaluate(-system.probe_current)
            free = sorted((voltage.free_symbols | incoming.free_symbols) & system.unknowns, key=str)
            changes = [(sp.diff(voltage, s), sp.diff(incoming, s)) for s in free]
            nonzero = [(v, i) for v, i in changes if i != 0]
            if not nonzero:
                if not any(v != 0 for v, i in changes): raise CalculationError('Port has no identifiable impedance relation')
                value = sp.oo
            else:
                value = sp.cancel(nonzero[0][0]/nonzero[0][1])
                if any(sp.simplify(v-value*i) != 0 for v, i in changes): raise CalculationError('Port impedance is not unique')
            unit = 'ohm'; direction = f"port {target['positive_net_id']} to {target['negative_net_id']}; independent sources deactivated"
        elif kind == 'voltage_gain':
            excitation = sp.Symbol('input_excitation', real=True)
            if 'input_component_id' in target:
                source = circuit.parts.get(target['input_component_id'])
                if not source or source['type'] != 'voltage_source':raise CalculationError('Voltage transfer input source is not observed')
                system=circuit.system(zero_sources=True,transfer=(source['id'],excitation))
            else:
                p,n=target['input_positive_net_id'],target['input_negative_net_id'];sources=[c for c in circuit.parts.values() if c['type']=='voltage_source' and set(c['terminals'].values())=={p,n}]
                if len(sources)>1:raise CalculationError('Transfer input has multiple parallel independent voltage sources')
                if sources:
                    source=sources[0];sign=1 if source['terminals']['positive']==p else -1;system=circuit.system(zero_sources=True,transfer=(source['id'],sign*excitation))
                else:system=circuit.system(zero_sources=True,port_transfer=(p,n,excitation))
            output = system.unique(system.voltage(target['positive_net_id'], target['negative_net_id']))
            value = sp.simplify(sp.diff(output, excitation)); unit = 'V/V'
            direction = f"V({target['positive_net_id']})-V({target['negative_net_id']}) per volt of the observed input; other independent sources zero"
        elif kind=='unresolved':
            if not quoted(target['question_quote'],question):raise CalculationError('Unresolved request lacks an exact question quote')
            raise CalculationError(target['reason'])
        else:
            if base is None: base = circuit.system()
            system = base
            if kind == 'voltage':
                value = system.unique(system.voltage(target['positive_net_id'], target['negative_net_id'])); unit = 'V'
                direction = f"V({target['positive_net_id']})-V({target['negative_net_id']})"
            elif kind == 'current':
                key = (target['component_id'], target['terminal'])
                if key not in system.currents: raise CalculationError('Current target does not name an owned component terminal')
                value = system.unique(system.currents[key]); unit = 'A'; direction = f'entering {key[0]}.{key[1]}'
            elif kind in ('reference_current','reference_voltage'):
                key = identifier(target['label'])
                if key not in system.references: raise CalculationError('Printed current has no observed binding')
                rows=facts['current_references'] if kind=='reference_current' else facts['voltage_references']
                if not any(identifier(r['label']) == key for r in rows): raise CalculationError('Reference is not an observed measurement of this kind')
                value = system.unique(system.references[key])*(-1 if target['sense']=='opposite' else 1)
                unit='A' if kind=='reference_current' else 'V';direction=target['sense']+' '+target['label']
            elif kind == 'voltage_ratio':
                numerator = system.unique(system.voltage(target['positive_net_id'], target['negative_net_id']))
                denominator = system.unique(system.voltage(target['input_positive_net_id'], target['input_negative_net_id']))
                if denominator == 0: raise CalculationError('Requested voltage ratio has a zero denominator')
                value = sp.cancel(numerator/denominator); unit = 'V/V'; direction = 'requested output/input voltage ratio'
            elif kind == 'power':
                cid = target['component_id']; part = circuit.parts[cid]
                if part['type'] not in PAIRED: raise CalculationError('Power calculation needs a modeled two-terminal device')
                keys = list(part['terminals']); key = 'positive' if 'positive' in keys else 'from' if 'from' in keys else keys[0]
                other = next(k for k in keys if k != key)
                v = system.unique(system.voltage(part['terminals'][key], part['terminals'][other])); i = system.unique(system.currents[cid, key])
                if circuit.mode == 'ac':
                    convention = circuit.convention
                    if convention not in ('rms','peak'):raise CalculationError('AC power needs a known RMS/peak input convention')
                    power = v*sp.conjugate(i)*(1 if convention == 'rms' else sp.Rational(1, 2))
                else: power = v*i
                power *= -1 if target['direction'] == 'delivered' else 1
                value, unit = {'real': (sp.re(power), 'W'), 'reactive': (sp.im(power), 'var'),
                               'apparent': (sp.Abs(power), 'VA'), 'complex': (power, 'VA')}[target['power_kind']]
                direction = target['direction']+' by '+cid
            else: raise CalculationError('Unknown requested calculation')
        for key in named_references:
            expected=system.references.get(key)
            if expected is None:raise CalculationError('Named measurement has no complete observed definition')
            if kind==reference_kinds[key] and identifier(target.get('label',''))==key:continue
            if kind=='voltage' and reference_kinds[key]=='reference_voltage':actual=system.voltage(target['positive_net_id'],target['negative_net_id'])
            elif kind=='current' and reference_kinds[key]=='reference_current':actual=system.currents[target['component_id'],target['terminal']]
            else:raise CalculationError('Requested printed measurement has an incompatible physical kind')
            if sp.simplify(actual-expected)!=0:raise CalculationError('Requested printed measurement conflicts with its canonical observed reference')
        if circuit.mode=='ac' and kind in ('voltage','current','reference_voltage','reference_current'):
            amplitude=target.get('amplitude','as_given')
            if amplitude!='as_given':
                if circuit.convention not in ('rms','peak'):raise CalculationError('Amplitude conversion needs the input phasor convention')
                if amplitude!=circuit.convention:value*=sp.sqrt(2) if amplitude=='peak' else 1/sp.sqrt(2)
            direction+='; '+(circuit.convention if amplitude=='as_given' else amplitude)+' phasor'
        results.append(record(target, value, unit, direction))
        workings.append(f"{target['name']} = {sp.sstr(value)} {unit} ({direction})")
        evidence.append({'target_id': target['id'], 'exact_result': str(value), 'system': system.evidence})
    if not results: raise CalculationError('No requested quantities were identified')
    assumptions=[circuit.mode.upper()+' steady state.']
    assumptions.extend(cid+' explicitly ideal and linear.' for cid in sorted(circuit.ideal))
    gauges={(g['net_id'],g['kind']) for item in evidence for g in item['system']['gauges']}
    assumptions.extend('V('+net+')=0 is a mathematical reference.' for net,kind in sorted(gauges) if kind=='mathematical_gauge')
    return ({'status': 'answered', 'answer': '; '.join(f"{r['name']}: {r['expression'] if r['kind']=='symbolic' else format(r['real'], '.8g')+(' '+format(r['imaginary'], '+.8g')+'j' if r['kind']=='phasor' else '')} {r['unit']}" for r in results),
            'equations': workings, 'assumptions': assumptions, 'missing_evidence': [], 'quantities': results},
            {'algorithm': 'one exact linear tableau / SymPy linsolve; no regularization or backup solver', 'analyses': evidence})


class Analysis:
    """Computed physical facts shared with Qwen; selected results reuse this state."""
    def __init__(self,facts,context,question):
        self.facts,self.context,self.question=facts,context,question
        self.circuit=Circuit(facts,context,question);self.base=None
        self.state={'node_voltages':{},'terminal_currents':{},'printed_references':{},'phasor_convention':self.circuit.convention}
        self.state['analysis']={'mode':self.circuit.mode,
            'angular_frequency_rad_s':str(self.circuit.omega) if self.circuit.omega is not None else None}
        self.state['source_interpretations']={cid:{'observed_value':self.circuit.parts[cid]['value'],
            'exact_phasor':str(wave[0]),'unit':wave[1],'convention':self.circuit.convention}
            for cid,wave in self.circuit.source_waves.items()}
        if set(context['requested_kinds']) & {'voltage','current','power','voltage_ratio'}:
            self.base=self.circuit.system()
            def describe(expr):
                value=self.base.evaluate(expr)
                identifiable=not bool(value.free_symbols & self.base.unknowns)
                return {'identifiable':identifiable,'exact_value':str(value) if identifiable else None}
            self.state['node_voltages']={n:describe(v) for n,v in self.base.volts.items()}
            self.state['terminal_currents']={cid+'.'+key:describe(value) for (cid,key),value in self.base.currents.items()}
            self.state['printed_references']={name:describe(value) for name,value in self.base.references.items()}
            self.state['gauges']=self.base.evidence['gauges']
    def answer(self,selection):
        plan={**self.context,**selection}
        return calculate(self.facts,plan,self.question,_prepared=self)


def prepare(facts,context,question):
    return Analysis(facts,context,question)
