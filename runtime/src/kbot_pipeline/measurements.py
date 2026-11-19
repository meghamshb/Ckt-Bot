"""Visual measurement ownership: printed symbols -> observed sites -> nets.

The reader selects existing physical contacts. The reasoner selects a named
measurement; it cannot relocate that measurement to a convenient solved node.
"""
from __future__ import annotations
import re


def voltage_symbols(text):
    """Conventional voltage variables in questions/control expressions, not units."""
    pattern=r'(?<![A-Za-z_])([vV](?:_\{[A-Za-z0-9]+\}|_[A-Za-z0-9]+|[A-Za-z]{1,2}[0-9]*|[0-9]+))(?![A-Za-z0-9_])'
    return list(dict.fromkeys(m.group(1) for m in re.finditer(pattern,text)))


def ground_referenced(label,question):
    # Explicit question scope, not a visual guess or a global grounded default.
    return bool(re.search(re.escape(label)+r'\s+(?:relative\s+to|with\s+respect\s+to|referenced\s+to)\s+(?:the\s+)?(?:printed\s+)?ground\b',question,re.I))


def printed_polarity_requested(question):
    return bool(re.search(r'\bprinted\b[^.?!]*\bpolarit',question,re.I))


def sites(candidate):
    result=[{'id':p['id'],'kind':'component_contact','component_id':p['component_id'],
        'point_px':p['anchor_px'],'net_id':p['net_id']} for p in candidate['pins']
        if p.get('anchor_px') is not None and p.get('net_id') is not None and p.get('state')=='supported']
    for port in candidate.get('external_ports',[]):
        if port.get('state')=='observed_external_terminal':
            result.append({'id':f'e{len([r for r in result if r["kind"]=="external_terminal"])+1}',
                'kind':'external_terminal','port_id':port['id'],'point_px':port['circle_center_px'],'net_id':port['net_id']})
    for terminal in candidate.get('measurement_terminals',[]):
        if any(r['kind']=='external_terminal' and r['point_px']==terminal['circle_center_px'] for r in result):continue
        result.append({'id':'e'+str(len([r for r in result if r['kind']=='external_terminal'])+1),'kind':'external_terminal','point_px':terminal['circle_center_px'],'net_id':terminal['net_id'],'evidence':terminal['evidence']})
    for event in candidate.get('intersections',[]):
        owners={arm['net_id'] for arm in event.get('arms',[])}
        if event['state']=='joined' and len(owners)==1:
            result.append({'id':'j'+str(len([r for r in result if r['kind']=='wire_junction'])+1),
                'kind':'wire_junction','event_id':event['id'],'point_px':event['point_px'],'net_id':next(iter(owners))})
    return result


def normalize_bindings(value,candidate,question="",component_readings=None):
    from .voltage_pairs import polarity_pairs
    from .linear import identifier
    pairs={p['id']:p for p in polarity_pairs(candidate)}
    observed={p['id']:p for p in sites(candidate)};result=[]
    for label,row in value.items():
        if not isinstance(label,str) or not label.strip():raise ValueError('Empty measurement name')
        row=dict(row);pair_id=row.get('pair_id')
        if identifier(label) in {identifier(v) for v in voltage_symbols(question)} and printed_polarity_requested(question) and not ground_referenced(label,question) and pair_id is None and row.get('basis')!='unobserved':raise ValueError('Printed voltage polarity must select an observed electrode pair')
        if pair_id is not None:
            if pair_id not in pairs:raise ValueError('Unknown observed voltage pair')
            pair=pairs[pair_id]
            if pair['kind']=='source_symbol' and component_readings is not None:
                source=component_readings[pair['component_id']]
                if identifier(label) not in {identifier(source.get(key) or '') for key in ('name','value')}:raise ValueError('A source rating/expression is not a different named voltage')
            row.update(positive_site_id=pair['positive_site_id'],negative_site_id=pair['negative_site_id'],basis='native_polarity_pair')
        out={'pair_id':pair_id,'label':label,'basis':row['basis'],'visual_evidence':row['evidence'],
            'positive_site_id':row['positive_site_id'],'negative_site_id':row['negative_site_id']}
        for sense in ('positive','negative'):
            sid=row[sense+'_site_id']
            if sid is not None and sid not in observed:raise ValueError('Voltage endpoint is not an observed physical site')
            out[sense+'_net_id']=observed[sid]['net_id'] if sid is not None else None
        if row['basis']=='unobserved' and any(out[k+'_net_id'] for k in ('positive','negative')):raise ValueError('Unobserved voltage cannot invent an endpoint')
        if row['basis']!='unobserved' and (not row['evidence'].strip() or any(out[k+'_net_id'] is None for k in ('positive','negative'))):raise ValueError('A voltage binding needs two observed sites and visual evidence')
        ground={n['id'] for n in candidate['nets'] if n.get('ground')}
        if ground_referenced(label,question) and out['negative_net_id'] is not None and out['negative_net_id'] not in ground:raise ValueError('Explicit ground reference bound to a non-ground contact')
        result.append(out)
    return result


def require_requested_voltages(facts,question):
    from .linear import identifier,CalculationError
    bound={identifier(r['label']) for r in facts['voltage_references'] if r['positive_net_id'] is not None and r['negative_net_id'] is not None}
    # Explicit assignments supply source amplitudes, not requested measurements.
    unbound=[label for label in voltage_symbols(question) if not re.search(re.escape(label)+r'\s*=',question)
        and identifier(label) not in bound]
    if unbound:raise CalculationError('Requested voltage has no observed physical binding: '+', '.join(unbound))


def requested_ratio(question):
    from .linear import identifier
    names=voltage_symbols(question)
    for a in names:
        for b in names:
            if a!=b and re.search(re.escape(a)+r'\s*/\s*'+re.escape(b),question):return identifier(a),identifier(b)
    return None


def given_voltage_drives(question,bindings):
    """Compile explicit numeric terminal drives from the question, never image text.

    The value and its units are copied together. The named voltage's already
    observed electrode pair supplies both endpoints; no second net choice exists.
    Supported syntax: drive/hold/set a named terminal at/to a voltage, or apply a
    voltage to a named terminal. Unrecognized voltage drive commands are explicit
    errors rather than silently treated as unloaded measurement ports.
    """
    from .linear import identifier
    refs={identifier(r['label']):r for r in bindings};result=[]
    names=voltage_symbols(question)
    label='(?:'+'|'.join(re.escape(n) for n in sorted(names,key=len,reverse=True))+')' if names else r'(?!)'
    number=r'[+−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    voltage=number+r'\s*[pnumkMGµμ]?V\b'
    forward=re.compile(r'(?P<label>'+label+r')\s*(?:(?:terminal|port|node|input)\s*)?(?:at|to|=|with)\s*(?P<value>'+voltage+r')')
    reverse=re.compile(r'(?P<value>'+voltage+r')\s+(?:to|at)\s+(?:the\s+)?(?P<label>'+label+r')(?:\s+(?:terminal|port|node|input))?')
    for sentence in re.split(r'(?<=[.!?])\s+',question):
        action=re.search(r'\b(?:drive|hold|set|apply|excite)\b',sentence,re.I)
        if not action:continue
        matches=sorted([*forward.finditer(sentence,action.end()),*reverse.finditer(sentence,action.end())],key=lambda m:m.start())
        if re.search(voltage,sentence) and not matches:raise ValueError('Explicit voltage drive syntax has no named terminal/value binding')
        for match in matches:
            ref=refs.get(identifier(match['label']))
            if ref is None or ref['positive_net_id'] is None or ref['negative_net_id'] is None:raise ValueError('Driven voltage has no observed electrode pair')
            result.append({'positive_net_id':ref['positive_net_id'],'negative_net_id':ref['negative_net_id'],
                'value':match['value'],'evidence_quote':sentence,'voltage_label':ref['label']})
    return result


def current_symbols(text):
    pattern=r'(?<![A-Za-z_])([iI](?:_\{[A-Za-z0-9]+\}|_[A-Za-z0-9]+|[0-9]+)|i[oxL]|I[oxL])(?![A-Za-z0-9_])'
    names=[m.group(1) for m in re.finditer(pattern,text)]
    # A printed subscript can be typed without its underscore: i_s(t) -> is(t).
    # Explicit time notation makes the whole token a quantity name; the bare
    # English word "is" is never introduced as a current by this rule.
    timed=r'(?<![A-Za-z0-9_Α-ω])([iI](?:_\{[A-Za-z0-9Α-ω]+\}|_[A-Za-z0-9Α-ω]+|[A-Za-z0-9Α-ω]+)?)\s*\(\s*t\s*\)(?![A-Za-z0-9_Α-ω])'
    names.extend(m.group(1) for m in re.finditer(timed,text))
    # Bare I is a measurement only in explicit quantity context, not a pronoun.
    if re.search(r'\bcurrent\s+(?:labelled\s+|named\s+)?I\b|\bI\s+(?:is\s+the\s+)?current\b',text):names.append('I')
    return list(dict.fromkeys(names))


def control_dependencies(component_readings, component_kinds):
    """Lexical dependencies of printed controlled-source expressions only.

    This discovers names, never their physical endpoints or quantity kind.
    Explicit dimensional suffixes are not variables; j and pi are constants.
    Bare capital I remains a physical symbol, unlike SymPy's internal imaginary I.
    """
    from .linear import clean,identifier
    result={}
    unit=r'(?:[pnumkMGµμ]?(?:V|A|F|H|S|Hz|Ω)|ohms?|rad/s)'
    for cid,row in component_readings.items():
        if component_kinds[cid] not in ('dependent_voltage_source','dependent_current_source'):continue
        raw=str(row.get('value') or '')
        suffix=re.search(r'\s+'+unit+r'\s*$',raw)
        # A separated trailing unit is dimensional only after a complete
        # operand. In 2* A or 2 + V, the last name is itself an operand.
        if suffix and raw[:suffix.start()].rstrip() and raw[:suffix.start()].rstrip()[-1] not in '+−-*/^(,':
            raw=raw[:suffix.start()]
        text=re.sub(r'j(?=\d)', 'j*',identifier(clean(raw)))
        # Consume numbers first so scientific-notation exponents are not names.
        tokens=re.findall(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-zΑ-ω][A-Za-z0-9Α-ω]*',text)
        names={t for t in tokens if t[0].isalpha() and t not in {'j','pi','sin','cos','exp','sqrt'}}
        result[cid]=names
    return result


def requested_kind_options(question):
    """Limit reading categories to quantity vocabulary actually in the request."""
    kinds=[]
    for kind,pattern in [('current',r'\bcurrents?\b'),('voltage',r'\bvoltages?\b'),('power',r'\bpower\b'),('resistance',r'\bresistance\b'),('impedance',r'\bimpedance\b'),('voltage_gain',r'\bgain\b'),('voltage_ratio',r'\bratio\b')]:
        if re.search(pattern,question,re.I):kinds.append(kind)
    if current_symbols(question) and 'current' not in kinds:kinds.append('current')
    if voltage_symbols(question) and not any(k in kinds for k in ('voltage_gain','voltage_ratio')) and 'voltage' not in kinds:kinds.append('voltage')
    if requested_ratio(question):kinds+=['voltage_gain','voltage_ratio']
    return list(dict.fromkeys(kinds))
