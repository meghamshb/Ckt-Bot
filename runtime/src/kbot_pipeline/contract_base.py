"""Versioned circuit contract. Structural validity never implies circuit accuracy."""
import json
from pathlib import Path

VERSION = 'kbot.circuit/1.0'


def validate(circuit):
    import jsonschema
    jsonschema.validate(circuit, json.loads(Path(__file__).with_name('circuit.schema.json').read_text()))
    components = {c['id']: c for c in circuit['components']}
    pins = {p['id']: p for p in circuit['pins']}
    nets = {n['id']: n for n in circuit['nets']}
    for key, lookup in [('components', components), ('pins', pins), ('nets', nets)]:
        if len(lookup) != len(circuit[key]):
            raise ValueError('Duplicate '+key+' IDs')
    for p in pins.values():
        if p['component_id'] not in components:
            raise ValueError('Unknown pin component')
        if p['net_id'] is not None and p['net_id'] not in nets:
            raise ValueError('Unknown pin net')
    for n in nets.values():
        if len(n['pins']) != len(set(n['pins'])):
            raise ValueError('Duplicate pin in net')
        for pid in n['pins']:
            if pid not in pins or pins[pid]['net_id'] != n['id']:
                raise ValueError('Nonreciprocal net incidence')
    for p in pins.values():
        if p['net_id'] and p['id'] not in nets[p['net_id']]['pins']:
            raise ValueError('Missing reciprocal net incidence')
    for c in components.values():
        if set(c['pins']) != {p['id'] for p in pins.values() if p['component_id'] == c['id']}:
            raise ValueError('Component pin inventory differs')
    for label in circuit['labels']:
        if label.get('component_id') is not None and label['component_id'] not in components:
            raise ValueError('Unknown label owner')
    claimed = [l['id'] for l in circuit['labels']]
    if len(claimed) != len(set(claimed)):
        raise ValueError('A label has multiple owners')
    return circuit
