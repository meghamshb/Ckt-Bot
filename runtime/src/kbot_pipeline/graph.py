"""Pin-aware assembly: measured paths, explicit crossing evidence, scoped aliases.

No nearest-neighbour gap completion, invented terminals, VLM netlist replacement,
or electrical-plausibility repairs. The preserved K-Bot extractor supplies pixels;
this module owns all ID binding and electrical equivalence operations.
"""
from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
from .io import sha256
from .contract_base import validate as validate_base


from ._graph.device_contract import TWO_PIN, physical_arities, required_roles
ALIASES = {'+':'positive', '-':'negative', '−':'negative', 'non_inverting':'noninverting', 'out':'output'}


class UnionFind:
    def __init__(self, ids):
        ids = list(ids)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate net ID')
        self.parent = {x:x for x in ids}
        self.order = {x:i for i,x in enumerate(ids)}

    def find(self, x):
        if x not in self.parent:
            raise ValueError(f'Unknown net ID: {x}')
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.order[a] > self.order[b]:
                a, b = b, a
            self.parent[b] = a


def extract(source, perception, mask, output, *, junction_convention='continuous_ink'):
    """Use the existing assembler once, in a fresh CPU process after SAM exits."""
    from ._graph.topology import assemble
    from PIL import Image
    from ._graph.symbol_identity import physical_components
    components,identity_events=physical_components(Image.open(source).convert('RGB'),perception['components'])
    ground_groups = [[c['id']] for c in components if c['type']=='ground']
    raw = assemble(source, components + perception.get('junctions', []), {
        'conductor_mask': str(mask), 'mask_source_policy': 'pretrained_provider_default',
        'output_dir': str(output), 'ground_groups': ground_groups,
        'semantic_pin_policy': 'observation_consensus_v1',
        'junction_convention': junction_convention,
        'terminal_localization_policy': 'source_body_v1',
    })
    pin_names = {p['id']:f'p{i+1}' for i,p in enumerate(raw['pins'])}
    net_names = {n['id']:f'n{i+1}' for i,n in enumerate(raw['nets'])}
    for p in raw['pins']:
        p['source_id'], p['id'] = p['id'], pin_names[p['id']]
        p['net_id'] = net_names.get(p['net_id'])
        p['role'] = ALIASES.get(p['role'],p['role'])
    for n in raw['nets']:
        n['source_id'], n['id'] = n['id'], net_names[n['id']]
        n['pins'] = [pin_names[p] for p in n['pins']]
    for issue in raw['unresolved']:
        if issue.get('pin_id') in pin_names:
            issue['pin_id'] = pin_names[issue['pin_id']]
    for i,event in enumerate(raw['intersections']):
        event['source_id'], event['id'] = event['id'], f'x{i+1}'
        event['arms'] = _arms(event, raw)
        occupied={p['net_id'] for p in raw['pins'] if p.get('net_id')}
        event['relevant_to_components']=any(a['net_id'] in occupied for a in event['arms'])
    raw['components'] = copy.deepcopy(components)
    raw['component_identity_observations']=identity_events
    raw['junctions'] = perception.get('junctions', [])
    raw['image'] = {'path':str(Path(source).resolve()), 'sha256':sha256(source)}
    from ._graph.final_references import geometry_uncertainties
    existing_issue_ids={x.get('id') for x in raw['unresolved'] if x.get('id')}
    raw['unresolved'].extend(x for x in geometry_uncertainties(raw) if x['id'] not in existing_issue_ids)
    from ._graph.current_references import observe_current_references
    from ._graph.wire_terminations import classify_terminations
    raw['unresolved'].extend(classify_terminations(source,raw))
    from ._graph.source_directions import source_roles
    from PIL import Image
    stroke=raw['evidence'].get('stroke_scene',{}).get('stroke_width_px',1.)
    raw['source_direction_observations']=source_roles(Image.open(source).convert('RGB'),components,raw['pins'],stroke)
    for observation in raw['source_direction_observations']:
        for role in ('from','to'):
            pin=next(p for p in raw['pins'] if p['id']==observation[role])
            if pin.get('role') not in (None,role):
                raw['unresolved'].append({'reason':'conflicting_native_source_direction','component_ids':[pin['component_id']],'blocking':True})
            else:
                pin['role']=role;pin['role_observation']=observation
    raw['reference_arrows']=observe_current_references(raw)
    return raw


def _arms(event, graph):
    """Name actual fragment endpoints at a cut; never manufacture bbox terminals."""
    fragment_net = {f:n['id'] for n in graph['nets'] for f in n.get('fragment_ids', [])}
    center = event['point_px']
    radius = event.get('cut_radius_px', 8) + 5
    points = []
    for fragment in graph['wire_fragments']:
        line = fragment['polyline_px']
        if len(line)<2:
            continue
        for point in (line[0],line[-1]):
            distance = math.dist(center,point)
            if not 1 < distance <= radius:
                continue
            if any(math.dist(point,p['point_px'])<2 for p in points):
                continue
            points.append({'point_px':point, 'net_id':fragment_net[fragment['id']],
                'fragment_id':fragment['id'],
                'vector':[(point[j]-center[j])/distance for j in (0,1)]})
    points.sort(key=lambda p:math.atan2(p['vector'][1],p['vector'][0]))
    for i,p in enumerate(points):
        p['id'] = event['id']+f'a{i+1}'
    return points


def _opposite_pairs(arms):
    if len(arms) != 4:
        raise ValueError('A bridge requires four observed arms')
    options = [((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
    scored=[]
    for pairs in options:
        dots=[sum(x*y for x,y in zip(arms[a]['vector'],arms[b]['vector'])) for a,b in pairs]
        if all(v <= -.75 for v in dots):
            scored.append((sum(dots),pairs))
    if len(scored)!=1:
        raise ValueError('Crossing arms have no unique opposing pairing')
    return scored[0][1]


def finalize(candidate, readings):
    """Bind readings to existing entities, then derive final pin-to-net incidence."""
    circuit = copy.deepcopy(candidate)
    components = {c['id']:c for c in circuit['components']}
    pins = {p['id']:p for p in circuit['pins']}
    nets = {n['id']:n for n in circuit['nets']}
    uf = UnionFind(nets)
    rows = readings['components']
    ids = [r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(components):
        raise ValueError('The diagram reader must return each supplied component exactly once')
    issues = []
    labels = []
    names_by_net={n:[] for n in nets};merge_records=[]
    from ._graph.ground_domains import DOMAINS
    def domains(root):
        return {(a['scope'],a['text']) for nid in nets if uf.find(nid)==root
            for a in names_by_net[nid] if a['kind']=='ground' or a['text'] in DOMAINS}
    def merge(a,b,kind,provenance):
        """Only this boundary may add final electrical equivalence."""
        before_a,before_b=uf.find(a),uf.find(b)
        record={'id':'merge-'+str(len(merge_records)+1),'kind':kind,'source_net_ids':[a,b],
            'source_roots':[before_a,before_b],'provenance':copy.deepcopy(provenance)}
        da,db=domains(before_a),domains(before_b)
        if kind not in ('observed_crossing_join','observed_bridge_continuation') and da and db and not da.intersection(db):
            record.update(status='rejected',reason='explicit_ground_domain_conflict')
            merge_records.append(record)
            issues.append({'reason':'ground_domain_equivalence_conflict','net_ids':[a,b],
                'domains':sorted(da|db),'merge_id':record['id'],'blocking':True})
            return False
        uf.union(a,b)
        record.update(status='already_equivalent' if before_a==before_b else 'applied',final_root=uf.find(a))
        merge_records.append(record)
        return True
    for row in rows:
        c = components[row['id']]
        for field in ('value','unit','parameters','model'):
            c[field] = row.get(field)
        name=row.get('name')
        c['names'] = [name] if name and name not in (c['id'],c['type']) else []
        c['pins'] = [p['id'] for p in pins.values() if p['component_id']==c['id']]
        c['evidence'] = c.get('evidence', []) + [{'source':'vision_model_reading','raw':row}]
        assigned=set()
        for meaning in row.get('pin_roles', []):
            pid=meaning['pin_id'];role=meaning['role']
            if pid not in pins or pins[pid]['component_id']!=c['id'] or pid in assigned:
                raise ValueError('The diagram reader referenced an unknown, duplicate or foreign pin')
            from .vision import ROLE_TYPES
            if role not in ROLE_TYPES.get(c['type'], []):
                raise ValueError('Pin role is incompatible with component type')
            assigned.add(pid)
            p=pins[pid]
            if p.get('role') and p['role']!=role:
                issues.append({'reason':'conflicting_observed_pin_role','component_id':c['id'],
                    'pin_id':pid,'pixel_role':p['role'],'vlm_role':role,'blocking':True})
                p['role']=None
            else:
                p['role']=role
                p['evidence'].append({'source':'vision_model_visible_role','role':role})
        labels.append({'id':'reading-'+c['id'], 'component_id':c['id'],
            'text':row.get('value'), 'name':row.get('name'), 'unit':row.get('unit')})

    # Intersections are semantic observations about supplied local pixels, not
    # permission for The diagram reader to redraw arbitrary electrical connections.
    events={e['id']:e for e in circuit.get('intersections', [])}
    resolved=set();seen=set()
    for row in readings.get('crossings', []):
        xid=row['id']
        if xid not in events or xid in seen:
            raise ValueError('Unknown or duplicate crossing ID')
        seen.add(xid);event=events[xid]
        if event['state']!='unresolved' or row['kind']=='unresolved' or not event.get('relevant_to_components',True):
            continue
        arms=event.get('arms', [])
        if row['kind']=='joined_dot' and 3 <= len(arms) <= 4:
            for arm in arms[1:]:
                merge(arms[0]['net_id'],arm['net_id'],'observed_crossing_join',{'crossing_id':xid,'kind':'joined_dot'})
        elif row['kind']=='bridge_overpass':
            try:
                pairs=_opposite_pairs(arms)
            except ValueError as error:
                issues.append({'reason':'crossing_pairing_unresolved','crossing_id':xid,
                    'detail':str(error),'blocking':True})
                continue
            for a,b in pairs:
                merge(arms[a]['net_id'],arms[b]['net_id'],'observed_bridge_continuation',
                    {'crossing_id':xid,'arm_ids':[arms[a]['id'],arms[b]['id']],'kind':'bridge_overpass'})
            event['arm_pairs']=[[arms[a]['id'],arms[b]['id']] for a,b in pairs]
        else:
            continue
        event.update(state='joined' if row['kind']=='joined_dot' else 'separate',
            semantic_evidence={'source':'vision_model_explicit_visual_mark','kind':row['kind']})
        resolved.add(event['source_id'])

    # Named nets are scoped exact aliases. Each occurrence must bind an existing
    # wire net. Measurement labels and textual component names are not aliases.
    groups={}
    for index,row in enumerate(readings.get('net_labels', [])):
        nid=row['net_id'];text=row['text'].strip();scope=row['scope'].strip()
        if nid not in nets or not text or not scope:
            raise ValueError('A net label needs a known net, nonempty text and scope')
        if text in nets or text in components or text in pins or text.startswith('@'):
            issues.append({'reason':'reference_annotation_echo_rejected','reading':row,'blocking':False})
            continue
        key=(scope,text,row['kind'])
        groups.setdefault(key,[]).append(nid)
        names_by_net[nid].append({'text':text,'scope':scope,'kind':row['kind']})
        labels.append({'id':f'net-label-{index+1}', **row, 'component_id':None})
    for key,ids in groups.items():
        for nid in ids[1:]:
            merge(ids[0],nid,'scoped_net_alias',{'scope':key[0],'text':key[1],'label_kind':key[2],
                'source':'diagram_reader_physical_net_label','verified':False})
    # Unlabelled observed ground symbols share the default image ground domain;
    # explicitly labelled ground domains remain separate unless physically joined.
    models={m['component_id']:m for m in circuit.get('component_models',[])}
    ground_witnesses={n:[] for n in nets}
    for c in components.values():
        if c['type']!='ground':continue
        owned=[pins[p] for p in c['pins']]
        if len(owned)==1 and owned[0].get('state')=='supported' and models.get(c['id'],{}).get('state')=='supported':
            ground_witnesses[owned[0]['net_id']].append(c['id'])
    for nid,net in nets.items():
        if net['ground'] and not ground_witnesses[nid]:
            issues.append({'reason':'physical_ground_witness_unavailable','net_ids':[nid],'blocking':True})
    unlabelled=[n['id'] for n in nets.values() if n['ground'] and ground_witnesses[n['id']]
        and not any(names_by_net[nid] for nid in nets if uf.find(nid)==uf.find(n['id']))]
    for nid in unlabelled[1:]:
        merge(unlabelled[0],nid,'same_image_supported_earth_ground',
            {'policy':'unlabelled_supported_earth_symbols_share_image_domain',
             'ground_component_ids':ground_witnesses[unlabelled[0]]+ground_witnesses[nid]})

    merged={}
    for nid,net in nets.items():
        root=uf.find(nid)
        entry=merged.setdefault(root,{'id':root,'pins':[],'fragment_ids':[], 'ground':False,
            'evidence':[], 'aliases':[], 'source_net_ids':[], 'verified':False})
        entry['source_net_ids'].append(nid)
        entry['fragment_ids'].extend(net.get('fragment_ids',[]))
        entry['ground'] |= net['ground']
        entry['evidence'].extend(net.get('evidence',[]))
        entry['aliases'].extend(names_by_net[nid])
    for p in pins.values():
        if p.get('net_id'):
            p['net_id']=uf.find(p['net_id'])
            merged[p['net_id']]['pins'].append(p['id'])
    for net in merged.values():
        net['fragment_ids']=sorted(set(net['fragment_ids']))
        net['evidence'].append('explicit_pin_incidence_and_scoped_alias_resolution')
    circuit['nets']=[n for n in merged.values() if n['pins']]
    circuit['unattached_wire_candidates']=[n for n in merged.values() if not n['pins']]
    mapping={nid:uf.find(nid) for nid in nets}
    circuit['net_equivalences']=merge_records
    circuit['equivalence_policy']={'authority':'final_assembler','mathematical_gauges_used':False,
        'ground_policy':'supported_shape_and_explicit_scoped_domains'}
    for port in circuit.get('external_ports', []):
        if port.get('net_id') in nets:
            port['source_net_id']=port['net_id']
            port['net_id']=uf.find(port['net_id'])
    for port in circuit.get('external_ports',[]):
        port['label']=readings.get('external_port_names',{}).get(port['id'])
    if 'analysis_context' in readings:
        context=copy.deepcopy(readings['analysis_context'])
        for drive in context.get('voltage_drives',[]):
            for field in ('positive_net_id','negative_net_id'):
                if drive[field] not in nets:raise ValueError('Question drive uses an unobserved net')
                drive[field]=uf.find(drive[field])
        circuit['analysis_context']=context
    circuit['labels']=labels
    circuit['arrows']=[]
    circuit['unbound_current_references']=readings.get('unbound_current_references',[])
    from ._graph.final_references import validate_wire_cut,mask_pixels_sha256,canonicalize_references
    proof_choices=[choice for arrow in candidate.get('reference_arrows',[]) for choice in arrow['choices'].values() if choice.get('cut_proof')]
    mask_hash=mask_pixels_sha256(candidate) if proof_choices else None
    before_events={event['id']:event['state'] for event in candidate.get('intersections',[])}
    changed_events={event['id'] for event in circuit.get('intersections',[]) if event['state']!=before_events.get(event['id'])}
    for index,row in enumerate(readings.get('current_references', [])):
        if 'terminal_current_terms' in row:
            observed=next((a for a in circuit['reference_arrows'] if a['id']==row['arrow_id']),None)
            if observed is None or row['choice_id'] not in observed['choices']:
                raise ValueError('Current reference must bind an observed wire section')
            if observed['kind']=='mesh_current':
                original_nodes={p['net_id'] for p in candidate['pins'] if p.get('net_id')}
                if len({uf.find(n) for n in original_nodes})!=len(original_nodes) or changed_events:
                    issues.append({'reason':'mesh_face_basis_changed_by_net_aliases','blocking':True})
            terms=row['terminal_current_terms']
            if terms!=observed['choices'][row['choice_id']]['terminal_current_terms'] or any(t['pin_id'] not in pins or t['coefficient'] not in (-1,1) for t in terms):
                raise ValueError('Current reference changed a measured wire-cut incidence')
            validation=None
            if observed['kind']=='wire_current':
                validation=validate_wire_cut(candidate,observed['choices'][row['choice_id']],terms,mapping,
                    changed_events=changed_events,mask_hash=mask_hash)
                if validation['status']!='validated':
                    issues.append({'reason':validation['reason'],'label':row['label'],
                        'component_ids':sorted({pins[t['pin_id']]['component_id'] for t in terms}),
                        'blocking':validation['blocking'],'cut_validation':validation})
                if validation['blocking']:
                    circuit['unbound_current_references'].append({'label':row['label'],'evidence':validation['reason']})
                    circuit.setdefault('invalidated_current_references',[]).append({'observation':copy.deepcopy(row),'validation':validation})
                    continue
            circuit['arrows'].append({'id':f'arrow-{index+1}',**row,'kind':observed['kind'],
                'evidence':observed['evidence'],'verified':False,**({'cut_validation':validation} if validation else {})})
        else:
            # Archived observations use component-pin current references.
            a=pins.get(row['from_pin_id']);b=pins.get(row['to_pin_id'])
            if not a or not b or a['id']==b['id'] or a['component_id']!=row['component_id'] or b['component_id']!=row['component_id']:
                raise ValueError('Archived current reference has invalid terminal ownership')
            circuit['arrows'].append({'id':f'arrow-{index+1}',**row,'kind':'archived_component_current',
                'evidence':'archived_vl_owned_component_pin_reference','verified':False})

    circuit['voltage_references']=[]
    bound={}
    for row in readings.get('voltage_references',[]):
        label=row['label'].strip()
        if not label or label in pins or label in nets or label in components:
            raise ValueError('Voltage measurement needs a printed label, not a generated reference ID')
        endpoints=[]
        for field in ('positive_net_id','negative_net_id'):
            nid=row[field]
            if nid is not None and nid not in nets:raise ValueError('Voltage reference uses an unknown net')
            endpoints.append(uf.find(nid) if nid is not None else None)
        # A source's named voltage cannot also be read with opposite polarity
        # as an independent measurement. Reject conflicting evidence before math.
        for source in components.values():
            if source['type'] not in ('voltage_source','dependent_voltage_source'):continue
            names=set(source.get('names',[]))|{source.get('value')}
            if label not in names:continue
            owned={pins[pid].get('role'):pins[pid].get('net_id') for pid in source['pins']}
            pair=(owned.get('positive'),owned.get('negative'))
            if all(pair) and all(endpoints) and tuple(endpoints)!=pair:
                issues.append({'reason':'conflicting_source_voltage_reference','component_id':source['id'],'label':label,'blocking':True})
        key=tuple(endpoints)
        if label in bound and bound[label]!=key:
            issues.append({'reason':'conflicting_voltage_reference','label':label,'blocking':True})
        if bound.get(label)==key:continue
        bound[label]=key
        circuit['voltage_references'].append({'label':label,'positive_net_id':key[0],'negative_net_id':key[1],
            'source_positive_net_id':row['positive_net_id'],'source_negative_net_id':row['negative_net_id'],
            'pair_id':row.get('pair_id'),'positive_site_id':row.get('positive_site_id'),'negative_site_id':row.get('negative_site_id'),'basis':row.get('basis'),'visual_evidence':row.get('visual_evidence'),'evidence':'vision_model_observed_site_binding_no_net_merges'})

    for issue in candidate.get('unresolved', []):
        if any(str(issue.get('id','')).startswith(x+':') for x in resolved):
            continue
        if issue.get('reason')=='op_amp_semantic_pin_role_unobserved' and pins.get(issue.get('pin_id'),{}).get('role'):
            continue
        issues.append(copy.deepcopy(issue))
    issues.extend({'reason':'visible_component_missing_from_detector','detail':x,'blocking':True}
        for x in readings.get('missing_components', []))
    issues.extend({'reason':'unresolved_visual_reading','detail':x,'blocking':True}
        for x in readings.get('issues', []))
    circuit['schema_version']='kbot.circuit/1.0'
    circuit['status']='best_effort_unverified'
    circuit['evidence']=[{'source':'existing_kbot_assembler','details':candidate.get('evidence',{})}]
    circuit['unresolved']=issues
    canonicalize_references(circuit,mapping)
    from ._graph.device_contract import normalize_component_models
    normalize_component_models(circuit['components'],circuit.get('component_models',[]))
    validate(circuit)
    circuit['electrical_coverage']=electrical_coverage(circuit)
    circuit['electrical_issues']=circuit['electrical_coverage']['issues']
    circuit['relations']=relations(circuit)
    return circuit


def validate(circuit):
    """Topology invariants through the public assembler interface, not a VLM judge."""
    validate_base(circuit)
    components={c['id']:c for c in circuit['components']}
    w,h=circuit.get('image',{}).get('size',[float('inf'),float('inf')])
    for p in circuit['pins']:
        for field in ('terminal_px','anchor_px'):
            point=p.get(field)
            if point is not None and (len(point)!=2 or not all(math.isfinite(v) for v in point)
                or not 0<=point[0]<w or not 0<=point[1]<h):
                raise ValueError('Pin coordinate is outside the canonical image')
        if p.get('net_id') and p.get('state') in ('ambiguous','unsupported','unresolved','unattached'):
            raise ValueError('An unresolved pin cannot assert electrical net membership')
    for c in components.values():
        if len(c['pins'])!=len(set(c['pins'])):
            raise ValueError('A component repeats a pin')
    return circuit


def electrical_coverage(circuit):
    """One relevant-coverage decision for numerical eligibility and relations."""
    issues=[];pins={p['id']:p for p in circuit['pins']}
    all_nets=circuit['nets']+circuit.get('unattached_wire_candidates',[])
    fragment_net={f:n['id'] for n in all_nets for f in n.get('fragment_ids',[])}
    occupied={p['net_id'] for p in pins.values() if p.get('net_id')}
    occupied.update(p['net_id'] for p in circuit.get('external_ports',[]) if p.get('state')=='observed_external_terminal' and p.get('net_id'))
    ignored=[];contracts={}
    for c in circuit['components']:
        cp=[pins[p] for p in c['pins']];kind=c['type']
        allowed=physical_arities(kind);arity_ok=len(cp) in allowed
        expected=allowed[0] if len(allowed)==1 else list(allowed)
        roles=required_roles(kind,len(cp))
        contracts[c['id']]={'observed_pin_count':len(cp),'allowed_physical_arities':list(allowed),
            'required_roles':sorted(roles),'physical_inventory_complete':arity_ok,
            **({'analysis_model_supported':len(cp)==3,'analysis_scope':'ideal_linear_three_signal_pins'} if kind=='op_amp' else {})}
        if not arity_ok:
            issues.append({'reason':'component_pin_inventory_unresolved','component_id':c['id'], 'count':len(cp),'expected':expected})
        if any(not p.get('net_id') for p in cp):
            issues.append({'reason':'pin_net_attachment_unresolved','component_id':c['id']})
        if any(p.get('net_id') and p.get('state')!='supported' for p in cp):
            issues.append({'reason':'pin_attachment_state_unresolved','component_id':c['id']})
        for role in roles:
            if sum(p.get('role')==role for p in cp)!=1:
                issues.append({'reason':'pin_role_unresolved','component_id':c['id'],'role':role})
        # Value/parameter availability belongs to the calculation module.
    for issue in circuit.get('unresolved', []):
        if not issue.get('blocking'):
            continue
        fragments=issue.get('fragment_ids',[]);affected={fragment_net[f] for f in fragments if f in fragment_net}
        has_owner=bool(issue.get('component_ids') or issue.get('component_id') or issue.get('pin_id'))
        if fragments and all(f in fragment_net for f in fragments) and not affected.intersection(occupied) and not has_owner:
            ignored.append(copy.deepcopy(issue));continue
        issues.append(issue)
    return {'status':'incomplete' if issues else 'complete','issues':issues,
        'irrelevant_observations':ignored,'device_contracts':contracts,
        'policy':'known_unattached_fragments_may_be_irrelevant;empty_or_unknown_scope_is_not_proof'}


def electrical_issues(circuit):
    """Compatibility entry point backed by the same scoped coverage contract."""
    return electrical_coverage(circuit)['issues']


def relations(circuit):
    """Exact two-terminal relations; a shared net alone does not mean series."""
    coverage=circuit.get('electrical_coverage') or electrical_coverage(circuit)
    incomplete=coverage['status']!='complete'
    if incomplete:
        return {'parallel_pairs': [], 'series_pairs': [], 'status': 'withheld',
            'reason': 'circuit_incomplete'}
    pins={p['id']:p for p in circuit['pins']}; by_component={}
    for c in circuit['components']:
        nets=[pins[p]['net_id'] for p in c['pins']]
        if c['type'] in TWO_PIN and len(nets)==2 and None not in nets and len(set(nets))==2:
            by_component[c['id']]=set(nets)
    nonbranch={c['id'] for c in circuit['components'] if c['type']=='ground'}
    boundary_nets={p['net_id'] for p in circuit.get('external_ports', [])
        if p.get('state')=='observed_external_terminal' and p.get('net_id')}
    boundary_nets.update(n['id'] for n in circuit['nets'] if n.get('ground'))
    parallel=[];series=[]
    net_pin_count={n['id']:sum(pins[p]['component_id'] not in nonbranch for p in n['pins']) for n in circuit['nets']}
    for a,b in itertools.combinations(by_component,2):
        x,y=by_component[a],by_component[b]
        if x==y:
            parallel.append({'components':[a,b],'nets':sorted(x)})
        common=x&y
        if len(common)==1:
            net=next(iter(common))
            if net_pin_count[net]==2 and net not in boundary_nets:
                series.append({'components':[a,b],'shared_net':net})
    return {'parallel_pairs':parallel,'series_pairs':series}
