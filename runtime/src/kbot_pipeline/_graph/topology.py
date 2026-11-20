"""Evidence-preserving conductor segmentation and pin/net assembly.

Reuses the Ckt-Bot topology-next pixel observer (2026-11-12 review-source-v1).
This is a deterministic baseline, not a trained conductor segmentation model.
Boxes define exclusion hypotheses; only observed lead contacts establish anchors.
"""
from pathlib import Path
import hashlib
import math
from PIL import Image, ImageDraw
from ._topology_geometry import observe_topology_geometry
from .semantic_pins import infer_pin_roles
from .segmentation_adapter import load_conductor_mask


def assemble(image_path, components, observations=None):
    """Return candidate electrical nets from actual pixels.

    Components: [{id, type, bbox: [x0,y0,x1,y1]}] in original-image pixels.
    observations may contain text [{id,bbox}], pins [{component_id, role,
    terminal_px, anchor_px, evidence}], ground_groups [[component_id,...]], and
    output_dir and conductor_mask (native PNG plus hash-bound .png.json sidecar). Pin observations must come from perception/manual annotation,
    never a VLM coordinate guess. Each ground group is an explicit same-reference
    observation; absent groups, supported earth-ground symbols share earth net.
    Ambiguous crossings are CUT and reported, never assumed joined.
    mask_contact_policy defaults to 'frozen'; 'native_short_lead_v1' and
    'contrast_supported_v2'/'contrast_fallback_v3' are explicit experimental conditions requiring an
    external conductor mask. The latter corroborates real mask leads against
    local RGB contrast/noise and two body-entry thresholds, without adding ink.
    v3 retains a strong native body/contact-band witness before weak corroboration.
    mask_contact_policy='probability_pin_path_v1' only adds missing-side pin links
    over short source-corroborated weak paths to observed body terminals, using
    the original sidecar high threshold and a fixed two-thirds auxiliary floor.
    v2 additionally accepts repeated-turn source entry when two
    thresholds agree on its native off-axis departure terminal.
    terminal_evidence_policy='body_local_v1' independently measures component
    body contrast and preserves separate native high-mask anchors. body_local_v2
    allows overlapping existing native uncertainty supports, emits interval-only
    terminal evidence with a null coordinate, and keeps attachment proof explicit.
    contact_group_policy='same_lead_v1' independently groups same-region bands
    only with filled native cross-section and probability evidence. same_lead_v2
    uses independent source-background evidence for the interior while retaining
    original high member support and the same-region inventory-only restriction.
    body_exclusion_policy='source_outline_fringe_v1' optionally removes only
    source-certified closed triangular body/fringe evidence with native lead
    protection. It leaves the original RGB crossing observer unchanged.
    conductor_gap_policy='supported_single_gap_v1' optionally adds evidenced graph
    links across exactly one weak pixel; it requires probability_u8/high .75 and
    leaves the conductor mask and RGB crossing policy unchanged.
    """
    observations = observations or {}
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    ids = [c['id'] for c in components]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate component id')
    parts = []
    for c in components:
        b = c['bbox']
        if len(b) != 4 or not all(math.isfinite(v) for v in b) or not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height):
            raise ValueError('Invalid component bbox in image frame')
        parts.append({'id':c['id'], 'class_name':c['type'], 'bbox_px':b})
    inventory = {'observations':{'image':{'width':width, 'height':height,
        'sha256':hashlib.sha256(Path(image_path).read_bytes()).hexdigest()},
        'detections':parts, 'ocr':[{'id':t.get('id',f'text-{i}'), 'bbox_px':t['bbox']}
        for i,t in enumerate(observations.get('text',[]) + [{'bbox':b} for b in observations.get('text_boxes',[])])]}}
    output = Path(observations['output_dir']) if observations.get('output_dir') else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    conductor_mask=None;conductor_evidence=None;conductor_probability=None
    gap_policy=observations.get('conductor_gap_policy','frozen')
    pin_path_policy=observations.get('mask_contact_policy') in ('probability_pin_path_v1','probability_pin_path_v2')
    needs_probability=gap_policy!='frozen' or pin_path_policy or observations.get('terminal_evidence_policy','frozen')!='frozen' or observations.get('contact_group_policy','frozen')!='frozen'
    if gap_policy not in ('frozen','supported_single_gap_v1'):raise ValueError('Unknown conductor gap policy')
    if needs_probability and observations.get('conductor_mask') is None:raise ValueError('Probability-aware topology policy requires conductor probabilities')
    if observations.get('conductor_mask') is not None:
        if output and Path(observations['conductor_mask']).resolve()==(output/'conductor-mask.png').resolve():
            raise ValueError('Conductor diagnostic output must not overwrite supplied mask')
        loaded=load_conductor_mask(observations['conductor_mask'],image_path,image.size,source_policy=observations.get('mask_source_policy','synthetic_validation'),with_probability=needs_probability)
        conductor_mask,conductor_evidence=loaded[:2]
        if needs_probability:conductor_probability=loaded[2]
    geometry = observe_topology_geometry(image, inventory,
        conductor_mask=conductor_mask,conductor_evidence=conductor_evidence,conductor_probability=conductor_probability,conductor_gap_policy=gap_policy,mask_contact_policy=observations.get('mask_contact_policy','frozen'),terminal_evidence_policy=observations.get('terminal_evidence_policy','frozen'),contact_group_policy=observations.get('contact_group_policy','frozen'),body_exclusion_policy=observations.get('body_exclusion_policy','frozen'),terminal_localization_policy=observations.get('terminal_localization_policy','frozen'),
        canonical_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        junction_convention=observations.get('junction_convention','explicit_marks'),
        mask_path=output/'conductor-mask.png' if output else None)
    from .device_contract import normalize_component_models
    normalize_component_models(components,geometry['component_models'])
    fragments = {f['id']: f for f in geometry['wire_fragments']}
    roots = {k:k for k in fragments}
    def find(x):
        while roots[x] != x:
            roots[x] = roots[roots[x]]; x = roots[x]
        return x
    def union(a,b):
        a,b = find(a),find(b)
        if a != b: roots[max(a,b)] = min(a,b)
    for link in geometry['connections']:
        if link['state'] == 'supported':
            for f in link['fragment_ids'][1:]: union(link['fragment_ids'][0], f)
    models = {m['component_id']:m for m in geometry['component_models']}
    unresolved = list(geometry['issues'])
    pins = []
    # Contact locations are wire-side anchors. A physical terminal requires a
    # separately observed native lead-to-body transition, never a bbox face.
    for t in geometry['terminals']:
        a = t['attachment']; f = a['fragment_id']
        p = {'id':t['id'], 'component_id':t['component_id'], 'role':None,
             'terminal_px':t.get('body_terminal_px'), 'anchor_px':t['point_px'], 'net_id':None,
             'state':a['state'], 'evidence':a['evidence_refs'],
             'geometric_side':t['geometric_role'], 'terminal_localization':t.get('terminal_localization'), '_fragment':f}
        if t.get('contact_group_evidence'):p['contact_group_evidence']=t['contact_group_evidence']
        pins.append(p)
    # The bounded sign observer produces pixel evidence, never VLM coordinates.
    semantic_policy=observations.get('semantic_pin_policy','frozen')
    automatic = infer_pin_roles(image_path,components,pins,semantic_pin_policy=semantic_policy)
    current_policy=observations.get('current_role_policy','off');current_observations=None
    if current_policy!='off':
        from .current_source_roles import infer_current_source_roles
        current_observations=infer_current_source_roles(image_path,components,pins,current_role_policy=current_policy)
        automatic['pins'].extend(current_observations['pins']);automatic['unresolved'].extend(current_observations['unresolved'])
    unresolved.extend(automatic['unresolved'])
    supplied = observations.get('pins',[])
    inferred = [p for p in automatic['pins'] if not any(
        q.get('component_id')==p['component_id'] and q.get('anchor_px') is not None
        and math.dist(q['anchor_px'],p['anchor_px'])<=3 for q in supplied)]
    # Bind semantic pin evidence jointly: one observation cannot claim two pins.
    used = set()
    for obs in supplied + inferred:
        if obs.get('component_id') not in ids or not obs.get('evidence'):
            raise ValueError('Pin evidence requires known component and provenance')
        anchor = obs.get('anchor_px')
        terminal = obs.get('terminal_px')
        for point in (anchor, terminal):
            if point is not None and (len(point)!=2 or not all(math.isfinite(v) for v in point) or not (0<=point[0]<width and 0<=point[1]<height)):
                raise ValueError('Pin coordinate outside image')
        matches = [p for p in pins if p['component_id']==obs['component_id'] and p['id'] not in used
                   and anchor is not None and p['anchor_px'] is not None
                   and math.dist(anchor,p['anchor_px']) <= 3]
        if len(matches) != 1:
            unresolved.append({'reason':'semantic_pin_anchor_not_unique','component_id':obs['component_id'], 'role':obs.get('role')})
            continue
        p = matches[0]; used.add(p['id']); p.update(role=obs.get('role'), terminal_px=terminal if terminal is not None else p['terminal_px'],
            evidence=p['evidence']+obs['evidence'])
        if obs.get('semantic_observations'):p['semantic_observations']=obs['semantic_observations']
    ground_ids = [c['id'] for c in components if c['type']=='ground' and models[c['id']]['state']=='supported']
    ground_policy=observations.get('ground_domain_policy','frozen');ground_domains=None
    if ground_policy not in ('frozen','explicit_ground_domains_v1'):raise ValueError('Unknown ground domain policy')
    if ground_policy!='frozen':
        if 'ground_groups' in observations:raise ValueError('Explicit groups and label-domain policy need one authority')
        from .ground_domains import assign_ground_domains
        ground_domains=assign_ground_domains([c for c in components if c['id'] in ground_ids],observations.get('labels',[]))
        groups=ground_domains['groups'];unresolved.extend(ground_domains['unresolved'])
    else:groups = observations.get('ground_groups',[ground_ids])
    ground_roots = []
    for group in groups:
        if not set(group) <= set(ground_ids):
            unresolved.append({'reason':'ground_alias_without_supported_ground_shape','component_ids':group}); continue
        fs = [p['_fragment'] for p in pins if p['component_id'] in group and p['_fragment']]
        for f in fs[1:]: union(fs[0], f)
        ground_roots.extend(fs)
    grouped = {}
    for f in fragments: grouped.setdefault(find(f),[]).append(f)
    net_for = {root:'net-'+hashlib.sha256('|'.join(sorted(fs)).encode()).hexdigest()[:12] for root,fs in grouped.items()}
    nets = [{'id':net_for[root], 'pins':[], 'fragment_ids':sorted(fs),
             'ground':any(find(f)==root for f in ground_roots), 'evidence':['observed_conductor_paths'] + (['same_reference_earth_ground_shape'] if any(find(f)==root for f in ground_roots) else []),
             'verified':False} for root,fs in sorted(grouped.items())]
    by_net = {n['id']:n for n in nets}
    if ground_domains is not None:
        for net in nets:
            names={ground_domains['domains'].get(p['component_id']) for p in pins if p['_fragment'] is not None and net_for[find(p['_fragment'])]==net['id'] and ground_domains['domains'].get(p['component_id']) is not None}
            if len(names)==1:net['ground_domain']=next(iter(names))
            elif names:net['ground_domains']=sorted(names);net['ground_domain_relation']='explicitly_observed_same_conductor'

    for link in geometry.get('gap_connections',[]):
        for root in {find(f) for f in link['fragment_ids']}:
            by_net[net_for[root]]['evidence'].append('supported_single_gap_v1:'+link['id'])
    for p in pins:
        f = p.pop('_fragment')
        if f:
            p['net_id'] = net_for[find(f)]; by_net[p['net_id']]['pins'].append(p['id'])
        if p['terminal_px'] is None:
            unresolved.append({'reason':'symbol_body_terminal_interval_only' if (p.get('terminal_localization') or {}).get('transition_uncertainty') else 'symbol_body_terminal_unlocalized', 'pin_id':p['id']})
        if p['role'] is None and next(c['type'] for c in components if c['id']==p['component_id'])=='op_amp':
            unresolved.append({'reason':'op_amp_semantic_pin_role_unobserved', 'pin_id':p['id']})
    result = {'pins':pins, 'nets':nets, 'evidence':{'method':'existing_kbot_pixel_topology',
        'segmentation':'external_mask_body_exclusion_and_crossing_screen' if conductor_mask is not None else 'threshold_body_exclusion_and_crossing_screen', **geometry['evidence'], **({'current_role_policy':current_policy,'current_role_observations':current_observations} if current_observations is not None else {}), **({'ground_domain_policy':ground_policy,'ground_domains':ground_domains} if ground_domains is not None else {}), **({'semantic_pin_policy':semantic_policy} if semantic_policy!='frozen' else {})},
        'unresolved':unresolved, 'intersections':geometry['intersections'], 'gap_connections':geometry.get('gap_connections',[]),
        'wire_fragments':geometry['wire_fragments'], 'component_models':geometry['component_models'],
        'external_ports':geometry['external_ports'], 'verified':False,
        'status':'partial' if unresolved or geometry['external_ports'] else 'observed'}
    if output:
        canvas = image.copy(); draw = ImageDraw.Draw(canvas)
        colors = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4']
        for i,n in enumerate(nets):
            color = colors[i%len(colors)]
            for f in n['fragment_ids']:
                line = [tuple(p) for p in fragments[f]['polyline_px']]
                if len(line)>1: draw.line(line,fill=color,width=2)
        for c in components: draw.rectangle(c['bbox'],outline='#888888',width=1); draw.text(tuple(c['bbox'][:2]),c['id'],fill='blue')
        for p in pins:
            if p['anchor_px']:
                x,y=p['anchor_px']; draw.ellipse((x-3,y-3,x+3,y+3),fill='orange'); draw.text((x+4,y+4),p['role'] or '?',fill='red')
        canvas.save(output/'pin-net-overlay.png')
        result['mask_path']=str((output/'conductor-mask.png').resolve())
        result['overlay_path']=str((output/'pin-net-overlay.png').resolve())
    return result
