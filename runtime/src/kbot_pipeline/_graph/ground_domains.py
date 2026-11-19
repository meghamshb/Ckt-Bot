"""Opt-in ground-reference grouping from supplied physical text regions only.

This module proposes aliases, not electrical connections or ground recognition.
The assembler must still require supported ground shapes and observed contacts.
"""
import math
import statistics
from .labels import _minimum_assignment

DOMAINS=frozenset(('GND','AGND','DGND','PGND','SGND'))
POLICY='explicit_ground_domains_v1'
MAX_DISTANCE=2.0
AMBIGUITY_MARGIN=0.2


def _box(item):
    value=item.get('bbox',item.get('bbox_px',item.get('bbox_xyxy')))
    if not isinstance(value,(list,tuple)) or len(value)!=4:return None
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):return None
    return list(value) if value[2]>value[0] and value[3]>value[1] else None


def _symbol_kind(component):
    # Use declared symbol subtype only. No pixel classification or semantic
    # inference from a minus sign, node number, component value or position.
    for key in ('ground_kind','ground_type','symbol_kind'):
        value=component.get(key)
        if isinstance(value,str):
            value=value.strip().lower()
            if value in ('earth','earth_ground'):return 'earth'
            if value in ('chassis','chassis_ground'):return 'chassis'
    return 'generic'


def assign_ground_domains(components,labels):
    """Return proposed groups/evidence; never mutate inputs or create coordinates.

    Only type=ground components are eligible; shape support remains the existing
    assembler's responsibility. Domain labels must be exact trimmed uppercase
    tokens, with original image bboxes. One text region and one component each
    occupy one assignment slot. Overlapping duplicate regions are not votes.
    """
    ids=[c['id'] for c in components]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate component IDs')
    grounds=sorted((c for c in components if c.get('type')=='ground'),key=lambda c:c['id'])
    domains={c['id']:None for c in grounds};unresolved=[];evidence=[];assignments=[]
    explicit=[];seen=set()
    for label in labels:
        text=label.get('text')
        if not isinstance(text,str) or text.strip() not in DOMAINS:continue
        rid=label.get('id')
        if not isinstance(rid,str) or not rid or rid in seen:
            raise ValueError('Ground labels require unique supplied region IDs')
        seen.add(rid);box=_box(label)
        explicit.append(dict(id=rid,domain=text.strip(),bbox=box))
        if box is None:unresolved.append(dict(label_id=rid,reason='ground_label_bbox_unavailable'))
    eligible=[c for c in grounds if _box(c) is not None]
    for c in grounds:
        if _box(c) is None:unresolved.append(dict(component_id=c['id'],reason='ground_component_bbox_unavailable'))
    duplicate=set()
    for i,a in enumerate(explicit):
        if a['bbox'] is None:continue
        for b in explicit[i+1:]:
            if b['bbox'] is None:continue
            x,y=a['bbox'],b['bbox'];area=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
            if area/min((x[2]-x[0])*(x[3]-x[1]),(y[2]-y[0])*(y[3]-y[1]))>.7:
                duplicate.update((a['id'],b['id']))
    for rid in sorted(duplicate):unresolved.append(dict(label_id=rid,reason='overlapping_physical_ground_labels'))
    regions=sorted((r for r in explicit if r['bbox'] is not None and r['id'] not in duplicate),key=lambda r:r['id'])
    diagonals=[math.hypot(b[2]-b[0],b[3]-b[1]) for b in map(_box,eligible)]
    scale=max(1.,statistics.median(diagonals)) if diagonals else 1.
    costs=[];forbidden=1e6
    for region in regions:
        a=region['bbox'];row=[]
        for component in eligible:
            b=_box(component)
            edge=math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))
            center=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)
            distance=(edge+.25*center)/scale
            row.append(distance if distance<MAX_DISTANCE else forbidden)
        costs.append(row+[MAX_DISTANCE]*len(regions))
    selected,optimum=_minimum_assignment(costs)
    for i,j in enumerate(selected):
        region=regions[i]
        if j>=len(eligible):
            unresolved.append(dict(label_id=region['id'],reason='ground_label_no_close_unique_owner'));continue
        alternative=[r[:] for r in costs];alternative[i][j]=forbidden
        _,other=_minimum_assignment(alternative);gap=other-optimum
        if gap<=AMBIGUITY_MARGIN:
            unresolved.append(dict(label_id=region['id'],reason='ground_label_ownership_ambiguous',alternative_optimum_gap=gap));continue
        cid=eligible[j]['id'];domains[cid]=region['domain']
        assignment=dict(component_id=cid,label_id=region['id'],domain=region['domain'],bbox=region['bbox'],
            evidence='supplied_label_bbox_joint_geometry',cost=costs[i][j],alternative_optimum_gap=gap)
        assignments.append(assignment)
    grouped={}
    for component in grounds:
        cid=component['id'];kind=_symbol_kind(component);domain=domains[cid]
        if _box(component) is None:key=('unusable',cid)
        elif not explicit:key=('legacy_unobserved_domain',kind)
        elif domain is None:
            key=('unassigned',cid)
            unresolved.append(dict(component_id=cid,reason='explicit_ground_domain_unobserved'))
        else:key=('observed_domain',kind,domain)
        grouped.setdefault(key,[]).append(cid)
    for key,group in grouped.items():
        evidence.append(dict(component_ids=group,source=key[0],symbol_kind=key[1] if key[0] in ('legacy_unobserved_domain','observed_domain') else None,
            domain=key[2] if key[0]=='observed_domain' else None,
            domain_observed=key[0]=='observed_domain'))
    return dict(policy=POLICY,groups=sorted(grouped.values()),domains=domains,assignments=assignments,
        evidence=evidence,unresolved=unresolved,explicit_domain_labels_present=bool(explicit),
        requires_shape_validation=True,verified=False,
        settings=dict(max_distance=MAX_DISTANCE,ambiguity_margin=AMBIGUITY_MARGIN,
            normalization='median supplied ground bbox diagonal'))
