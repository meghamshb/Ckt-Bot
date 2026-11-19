"""Canonical final-net references and revision-bound measurement evidence."""
from __future__ import annotations

import copy
import hashlib
from PIL import Image


def mask_pixels_sha256(candidate):
    """Match the observer's binary-u8 pixel identity, not the PNG encoding."""
    with Image.open(candidate['mask_path']) as image:
        bits=image.convert('L').point(lambda value: int(value>0))
        return hashlib.sha256(bits.tobytes()).hexdigest()


def validate_wire_cut(candidate, choice, terms, mapping, *, changed_events=(), mask_hash=None):
    """Keep unaffected cuts; never carry a closed-side proof across a relevant union."""
    source=choice.get('net_id')
    members=[nid for nid,target in mapping.items() if target==mapping.get(source)]
    relevant_union=len(members)>1
    proof=choice.get('cut_proof')
    result={'source_net_id':source,'final_net_id':mapping.get(source),
        'source_equivalence_class':sorted(members),'relevant_union':relevant_union}
    if proof is None:
        return {**result,'status':'invalidated' if relevant_union else 'legacy_unverified',
            'reason':'current_cut_proof_unavailable','blocking':relevant_union}
    from .current_references import partition_revision
    reasons=[]
    if proof.get('graph_revision')!=partition_revision(candidate):reasons.append('source_partition_revision_mismatch')
    if proof.get('source_net_id')!=source or source not in mapping:reasons.append('source_net_mismatch')
    if proof.get('closure_proved') is not True:reasons.append('closed_side_not_proved')
    if proof.get('terminal_current_terms')!=terms:reasons.append('signed_cut_terms_changed')
    if mask_hash is None or proof.get('mask_pixels_sha256')!=mask_hash:reasons.append('conductor_mask_identity_mismatch')
    selected=proof.get('selected_side');side=proof.get(selected,{}) if selected in ('behind','ahead') else {}
    behind=set(proof.get('behind',{}).get('pin_ids',[]));ahead=set(proof.get('ahead',{}).get('pin_ids',[]))
    if behind&ahead:reasons.append('cut_sides_overlap')
    expected=sorted((pid,1 if selected=='ahead' else -1) for pid in side.get('pin_ids',[]))
    actual=sorted((t['pin_id'],t['coefficient']) for t in terms)
    if not expected or side.get('boundary_ids') or actual!=expected:reasons.append('selected_side_is_not_the_recorded_closed_side')
    owned={p['id'] for p in candidate['pins'] if p.get('net_id')==source}
    if not (behind|ahead)<=owned:reasons.append('cut_side_uses_foreign_terminal')
    if relevant_union:reasons.append('source_net_equivalence_changed')
    if set(proof.get('dependent_event_ids',[]))&set(changed_events):reasons.append('crossing_dependency_changed')
    return {**result,'status':'invalidated' if reasons else 'validated',
        'reason':'current_cut_proof_invalidated' if reasons else 'unchanged_physical_cut',
        'details':reasons,'blocking':bool(reasons),'source_graph_revision':proof.get('graph_revision')}


def canonicalize_references(circuit, mapping):
    """All exported net IDs declare final active/unattached scope; provenance stays original."""
    active={n['id'] for n in circuit['nets']}
    unattached={n['id'] for n in circuit.get('unattached_wire_candidates',[])}
    known=active|unattached
    def remap(row,field='net_id'):
        old=row.get(field)
        if old is None:return
        if old not in mapping and old not in known:raise ValueError('Final observation references an unknown net: '+str(old))
        row.setdefault('source_'+field,old)
        row[field]=mapping.get(old,old)
        row[field+'_scope']='active' if row[field] in active else 'unattached'
    for field in ('external_ports','measurement_terminals'):
        for row in circuit.get(field,[]):
            remap(row)
            attachment=row.get('electrical_attachment')
            if isinstance(attachment,dict):
                remap(attachment)
                for witness in attachment.get('witnesses',[]):remap(witness)
    for event in circuit.get('intersections',[]):
        for arm in event.get('arms',[]):remap(arm)
        if 'bound_arm_regions' in event:event['raster_region_namespace']='source_geometry'
    for arrow in circuit.get('reference_arrows',[]):
        for choice in arrow.get('choices',{}).values():remap(choice)
    for row in circuit.get('arrows',[]):remap(row)
    for row in circuit.get('voltage_references',[]):
        for field in ('positive_net_id','negative_net_id'):remap(row,field)
    circuit['net_id_mapping']=dict(mapping)
    circuit['reference_namespace']='final_net_ids_with_original_source_ids'


def geometry_uncertainties(candidate):
    """Propagate conditioning uncertainty without guessing that an empty map is harmless."""
    scene=candidate.get('evidence',{}).get('stroke_scene',{})
    by_region={}
    for fragment in candidate.get('wire_fragments',[]):
        for key in ('ink_region','region_id'):
            if key in fragment:by_region.setdefault(fragment[key],set()).add(fragment['id'])
    result=[]
    for row in scene.get('uncertain_overlaps',[]):
        item={'id':row['id']+':unresolved','reason':'ambiguous_conductor_annotation_overlap',
            'blocking':True,'affects_equivalence':True,'object_id':row.get('object_id'),
            'source_extent_px':row.get('source_extent_px'),'geometry_observation':copy.deepcopy(row)}
        ids=set(row.get('affected_fragment_ids',[]))
        for region in row.get('after_regions',[]):ids.update(by_region.get(region,()))
        if ids:item['fragment_ids']=sorted(ids)
        components=row.get('affected_component_ids',[])
        if components:item['component_ids']=sorted(set(components))
        if row.get('component_bearing') is False and row.get('irrelevance_evidence'):
            item.update(blocking=False,scope='observed_annotation_only',scope_evidence=row['irrelevance_evidence'])
        result.append(item)
    return result
