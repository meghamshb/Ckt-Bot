"""Bind current arrows to occupied wire sections, not component proximity.

A geometric cut of a wire tree exposes its signed terminal-current sum by
KCL. No voltage/current value is solved here. Wire loops and external return
paths without a determinate cut remain explicitly unbound.
"""
import math
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def partition_revision(candidate):
    """Identity of terminal ownership/partition, independent of net numbering."""
    groups={}
    for pin in candidate.get('pins',[]):
        if pin.get('net_id'):
            groups.setdefault(pin['net_id'],[]).append(('pin',pin['id'],pin['component_id']))
    for port in candidate.get('external_ports',[]):
        if port.get('net_id') and port.get('state')=='observed_external_terminal':
            groups.setdefault(port['net_id'],[]).append(('boundary',port['id'],attachment_revision(port,candidate)))
    partition=sorted(sorted(members) for members in groups.values())
    return hashlib.sha256(json.dumps(partition,separators=(',',':')).encode()).hexdigest()


def attachment_revision(port,candidate=None):
    """Bind physical attachment provenance, invariant to an owned-net rename."""
    def normalize(value,key=None):
        if isinstance(value,dict):return {k:normalize(v,k) for k,v in value.items() if k not in ('source_net_id','net_id_scope')}
        if isinstance(value,list):return [normalize(v) for v in value]
        return '@owned-net' if key=='net_id' and value==port.get('net_id') else value
    payload={'marker_center':port.get('circle_center_px'),'marker_radius':port.get('circle_radius_px'),
        'marker_shape':port.get('terminal_shape'),'attachment':normalize(port.get('electrical_attachment'))}
    if candidate is not None:
        record=port.get('electrical_attachment') or {};fragments={f['id']:f for f in candidate.get('wire_fragments',[])}
        ids=sorted({w.get('fragment_id') for w in record.get('witnesses',[]) if isinstance(w.get('fragment_id'),str)})
        payload['actual_fragments']=[{'id':fid,'polyline_px':fragments.get(fid,{}).get('polyline_px'),
            'incidence_complete':fragments.get(fid,{}).get('incidence_complete'),
            'owners':sorted('@owned-net' if n['id']==port.get('net_id') else n['id']
                for n in candidate.get('nets',[]) if fid in n.get('fragment_ids',[]))} for fid in ids]
        source=candidate.get('image',{}).get('path')
        payload['actual_source_sha256']=hashlib.sha256(Path(source).read_bytes()).hexdigest() if source and Path(source).is_file() else None
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _path_pixels(path,shape):
    """Exact declared native cell centres, never a neighbourhood sample."""
    try:points=np.asarray(path,dtype=float)
    except (TypeError,ValueError):return None
    if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all():return None
    pixels=np.floor(points).astype(int)
    if not np.allclose(points,pixels+.5,rtol=0,atol=1e-8):return None
    if np.any(pixels<0) or np.any(pixels[:,0]>=shape[1]) or np.any(pixels[:,1]>=shape[0]):return None
    if len(pixels)>1 and np.any(np.max(np.abs(np.diff(pixels,axis=0)),axis=1)>1):return None
    return pixels


def _fragment_contains_cell(polyline,pixel):
    """Exact segment/unit-cell intersection; no distance-based association."""
    lower=np.asarray(pixel,dtype=float);upper=lower+1
    for first,last in zip(polyline,polyline[1:]):
        a=np.asarray(first,dtype=float);delta=np.asarray(last,dtype=float)-a;lo,hi=0.,1.
        for axis in (0,1):
            if abs(delta[axis])<1e-12:
                if not lower[axis]-1e-9<=a[axis]<=upper[axis]+1e-9:break
            else:
                bounds=(lower[axis]-a[axis])/delta[axis],(upper[axis]-a[axis])/delta[axis]
                lo=max(lo,min(bounds));hi=min(hi,max(bounds))
        else:
            if lo<=hi+1e-9:return True
    return False


def _boundary_root(candidate,port,mask,labels,roots,cut_extent):
    """All producer-proved attachment cells must survive on one cut side."""
    record=port.get('electrical_attachment')
    if not isinstance(record,dict) or record.get('schema')!='kbot.external-attachment/1':return None
    frame=record.get('frame',{});h,w=mask.shape
    if frame.get('width')!=w or frame.get('height')!=h or frame.get('coordinate_convention')!='pixel_centers':return None
    if record.get('net_id')!=port.get('net_id'):return None
    source=candidate.get('image',{}).get('path')
    if not source or not Path(source).is_file():return None
    if record.get('source_image_sha256')!=hashlib.sha256(Path(source).read_bytes()).hexdigest():return None
    if record.get('mask_pixels_sha256')!=hashlib.sha256(mask.tobytes()).hexdigest():return None
    image=cv2.imread(str(source),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=mask.shape:return None
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);ink=gray<=float(np.percentile(gray,90))-48
    marker=record.get('marker',{})
    if marker.get('center_px')!=port.get('circle_center_px') or marker.get('radius_px')!=port.get('circle_radius_px') or marker.get('shape')!=port.get('terminal_shape'):return None
    if marker.get('shape') not in ('open_round','filled_round'):return None
    try:
        center=np.asarray(marker['center_px'],float);radius=float(marker['radius_px']);uncertainty=float(marker['uncertainty_px'])
    except (KeyError,TypeError,ValueError):return None
    if center.shape!=(2,) or not np.isfinite(center).all() or not math.isfinite(radius+uncertainty) or radius<=0 or uncertainty<0:return None
    yy,xx=np.nonzero(cut_extent)
    if np.any(np.hypot(xx+.5-center[0],yy+.5-center[1])<=radius+uncertainty+math.sqrt(.5)):return None
    fragments={f['id']:f for f in candidate.get('wire_fragments',[])};owners={}
    for net in candidate.get('nets',[]):
        for fid in net.get('fragment_ids',[]):owners.setdefault(fid,set()).add(net['id'])
    witnesses=record.get('witnesses');side_roots=set()
    if not isinstance(witnesses,list) or not witnesses:return None
    for witness in witnesses:
        fid=witness.get('fragment_id');fragment=fragments.get(fid)
        if fragment is None or fragment.get('incidence_complete') is not True or owners.get(fid)!={port['net_id']} or witness.get('net_id')!=port['net_id']:return None
        identity={'id':fid,'polyline_px':fragment['polyline_px']}
        if witness.get('fragment_sha256')!=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest():return None
        if not witness.get('owner_provenance'):return None
        lead=_path_pixels(witness.get('occupied_lead_path_px'),mask.shape)
        path=_path_pixels(witness.get('source_continuation_path_px'),mask.shape)
        if lead is None or path is None or not np.array_equal(lead[0],path[0]):return None
        if not all(_fragment_contains_cell(fragment['polyline_px'],pixel) for pixel in lead):return None
        if not np.all(mask[lead[:,1],lead[:,0]]&ink[lead[:,1],lead[:,0]]):return None
        if not np.all(ink[path[:,1],path[:,0]]):return None
        if abs(float(np.linalg.norm(path[-1]+.5-center))-radius)>uncertainty+math.sqrt(.5):return None
        if cut_extent[lead[:,1],lead[:,0]].any() or cut_extent[path[:,1],path[:,0]].any():return None
        occupied=labels[lead[:,1],lead[:,0]]
        if not np.all(occupied>0):return None
        side_roots.update(int(roots[label]) for label in occupied)
    return next(iter(side_roots)) if len(side_roots)==1 else None


def _sample(labels, point):
    x,y=[int(v) for v in point];h,w=labels.shape
    values=set(labels[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)].ravel())-{0}
    return int(next(iter(values))) if len(values)==1 else None


def _roots_for_bridges(candidate, labels):
    roots=np.arange(int(labels.max())+1)
    def union(a,b):
        if a is not None and b is not None:
            x,y=roots[a],roots[b];roots[roots==y]=x
    for event in candidate.get('intersections',[]):
        if event['state']!='separate':continue
        points=event.get('arm_points_px',[])
        for a,b in event.get('local_arm_partitions',[]):
            if a>=len(points) or b>=len(points):return None
            u,v=_sample(labels,points[a]),_sample(labels,points[b])
            if u is None or v is None:return None
            union(u,v)
    return roots


def _section_terms(candidate, mask, base_labels, width, point, tangent, net_id, *, include_proof=False):
    h,w=mask.shape;px,py=point;x,y=int(px),int(py)
    radius=max(1.,float(width[y,x]));margin=math.ceil(3*radius+2)
    left,top=max(0,x-margin),max(0,y-margin);right,bottom=min(w,x+margin+1),min(h,y+margin+1)
    yy,xx=np.mgrid[top:bottom,left:right];delta=np.stack((xx+.5-px,yy+.5-py),axis=-1);normal=np.array([-tangent[1],tangent[0]])
    cut_geometry=(abs(delta@tangent)<=max(1.,radius))&(abs(delta@normal)<=2.5*radius+1)
    cut=cut_geometry&(base_labels[top:bottom,left:right]==base_labels[y,x])
    cut_extent=np.zeros_like(mask,dtype=bool);cut_extent[top:bottom,left:right]=cut_geometry
    separated=mask.copy();separated[top:bottom,left:right][cut]=0
    _,labels=cv2.connectedComponents(separated,8);roots=_roots_for_bridges(candidate,labels)
    if roots is None:return None
    behind=_sample(labels,np.array(point)-margin*tangent);ahead=_sample(labels,np.array(point)+margin*tangent)
    if behind is None or ahead is None or roots[behind]==roots[ahead]:return None
    sides=[{'name':'behind','root':roots[behind],'coefficient':-1,'pins':[],'boundaries':[]},
           {'name':'ahead','root':roots[ahead],'coefficient':1,'pins':[],'boundaries':[]}]
    types={c['id']:c['type'] for c in candidate['components']}
    for pin in candidate['pins']:
        if pin.get('net_id')!=net_id:continue
        if not pin.get('anchor_px'):return None
        part=_sample(labels,pin['anchor_px'])
        if part is None:return None
        if not any(roots[part]==side['root'] for side in sides):return None
        for side in sides:
            if roots[part]==side['root']:
                if types[pin['component_id']]=='ground':side['boundaries'].append(pin['id'])
                else:side['pins'].append(pin['id'])
    boundary_attachments=[]
    for port in candidate.get('external_ports',[]):
        if port.get('state')!='observed_external_terminal' or port.get('net_id')!=net_id:continue
        root=_boundary_root(candidate,port,mask,labels,roots,cut_extent)
        if root is None:return None
        if not any(root==side['root'] for side in sides):return None
        for side in sides:
            if root==side['root']:
                side['boundaries'].append(port['id'])
                boundary_attachments.append({'port_id':port['id'],'attachment_revision':attachment_revision(port),'side':side['name']})
    usable=[s for s in sides if not s['boundaries'] and s['pins']]
    if not usable:return None
    # Fewer terminal terms is an equivalent, more compact KCL statement.
    side=min(usable,key=lambda s:(len(s['pins']),s['coefficient']!=1))
    terms=[{'pin_id':p,'coefficient':side['coefficient']} for p in sorted(side['pins'])]
    if not include_proof:return terms
    dependencies=[event['id'] for event in candidate.get('intersections',[])
        if any(arm.get('net_id')==net_id for arm in event.get('arms',[]))]
    proof={'graph_revision':partition_revision(candidate),'source_net_id':net_id,
        'mask_pixels_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
        'point_px':np.round(point,3).tolist(),'tangent':np.round(tangent,5).tolist(),
        'selected_side':side['name'],'dependent_event_ids':sorted(dependencies),
        'closure_proved':True,'terminal_current_terms':terms,'boundary_attachments':boundary_attachments}
    for item in sides:
        proof[item['name']]={'pin_ids':sorted(item['pins']),
            'boundary_ids':sorted(item['boundaries'])}
    return {'terminal_current_terms':terms,'cut_proof':proof}


def observe_current_references(candidate):
    evidence=candidate.get('evidence',{})
    if not isinstance(evidence,dict):return candidate.get('reference_arrows',[])
    from .mesh_references import observe_mesh_references
    mesh=observe_mesh_references(candidate)
    candidate['mesh_reference_evidence']=mesh['evidence']
    curved=mesh['references']
    objects=[o for o in evidence.get('stroke_scene',{}).get('reference_arrow_objects',[]) if o.get('direction_unit')]
    # A curved object owns its own arrowhead; it is not a second straight mark.
    objects=[o for o in objects if not any(m['bbox_px'][0]<=o['bbox_px'][0] and m['bbox_px'][1]<=o['bbox_px'][1] and m['bbox_px'][2]>=o['bbox_px'][2] and m['bbox_px'][3]>=o['bbox_px'][3] for m in curved)]
    mask=(cv2.imread(str(candidate['mask_path']),cv2.IMREAD_GRAYSCALE)>0).astype(np.uint8)
    _,base_labels=cv2.connectedComponents(mask,8)
    width=cv2.distanceTransform(mask,cv2.DIST_L2,3)
    fragment_net={f:n['id'] for n in candidate['nets'] if n['pins'] for f in n.get('fragment_ids',[])}
    fragments=[]
    for fragment in candidate['wire_fragments']:
        if fragment['id'] not in fragment_net:continue
        points=np.array(fragment['polyline_px'],np.float32)
        if len(points)<2:continue
        local=[float(width[int(y),int(x)]) for x,y in points if 0<=int(y)<mask.shape[0] and 0<=int(x)<mask.shape[1]]
        radius=max(1.,float(np.median(local)))
        poly=cv2.approxPolyDP(points.reshape(-1,1,2),1.25*radius,False).reshape(-1,2)
        for a,b in zip(poly,poly[1:]):fragments.append((fragment['id'],fragment_net[fragment['id']],a.astype(float),b.astype(float),radius))
    observed=[]
    for obj in objects:
        direction=np.array(obj['direction_unit']);box=obj['bbox_px'];center=(np.array(box[:2])+np.array(box[2:]))/2
        length=math.hypot(box[2]-box[0],box[3]-box[1]);proposals=[]
        for fid,nid,a,b,radius in fragments:
            v=b-a;span=float(np.linalg.norm(v));margin=math.ceil(3*radius+2)
            if span<=2*margin:continue
            axis=v/span;alignment=float(axis@direction)
            if abs(alignment)<.9:continue
            t=float((center-a)@axis)
            if not 0<=t<=span:continue
            point=a+axis*np.clip(t,margin,span-margin)
            distance=float(np.linalg.norm(point-center))
            if distance>.65*length:continue
            x,y=[int(v) for v in point]
            if not mask[y,x]:
                yy,xx=np.nonzero(mask[max(0,y-2):y+3,max(0,x-2):x+3]);pixels=np.column_stack((xx+max(0,x-2)+.5,yy+max(0,y-2)+.5))
                if not len(pixels):continue
                point=pixels[np.argmin(np.sum((pixels-point)**2,axis=1))]
            tangent=axis*(1 if alignment>0 else -1)
            if any(e.get('cut_disk_px') and np.linalg.norm(point-np.array(e['cut_disk_px']['center']))<=e['cut_disk_px']['outer_radius'] for e in candidate.get('intersections',[])):continue
            section=_section_terms(candidate,mask,base_labels,width,point,tangent,nid,include_proof=True)
            if section is None:continue
            proposals.append((distance,{'fragment_id':fid,'net_id':nid,'point_px':np.round(point,3).tolist(),'wire_direction_unit':np.round(tangent,5).tolist(),**section}))
        proposals.sort(key=lambda p:p[0]);choices={};seen=set()
        for distance,proposal in proposals:
            key=tuple((t['pin_id'],t['coefficient']) for t in proposal['terminal_current_terms'])
            if key in seen:continue
            seen.add(key);choices['w'+str(len(choices)+1)]=proposal
        observed.append({'id':'ar'+str(len(observed)+1),'object_id':obj.get('id'),'bbox_px':box,'direction_unit':obj['direction_unit'],'choices':choices,
            'kind':'wire_current','evidence':'native_pointed_arrow_and_occupied_wire_cut_KCL'})
    for reference in curved:
        observed.append({'id':'ar'+str(len(observed)+1),**reference})
    return observed
