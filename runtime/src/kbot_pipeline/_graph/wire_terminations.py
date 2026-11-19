"""Distinguish source-observed round terminals from unexplained wire ends.

This module only classifies existing endpoints. It never fills a mask gap or
connects nearby nets. Unexplained ends remain explicit topology blockers.
"""
import math
import hashlib
import json
from pathlib import Path
from collections import deque
import cv2
import numpy as np
from PIL import Image


def classify_terminations(source, graph):
    rgb=np.asarray(Image.open(source).convert('RGB'))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    paper=float(np.percentile(gray,90))
    ink=(gray<=paper-48).astype(np.uint8)
    contours,hierarchy=cv2.findContours(ink,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    circles=[]
    forbidden=np.zeros_like(ink)
    for component in graph['components']:
        a,b,c,d=map(int,component['bbox']);forbidden[b:d,a:c]=1
    widths=cv2.distanceTransform(ink,cv2.DIST_L2,3)
    stroke=max(1.,float(np.median(widths[ink!=0])))
    if hierarchy is not None:
        for i,contour in enumerate(contours):
            if hierarchy[0,i,3]<0:continue  # Need a genuinely enclosed white hole.
            area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
            if not 12<=area<=500 or perimeter==0 or 4*math.pi*area/perimeter**2<.72:continue
            x,y,w,h=cv2.boundingRect(contour)
            if not .75<=w/h<=1.33:continue
            center=np.array([x+(w-1)/2,y+(h-1)/2]);radius=(w+h)/4
            if any(c['bbox'][0]-2<=center[0]<=c['bbox'][2]+2 and c['bbox'][1]-2<=center[1]<=c['bbox'][3]+2 for c in graph['components']):continue
            if _one_exit(ink,center,radius):circles.append((center,radius,'open_round'))
    for center,radius in _filled_circles(ink):
        if any(c['bbox'][0]-2<=center[0]<=c['bbox'][2]+2 and c['bbox'][1]-2<=center[1]<=c['bbox'][3]+2 for c in graph['components']):continue
        if _one_exit(ink,center,radius):circles.append((center,radius,'filled_round'))
    fragment_net={f:n['id'] for n in graph['nets'] for f in n.get('fragment_ids',[])}
    occupied={p['net_id'] for p in graph['pins'] if p.get('net_id')}
    # Measurement circles are physical electrodes even when a short ground
    # lead was already owned by a component and emitted no open-end candidate.
    graph['measurement_terminals']=[]
    conductor=np.asarray(Image.open(graph['mask_path']).convert('L'))>0
    anchors=[(p['anchor_px'],p['net_id']) for p in graph['pins'] if p.get('state')=='supported' and p.get('anchor_px') and p.get('net_id')]
    circle_owners={}
    for center,radius,kind in circles:
        witnesses=[(point,net) for point,net in anchors if _source_terminal_lead(ink,forbidden,np.asarray(point),center,radius,stroke,graph.get('intersections',()))]
        fragment_witnesses=_fragment_terminal_contacts(graph,fragment_net,occupied,conductor,
            ink,forbidden,center,radius,stroke)
        witnesses.extend((row['point_px'],row['net_id']) for row in fragment_witnesses)
        owners={net for _,net in witnesses}
        circle_owners[tuple(center)]=owners
        if len(owners)==1:
            owner=next(iter(owners))
            attachment=_electrical_attachment(source,graph,conductor,center,radius,kind,stroke,owner,fragment_witnesses)
            graph['measurement_terminals'].append({'circle_center_px':center.tolist(),'circle_radius_px':radius,'terminal_shape':kind,'net_id':owner,'source_contact_witnesses':[point for point,_ in witnesses],'fragment_contact_witnesses':fragment_witnesses,
                'electrical_attachment':attachment,'electrical_attachment_status':'observed' if attachment else 'missing_occupied_fragment_witness',
                'evidence':'native_one_exit_circle_with_continuous_lead_to_unique_existing_net'})
    issues=[]
    internal_caps=[]
    for endpoint in graph.get('external_ports',[]):
        nid=fragment_net.get(endpoint['fragment_id']);endpoint['net_id']=nid
        if nid not in occupied:continue
        point=np.array(endpoint['point_px']);matches=[]
        for center,radius,kind in circles:
            owners=circle_owners[tuple(center)]
            if owners!={nid}:continue
            if not _source_terminal_lead(ink,forbidden,point,center,radius,stroke,graph.get('intersections',())):continue
            matches.append((center,radius,kind))
        matches.sort(key=lambda item:float(np.linalg.norm(point-item[0])))
        if len(matches)>1 and np.linalg.norm(point-matches[1][0])-np.linalg.norm(point-matches[0][0])>2*stroke:matches=matches[:1]
        if len(matches)==1:
            center,radius,kind=matches[0]
            endpoint.update(state='observed_external_terminal',circle_center_px=center.tolist(),
                circle_radius_px=radius,terminal_shape=kind,evidence='native_round_terminal_one_exit_and_continuous_source_lead')
            terminal=next(t for t in graph['measurement_terminals'] if t['circle_center_px']==center.tolist() and t['net_id']==nid)
            endpoint['electrical_attachment']=terminal['electrical_attachment']
            endpoint['electrical_attachment_status']=terminal['electrical_attachment_status']
        else:
            proof=_producer_cut_cap(graph,endpoint,ink,fragment_net,circles)
            if proof is not None:
                internal_caps.append({'historical_endpoint':dict(endpoint),**proof})
                continue
            endpoint['state']='unexplained_wire_termination'
            issues.append({'id':endpoint['id']+':unresolved','reason':'unexplained_wire_termination',
                'fragment_ids':[endpoint['fragment_id']],'point_px':endpoint['point_px'],
                'blocking':True,'affects_equivalence':True})
    if internal_caps:
        ids={row['historical_endpoint']['id'] for row in internal_caps}
        graph['external_ports']=[row for row in graph['external_ports'] if row['id'] not in ids]
        graph.setdefault('internal_wire_caps',[]).extend(internal_caps)
    for index,terminal in enumerate(graph['measurement_terminals']):
        if any(port.get('circle_center_px')==terminal['circle_center_px'] for port in graph['external_ports']):continue
        graph['external_ports'].append({'id':'native-terminal-'+str(index+1),'point_px':terminal['circle_center_px'],'circle_center_px':terminal['circle_center_px'],'circle_radius_px':terminal['circle_radius_px'],'terminal_shape':terminal['terminal_shape'],'net_id':terminal['net_id'],'state':'observed_external_terminal','evidence':terminal['evidence'],
            'electrical_attachment':terminal['electrical_attachment'],'electrical_attachment_status':terminal['electrical_attachment_status']})
    return issues


def _electrical_attachment(source,graph,conductor,center,radius,kind,stroke,owner,witnesses):
    """Package existing occupied-fragment/source proofs; marker is display geometry."""
    fragments={f['id']:f for f in graph.get('wire_fragments',[])}
    rows=[]
    for witness in witnesses:
        fragment=fragments.get(witness['fragment_id'])
        owners={n['id'] for n in graph.get('nets',[]) if n.get('pins') and witness['fragment_id'] in n.get('fragment_ids',[])}
        if (fragment is None or witness['net_id']!=owner
                or owners!={owner} or not witness.get('occupied_lead_path_px') or not witness.get('source_continuation_path_px')
                or witness['occupied_lead_path_px'][0]!=witness['source_continuation_path_px'][0]):
            return None
        identity={'id':fragment['id'],'polyline_px':fragment['polyline_px']}
        rows.append({'fragment_id':fragment['id'],'net_id':owner,
            'fragment_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'occupied_lead_path_px':witness['occupied_lead_path_px'],
            'source_continuation_path_px':witness['source_continuation_path_px'],
            'owner_provenance':'unique_existing_component_bearing_fragment_and_native_one_exit_marker_source_path'})
    if not rows:return None
    h,w=conductor.shape
    return {'schema':'kbot.external-attachment/1','net_id':owner,
        'frame':{'coordinate_convention':'pixel_centers','width':w,'height':h},
        'source_image_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'mask_pixels_sha256':hashlib.sha256(conductor.astype(np.uint8).tobytes()).hexdigest(),
        'marker':{'center_px':center.tolist(),'radius_px':radius,'uncertainty_px':2*stroke,'shape':kind},
        'witnesses':rows}


def _producer_cut_cap(graph,endpoint,ink,fragment_net,circles=()):
    """Prove a skeleton cap was created by one recorded overpass cut.

    Counterfactual masks are diagnostic copies only. No conductor pixel, pin,
    fragment or net is added. Missing source/cut/arm provenance abstains.
    """
    from ._topology_geometry import _thin,_adjacency
    point=np.asarray(endpoint['point_px'],float);pixel=np.floor(point).astype(int);x,y=map(int,pixel)
    if not np.array_equal(point,pixel+.5):return None
    local=[f for f in graph.get('wire_fragments',[]) if f.get('polyline_px') and
           (f['polyline_px'][0]==point.tolist() or f['polyline_px'][-1]==point.tolist())]
    if len(local)!=1:return None
    fragment=local[0];nid=fragment_net.get(fragment['id'])
    if nid is None or nid!=endpoint.get('net_id'):return None
    parent=Path(graph['mask_path']).parent
    paths={name:parent/name for name in ('wire-input-mask.png','wire-after-body-mask.png','crossing-cut-mask.png',
                                      'component-body-exclusion-mask.png','conductor-mask.png')}
    if any(not path.is_file() for path in paths.values()):return None
    arrays={name:cv2.imread(str(path),0) for name,path in paths.items()}
    if any(value is None or value.shape!=ink.shape for value in arrays.values()):return None
    pre=(arrays['wire-after-body-mask.png']>0).astype(np.uint8)
    sam=arrays['wire-input-mask.png']>0
    post=(arrays['conductor-mask.png']>0).astype(np.uint8)
    all_cut=arrays['crossing-cut-mask.png']>0
    body=arrays['component-body-exclusion-mask.png']>0
    if not (0<=y<ink.shape[0] and 0<=x<ink.shape[1] and post[y,x] and sam[y,x] and ink[y,x]):return None
    if not np.array_equal(post,(pre&~all_cut&~body).astype(np.uint8)):return None
    _,regions=cv2.connectedComponents(post,8);region=int(regions[y,x])
    skeleton,done,_=_thin(post)
    if not done:return None
    adjacency=_adjacency(skeleton,regions)
    if len(adjacency.get((x,y),[]))!=1:return None
    yy,xx=np.indices(ink.shape);solutions=[]
    for event in graph.get('intersections',[]):
        disk=event.get('cut_disk_px');box=event.get('proof_box_px')
        arms=event.get('arm_points_px');owners=event.get('bound_arm_regions')
        if (event.get('state')!='separate' or not disk or not box or not arms
                or not owners or len(arms)!=len(owners) or len(set(owners))!=len(owners)):continue
        if not all(k in disk for k in ('center','inner_radius','stroke_radius')):continue
        stroke=float(disk['stroke_radius'])
        if any(np.linalg.norm(point-center)<=radius+2*stroke for center,radius,_ in circles):continue
        left,top,right,bottom=map(int,box)
        if not (0<=left<right<=ink.shape[1] and 0<=top<bottom<=ink.shape[0]):continue
        window=(xx>=left)&(xx<right)&(yy>=top)&(yy<bottom)
        # Recorded disk coordinates denote pixel centres; reproduce its exact
        # producer footprint, then require the persisted cut to contain it.
        center=np.asarray(disk['center'],float)
        event_cut=(np.hypot(xx+.5-center[0],yy+.5-center[1])<=float(disk['inner_radius']))&window
        if np.any(event_cut&~all_cut) or np.any(event_cut&body):continue
        targets=[]
        for index,(arm,owner) in enumerate(zip(arms,owners)):
            ax,ay=np.floor(arm).astype(int)
            if not (0<=ax<ink.shape[1] and 0<=ay<ink.shape[0]):break
            if int(regions[ay,ax])!=int(owner):break
            if int(owner)==region and post[ay,ax] and ink[ay,ax]:targets.append((index,(int(ax),int(ay))))
        else:
            if len(targets)!=1:continue
            index,target=targets[0]
            allowed=(post>0)&sam&(ink>0)&window&(regions==region)&~body
            if not allowed[y,x]:continue
            queue=deque([(x,y)]);previous={(x,y):None}
            while queue and target not in previous:
                px,py=queue.popleft()
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        q=(px+dx,py+dy);qx,qy=q
                        if not(dx or dy) or q in previous or not(0<=qx<ink.shape[1] and 0<=qy<ink.shape[0]):continue
                        if allowed[qy,qx]:previous[q]=(px,py);queue.append(q)
            if target not in previous:continue
            arm_path=[];q=target
            while q is not None:arm_path.append(list(q));q=previous[q]
            counterfactual=post|(pre&event_cut.astype(np.uint8))
            cs,converged,_=_thin(counterfactual)
            if not converged:continue
            _,clabels=cv2.connectedComponents(counterfactual,8);ca=_adjacency(cs,clabels)
            neighbours=ca.get((x,y),[])
            if len(neighbours)!=2:continue
            branches=[]
            for first in neighbours:
                path=[(x,y),first];hit=False;valid=True
                for _ in range((right-left)*(bottom-top)):
                    qx,qy=path[-1]
                    if not(pre[qy,qx] and sam[qy,qx] and ink[qy,qx] and not body[qy,qx]):valid=False;break
                    if event_cut[qy,qx]:hit=True;break
                    if not window[qy,qx]:break
                    options=[p for p in ca.get(path[-1],[]) if p!=path[-2]]
                    if len(options)!=1 or options[0] in path:break
                    path.append(options[0])
                branches.append({'path_px':[list(p) for p in path],'enters_recorded_cut':hit,'source_and_original_occupied':valid})
            cut_branches=[b for b in branches if b['enters_recorded_cut'] and b['source_and_original_occupied']]
            away=[b for b in branches if not b['enters_recorded_cut'] and b['source_and_original_occupied'] and len(b['path_px'])>=math.ceil(2*stroke)+1]
            if len(cut_branches)!=1 or len(away)!=1:continue
            solutions.append({'state':'observed_internal_producer_cut_cap','point_px':point.tolist(),
                'actual_endpoint_fragment_id':fragment['id'],'historical_region_representative_fragment_id':endpoint['fragment_id'],
                'net_id':nid,'event_id':event['id'],'producer_event_id':event.get('source_id'),
                'arm_namespace':'producer_arm_points_px_and_bound_arm_regions','producer_arm_index':index,
                'producer_arm_point_px':arms[index],'original_occupied_region':region,
                'occupied_source_path_to_arm_px':arm_path[::-1],'counterfactual_branches':branches,
                'cut_footprint_sha256':hashlib.sha256(event_cut.tobytes()).hexdigest(),
                'artifact_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
                'source_ink_sha256':hashlib.sha256(ink.tobytes()).hexdigest(),
                'evidence':'recorded_cut_creates_actual_local_endpoint_with_two_sided_original_source_SAM_and_unique_producer_arm',
                'mask_pixels_added':0,'net_regions_unioned':0})
    return solutions[0] if len(solutions)==1 else None


def _fragment_terminal_contacts(graph,fragment_net,occupied,conductor,ink,forbidden,center,radius,stroke):
    """Witness the lead on its existing fragment, never through another arm.

    An outward lead span must lie on one uninterrupted occupied segment of an
    already component-bearing fragment and point toward the marker. Circle-rim
    ink alone cannot establish an owner. Source ink must independently confirm
    the short continuation to the rim; this does not extend the conductor mask.
    """
    result=[];h,w=conductor.shape
    for fragment in graph.get('wire_fragments',[]):
        nid=fragment_net.get(fragment['id'])
        if nid not in occupied:continue
        for a,b in zip(fragment['polyline_px'],fragment['polyline_px'][1:]):
            a=np.asarray(a);b=np.asarray(b);length=float(np.linalg.norm(b-a))
            if not length:continue
            points=a+np.linspace(0,1,max(2,math.ceil(2*length)+1))[:,None]*(b-a)
            distances=np.linalg.norm(points-center,axis=1)
            outside=np.flatnonzero((distances>=radius+4*stroke)&(distances<=radius+6*stroke))
            if not len(outside):continue
            # Test every observed outside sample; never rank competing owners.
            for index in outside:
                point=points[index];toward=center-point;direction=(b-a)/length
                alignment=float(np.dot(direction,toward)/np.linalg.norm(toward))
                if abs(alignment)<.9:continue
                step=math.ceil(2*stroke/(length/(len(points)-1)))
                outward=index-step if alignment>0 else index+step
                if not 0<=outward<len(points):continue
                section=points[min(index,outward):max(index,outward)+1]
                pixels=np.floor(section).astype(int)
                if np.any(pixels<0) or np.any(pixels[:,0]>=w) or np.any(pixels[:,1]>=h):continue
                x,y=pixels[:,0],pixels[:,1]
                if not np.all(conductor[y,x]& (ink[y,x]>0)& (forbidden[y,x]==0)):continue
                continuation=_source_terminal_lead(ink,forbidden,point,center,radius,stroke,graph.get('intersections',()),include_proof=True)
                if continuation is None:continue
                # These are the actual already-validated fragment samples,
                # ordered from the shared source-path start toward the net.
                ordered=section if index<=outward else section[::-1]
                native=np.floor(ordered).astype(int);lead=[]
                for pixel in native:
                    physical=(pixel+.5).tolist()
                    if not lead or physical!=lead[-1]:lead.append(physical)
                result.append({'fragment_id':fragment['id'],'net_id':nid,
                    'point_px':point.tolist(),'outward_span_px':[point.tolist(),points[outward].tolist()],
                    'occupied_lead_path_px':lead,'source_continuation_path_px':continuation['path_px'],
                    'evidence':'direction_aligned_existing_occupied_lead_outside_rim_and_continuous_source_continuation'})
                break  # One proof per segment; every fragment owner is checked.
    return result


def _one_exit(ink,center,radius):
    """A round marker with one observed lead; junctions have more exits."""
    reach=math.ceil(radius+10)
    left,top=max(0,int(center[0])-reach),max(0,int(center[1])-reach)
    right,bottom=min(ink.shape[1],int(center[0])+reach+1),min(ink.shape[0],int(center[1])+reach+1)
    yy,xx=np.mgrid[top:bottom,left:right]
    distance=np.hypot(xx-center[0],yy-center[1])
    ring=(ink[top:bottom,left:right] & ((distance>=radius+2)&(distance<=radius+9))).astype(np.uint8)
    count,labels=cv2.connectedComponents(ring,connectivity=8);exits=0
    for label in range(1,count):
        radii=distance[labels==label]
        if np.any(radii<=radius+4) and np.any(radii>=radius+7):exits+=1
    return exits==1


def _filled_circles(ink):
    """Disk-shaped source ink maxima, with one-lead validation at the caller.

    Roundness is checked on radial boundaries; a thick stroke or a square
    extremum is not accepted just because it contains an inscribed disk.
    """
    distance=cv2.distanceTransform(ink,cv2.DIST_L2,3)
    if not ink.any():return []
    stroke_radius=max(1.,float(np.median(distance[ink!=0])))
    maxima=((distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))-1e-5)&(distance>=2*stroke_radius)).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(maxima,connectivity=8)
    circles=[];h,w=ink.shape
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA]>9:continue
        center=centers[i];x,y=np.rint(center).astype(int);radius=float(distance[y,x])
        if not 12<=math.pi*radius**2<=500:continue
        extent=math.ceil(2*radius+2)
        if min(x,y,w-1-x,h-1-y)<extent:continue
        boundaries=[]
        for angle in np.linspace(0,2*math.pi,24,endpoint=False):
            steps=np.arange(0,extent+.5,.5)
            xx=np.rint(center[0]+steps*math.cos(angle)).astype(int)
            yy=np.rint(center[1]+steps*math.sin(angle)).astype(int)
            absent=np.flatnonzero(ink[yy,xx]==0)
            boundaries.append(float(steps[absent[0]]) if len(absent) else float(extent))
        if sum(.75*radius<=r<=1.4*radius for r in boundaries)<18:continue
        circles.append((center,radius))
    return circles


def _source_terminal_lead(ink,forbidden,point,center,radius,stroke,intersections=(),*,include_proof=False):
    """A source-occupied outward lead, independent of missing SAM pixels.

    The corridor contains original ink only. White gaps, intervening bodies and
    adjacent disconnected strokes cannot provide the connectivity witness.
    """
    margin=math.ceil(radius+2*stroke);a=max(0,int(min(point[0],center[0]))-margin);b=max(0,int(min(point[1],center[1]))-margin);c=min(ink.shape[1],int(max(point[0],center[0]))+margin+1);d=min(ink.shape[0],int(max(point[1],center[1]))+margin+1)
    corridor=np.zeros((d-b,c-a),np.uint8);p=np.rint(point-[a,b]).astype(int);q=np.rint(center-[a,b]).astype(int)
    cv2.line(corridor,tuple(p),tuple(q),1,max(3,2*math.ceil(stroke)+1));cv2.circle(corridor,tuple(q),math.ceil(radius+stroke),1,-1)
    corridor&=ink[b:d,a:c]&(1-forbidden[b:d,a:c])
    # Raw ink connectivity cannot override a known separate/ambiguous crossing.
    # All callers use this same exclusion; no arm identity is inferred here.
    for event in intersections:
        if event.get('state')=='joined':continue
        disk=event.get('cut_disk_px')
        extent=event.get('source_extent_px') or event.get('proof_box_px') or event.get('bbox_px')
        if disk and disk.get('center') is not None and disk.get('outer_radius') is not None:
            xy=np.rint(np.asarray(disk['center'])-[a,b]).astype(int)
            cv2.circle(corridor,tuple(xy),math.ceil(disk['outer_radius']+stroke),0,-1)
        elif extent is not None:
            left,top,right,bottom=extent
            cv2.rectangle(corridor,(math.floor(left-a-stroke),math.floor(top-b-stroke)),
                (math.ceil(right-a+stroke),math.ceil(bottom-b+stroke)),0,-1)
        elif event.get('point_px') is not None:
            location=np.asarray(event['point_px'])
            arms=[arm['point_px'] for arm in event.get('observed_arms',[]) if arm.get('point_px') is not None]
            reach=max([float(event.get('cut_radius_px') or max(3,2*stroke+1))]
                +[float(np.linalg.norm(np.asarray(arm)-location)) for arm in arms])
            cv2.circle(corridor,tuple(np.rint(location-[a,b]).astype(int)),math.ceil(reach+stroke),0,-1)
        else:return None if include_proof else False  # Unknown crossing location is not clearance.
    _,labels=cv2.connectedComponents(corridor,8)
    seeds=labels[max(0,p[1]-1):p[1]+2,max(0,p[0]-1):p[0]+2];ids=np.unique(seeds);ids=ids[ids>0]
    yy,xx=np.mgrid[b:d,a:c];ring=(np.abs(np.hypot(xx-center[0],yy-center[1])-radius)<=2*stroke)&(corridor>0)
    valid=len(ids)==1 and bool(np.any(labels[ring]==ids[0]))
    if not include_proof:return valid
    if not valid:return None
    start=tuple((np.floor(point).astype(int)-[a,b]).tolist());sx,sy=start
    if not(0<=sx<c-a and 0<=sy<d-b and corridor[sy,sx]):return None
    # The saved path uses actual pixel centres throughout. It starts at the
    # occupied fragment witness, not a nearby pixel in its sampling window.
    physical_ring=(np.abs(np.hypot(xx+.5-center[0],yy+.5-center[1])-radius)<=2*stroke)&(corridor>0)
    queue=deque([start]);previous={start:None};target=None
    while queue:
        current=queue.popleft();cx,cy=current
        if physical_ring[cy,cx]:target=current;break
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                q=(cx+dx,cy+dy);qx,qy=q
                if not(dx or dy) or q in previous or not(0<=qx<c-a and 0<=qy<d-b):continue
                if corridor[qy,qx]:previous[q]=current;queue.append(q)
    if target is None:return None
    path=[]
    while target is not None:
        path.append([target[0]+a+.5,target[1]+b+.5]);target=previous[target]
    return {'path_px':path[::-1],'coordinate_convention':'pixel_centers',
        'source_rim_band_half_width_px':2*stroke}
