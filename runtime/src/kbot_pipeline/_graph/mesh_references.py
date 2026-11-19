"""Observe oriented planar mesh coordinates, without solving circuit values.

The drawing cells are an auxiliary reference view, never conductor pixels. Each
cell column must be a circulation of the observed pin/net graph and the complete
set must have its cycle rank. A dual-tree path then expresses each mesh current
as signed existing terminal currents. Curved arrow geometry selects a cell and
orientation; Qwen only reads its printed name.
"""
import math
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import sympy as sp
from PIL import Image


def _side(labels,point,normal):
    h,w=labels.shape
    for distance in np.arange(.5,math.hypot(h,w),.5):
        x,y=np.floor(point+normal*distance).astype(int)
        if not(0<=x<w and 0<=y<h):return 0
        if labels[y,x]<0:return 0
        if labels[y,x]>0:return int(labels[y,x])
    return 0


def _cells(candidate,mask):
    # Only conductor regions witnessed by supported component contacts belong
    # to the electrical drawing. Detached SAM text/arrow islands are reference
    # ink, not circuit boundaries, even if they enclose background holes.
    _,regions=cv2.connectedComponents(mask,8)
    owners={int(regions[int(p['anchor_px'][1]),int(p['anchor_px'][0])]) for p in candidate['pins'] if p.get('state')=='supported' and p.get('anchor_px')}
    owners.discard(0)
    branches=[];nodes=set();boundary=np.isin(regions,list(owners)).astype(np.uint8);width=candidate['evidence']['stroke_scene']['stroke_width_px']
    for component in candidate['components']:
        if component['type']=='ground':continue
        pins=[p for p in candidate['pins'] if p['component_id']==component['id']]
        if len(pins)!=2 or any(p.get('state')!='supported' or not p.get('net_id') or not p.get('anchor_px') for p in pins):return None
        a,b=[np.array(p['anchor_px'],float) for p in pins]
        if np.linalg.norm(b-a)==0:return None
        cv2.line(boundary,tuple(a.astype(int)),tuple(b.astype(int)),1,max(1,round(width)))
        branches.append((component,pins,a,b));nodes.update(p['net_id'] for p in pins)
    if not branches or any(p.get('state')=='observed_external_terminal' for p in candidate.get('external_ports',[])):return None
    _,labels=cv2.connectedComponents(1-boundary,connectivity=4)
    outside=set(labels[0])|set(labels[-1])|set(labels[:,0])|set(labels[:,-1]);outside.discard(0)
    labels[np.isin(labels,list(outside))]=-1
    adjacent=[];face_ids=set()
    for component,pins,a,b in branches:
        axis=(b-a)/np.linalg.norm(b-a);normal=np.array([-axis[1],axis[0]])
        pairs=[(_side(labels,a+(b-a)*t,normal),_side(labels,a+(b-a)*t,-normal)) for t in (.35,.5,.65)]
        if len(set(pairs))!=1:return None
        right,left=pairs[0];adjacent.append((right,left));face_ids.update((right,left))
    face_ids=sorted(face_ids-{0});node_ids=sorted(nodes)
    if not face_ids:return None
    incidence=sp.zeros(len(nodes),len(branches));cycles=sp.zeros(len(branches),len(face_ids));neighbors={n:set() for n in nodes}
    for i,((_,pins,_,_),(right,left)) in enumerate(zip(branches,adjacent)):
        a,b=[p['net_id'] for p in pins];incidence[node_ids.index(a),i]+=1;incidence[node_ids.index(b),i]-=1;neighbors[a].add(b);neighbors[b].add(a)
        for face,sign in ((right,1),(left,-1)):
            if face:cycles[i,face_ids.index(face)]+=sign
    remaining=set(nodes);connected=0
    while remaining:
        connected+=1;stack=[remaining.pop()]
        while stack:
            for other in neighbors[stack.pop()]&remaining:remaining.remove(other);stack.append(other)
    rank=len(branches)-len(nodes)+connected
    if len(face_ids)!=rank or cycles.rank()!=rank or incidence*cycles!=sp.zeros(len(nodes),len(face_ids)):return None
    # I_edge = m_right - m_left. The outside mesh is zero. A dual spanning
    # tree provides exact integer coordinates, avoiding a numerical pseudoinverse.
    dual={f:[] for f in [0]+face_ids}
    for (_,pins,_,_),(right,left) in zip(branches,adjacent):
        if right==left:continue
        dual[left].append((right,pins[0]['id'],1));dual[right].append((left,pins[0]['id'],-1))
    expressions={0:{}};queue=deque([0])
    while queue:
        face=queue.popleft()
        for other,pin,sign in dual[face]:
            if other in expressions:continue
            terms=dict(expressions[face]);terms[pin]=terms.get(pin,0)+sign;expressions[other]={p:s for p,s in terms.items() if s};queue.append(other)
    if set(expressions)!=set(dual):return None
    coordinate=sp.Matrix([[expressions[face].get(pins[0]['id'],0) for _,pins,_,_ in branches] for face in face_ids])
    if coordinate*cycles!=sp.eye(rank):return None
    return labels,expressions,{'cycle_rank':rank,'incidence_times_face_cycles_zero':True,'complete_face_basis':True,'coordinate_times_cycles_identity':True,
        'branch_pin_ids':[pins[0]['id'] for _,pins,_,_ in branches],'face_ids':face_ids,'clockwise_branch_matrix':np.array(cycles.tolist(),dtype=int).tolist(),
        'method':'observed_graph_circulations_and_oriented_dual_tree','auxiliary_body_lines_never_used_as_conductors':True}


def _curves(bits):
    from ._topology_geometry import _thin
    count,labels,stats,_=cv2.connectedComponentsWithStats(bits,8);result=[]
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i])
        if area<24:continue
        obj=(labels[y:y+h,x:x+w]==i).astype(np.uint8);distance=cv2.distanceTransform(obj,cv2.DIST_L2,3);stroke=float(np.median(distance[distance>0]));peak=float(distance.max())
        if peak<2*stroke:continue
        cores=(distance>=.8*peak).astype(np.uint8)
        if cv2.connectedComponents(cores,8)[0]!=2:continue
        hy,hx=np.nonzero(cores);head=np.array([hx.mean(),hy.mean()])
        yy,xx=np.mgrid[:h,:w];shaft=obj.copy();shaft[np.hypot(xx-head[0],yy-head[1])<=2*peak]=0
        n,parts,partstats,_=cv2.connectedComponentsWithStats(shaft,8)
        if n<2:continue
        main=1+int(np.argmax(partstats[1:,cv2.CC_STAT_AREA]));shaft=(parts==main).astype(np.uint8)
        if shaft.sum()<.5*obj.sum():continue
        skeleton,converged,_=_thin(shaft)
        if not converged:continue
        count=cv2.filter2D(skeleton,-1,np.ones((3,3),np.uint8))-skeleton
        ey,ex=np.nonzero((skeleton!=0)&(count==1));ends=np.column_stack((ex,ey))
        if len(ends)!=2:continue
        endpoint_distances=np.sort(np.linalg.norm(ends-head,axis=1))
        if endpoint_distances[0]>4*peak or endpoint_distances[1]-endpoint_distances[0]<2*peak:continue
        yy,xx=np.nonzero(skeleton);points=np.column_stack((xx,yy)).astype(np.float32)
        if len(points)<12:continue
        try:ellipse=cv2.fitEllipse(points.reshape(-1,1,2))
        except cv2.error:continue
        center=np.array(ellipse[0]);axes=np.array(ellipse[1])/2;angle=math.radians(ellipse[2]);rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        if axes.min()<4*peak or axes.max()>3*axes.min():continue
        coordinates=(points-center)@rotation/axes;angles=np.sort(np.arctan2(coordinates[:,1],coordinates[:,0])%(2*math.pi));gaps=np.diff(np.r_[angles,angles[0]+2*math.pi]);gap=int(np.argmax(gaps));coverage=2*math.pi-gaps[gap]
        error=float(np.quantile(abs(np.linalg.norm(coordinates,axis=1)-1)*axes.min(),.9))
        if coverage<math.pi/2 or error>2*stroke:continue
        u=(head-center)@rotation/axes
        if abs(np.linalg.norm(u)-1)*axes.min()>peak:continue
        theta=math.atan2(u[1],u[0])%(2*math.pi);start=angles[(gap+1)%len(angles)];end=angles[gap];angular=lambda a,b:abs((a-b+math.pi)%(2*math.pi)-math.pi)
        sense=1 if angular(theta,end)<angular(theta,start) else -1
        tangent=rotation@np.array([-axes[0]*math.sin(theta),axes[1]*math.cos(theta)])*sense;tangent/=np.linalg.norm(tangent)
        yy,xx=np.nonzero(obj);all_points=np.column_stack((xx,yy));near=all_points[np.linalg.norm(all_points-head,axis=1)<=3*peak];normal=np.array([-tangent[1],tangent[0]]);along=(near-head)@tangent;across=(near-head)@normal;tip=along>=along.max()-stroke;shoulder=(along<along.max()-2*stroke)&(along>along.max()-3*peak)
        if not shoulder.any():continue
        tip_width=float(np.ptp(across[tip])+1);head_width=float(np.ptp(across[shoulder])+1)
        if head_width<3*stroke or tip_width>.55*head_width:continue
        result.append({'bbox_px':[x,y,x+w,y+h],'center_px':(center+[x,y]).tolist(),'head_px':(head+[x,y]).tolist(),'direction_unit':tangent.tolist(),'orientation':sense,
            'fit':{'ellipse_axes_px':axes.tolist(),'arc_coverage_rad':float(coverage),'residual90_px':error,'shaft_radius_px':stroke,'head_radius_px':peak,'tip_width_px':tip_width,'head_width_px':head_width}})
    return result


def observe_mesh_references(candidate):
    mask=(cv2.imread(str(candidate['mask_path']),0)>0).astype(np.uint8);cells=_cells(candidate,mask)
    if cells is None:return {'references':[],'evidence':{'status':'no_complete_observed_planar_face_basis'}}
    faces,expressions,proof=cells
    gray=cv2.cvtColor(np.asarray(Image.open(candidate['image']['path']).convert('RGB')),cv2.COLOR_RGB2GRAY);_,regions=cv2.connectedComponents(mask,8)
    occupied={int(regions[int(p['anchor_px'][1]),int(p['anchor_px'][0])]) for p in candidate['pins'] if p.get('anchor_px')};occupied.discard(0)
    excluded=cv2.dilate(np.isin(regions,list(occupied)).astype(np.uint8),np.ones((3,3),np.uint8))
    for c in candidate['components']:
        a,b,d,e=map(int,c['bbox']);excluded[b:e,a:d]=1
    observations=[];paper=float(np.percentile(gray,90))
    for contrast in (48,60):observations.append(_curves(((gray<=paper-contrast)&(excluded==0)).astype(np.uint8)))
    references=[]
    for first in observations[0]:
        matches=[r for r in observations[1] if r['orientation']==first['orientation'] and np.linalg.norm(np.array(first['center_px'])-r['center_px'])<=max(first['fit']['shaft_radius_px'],r['fit']['shaft_radius_px'])*2]
        if len(matches)!=1:continue
        second=matches[0];points=[np.floor(r['center_px']).astype(int) for r in (first,second)]
        if any(not(0<=x<faces.shape[1] and 0<=y<faces.shape[0]) for x,y in points):continue
        ids=[int(faces[y,x]) for x,y in points]
        if ids[0]<=0 or ids[0]!=ids[1] or ids[0] not in expressions:continue
        # The fitted center and entire native arrow rectangle must occupy one
        # observed cell. No closest-component or closest-loop assignment.
        a,b,d,e=first['bbox_px'];inside=set(np.unique(faces[b:e,a:d]))-{0}
        if inside!={ids[0]}:continue
        terms=[{'pin_id':p,'coefficient':int(sign*first['orientation'])} for p,sign in sorted(expressions[ids[0]].items())]
        if not terms:continue
        refs={'bbox_px':first['bbox_px'],'point_px':first['center_px'],'direction_unit':first['direction_unit'],'kind':'mesh_current',
            'orientation':'clockwise' if first['orientation']==1 else 'counterclockwise','choices':{'m1':{'face_id':ids[0],'point_px':first['center_px'],'terminal_current_terms':terms}},
            'evidence':'native_curved_pointed_arrow_two_thresholds_inside_proved_planar_face','observations':[first,second]}
        references.append(refs)
    target=Path(candidate['mask_path']).parent/'mesh-face-labels.png';cv2.imwrite(str(target),faces.astype(np.uint16));return {'references':references,'evidence':{'status':'observed','face_labels_path':str(target),'face_basis':proof,'arrow_count':len(references)}}
