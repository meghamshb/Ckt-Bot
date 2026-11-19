"""Bounded pixel-only polarity observations; no inferred VLM geometry.

Accepts isolated printed + and − marks inside an observed closed symbol body.
This intentionally abstains on touched marks, extra interior glyphs and ambiguity.
It does not localize symbol-body terminals or identify op-amp supply roles.
"""
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def _straight_bar(mask):
    """A connected filled bar with one pixel-cell boundary uncertainty.

    An antialias fringe changes the bounding box without changing the bar.
    Every section must still be filled and share a common principal row.
    This is used only for distinct glyphs inside the observed source body.
    """
    h,w=mask.shape
    if h>w:mask=mask.T;h,w=mask.shape
    if w<5 or w<3*h:return False
    tops=[];bottoms=[]
    for column in mask.T:
        occupied=np.flatnonzero(column)
        if not len(occupied) or occupied[-1]-occupied[0]+1!=len(occupied):return False
        tops.append(int(occupied[0]));bottoms.append(int(occupied[-1]))
    # The measured central bar is straight. Rounded endpoint cells belong to
    # caps whose axial extent cannot exceed the observed half-stroke plus its
    # boundary cell; they are not evidence for a curved central bar.
    cap=math.ceil(h/2);core_top=tops[cap:w-cap];core_bottom=bottoms[cap:w-cap]
    if (not core_top or max(core_top)>min(core_bottom)
            or max(core_top)-min(core_top)>1 or max(core_bottom)-min(core_bottom)>1):return False
    if max(tops)>min(bottoms):return False
    center=float(np.median((np.asarray(core_top)+core_bottom)/2))
    for indices in (range(cap+1),range(w-1,w-cap-2,-1)):
        heights=[]
        for i in indices:
            if abs((tops[i]+bottoms[i])/2-center)>.5:return False
            heights.append(bottoms[i]-tops[i]+1)
        if np.any(np.diff(heights)<-1):return False
    return True


def _sign(mask):
    h,w=mask.shape
    if min(h,w)<1 or max(h,w)<5:return None
    if _straight_bar(mask):return '-'
    if not .65<=w/h<=1.55 or min(h,w)<5:return None
    # Fit full bars using majority cross-sections. Antialiasing can change
    # edge pixels between contrasts while preserving the same cross skeleton.
    rowsums=mask.sum(1);colsums=mask.sum(0)
    if rowsums.max()<.9*w or colsums.max()<.9*h:return None
    rows=np.flatnonzero(rowsums>.5*w);cols=np.flatnonzero(colsums>.5*h)
    if not len(rows) or not len(cols) or len(rows)>.4*h or len(cols)>.4*w:return None
    if rows[-1]-rows[0]+1!=len(rows) or cols[-1]-cols[0]+1!=len(cols):return None
    if not (.2*h<rows.mean()<.8*h and .2*w<cols.mean()<.8*w):return None
    expected=np.zeros_like(mask);expected[rows,:]=1;expected[:,cols]=1
    edge=max(1,math.ceil(min(len(rows),len(cols))/2))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*edge+1,2*edge+1))
    if np.any(mask & (1-cv2.dilate(expected,kernel))):return None
    if np.any(expected & (1-cv2.dilate(mask,kernel))):return None
    if np.any(cv2.erode(expected,kernel) & (1-mask)):return None
    return '+'


def _observe(gray,box,delta):
    x0,y0,x1,y1=box;crop=gray[y0:y1,x0:x1]
    ink=(crop<=float(np.percentile(gray,90))-delta).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
    if count<4:return None
    body=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    contours,_=cv2.findContours((labels==body).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<100:return None
    # Only distinct interior marks count. Any extra interior glyph blocks claims.
    signs=[]
    for i in range(1,count):
        if i==body:continue
        x,y,w,h,area=stats[i]
        center=(float(x+(w-1)/2),float(y+(h-1)/2))
        if cv2.pointPolygonTest(contour,center,True)<=2:continue
        sign=_sign((labels[y:y+h,x:x+w]==i).astype(np.uint8))
        if sign is None:return None
        signs.append((sign,[center[0]+x0,center[1]+y0]))
    if sorted(s for s,_ in signs)!=['+','-']:return None
    hull=cv2.convexHull(contour)
    poly=cv2.approxPolyDP(hull,.04*cv2.arcLength(hull,True),True)
    vertices=[[float(p[0][0]+x0),float(p[0][1]+y0)] for p in poly] if len(poly)==3 else None
    return sorted(signs),vertices


def infer_pin_roles(image_path,components,pins,semantic_pin_policy="frozen"):
    """Return assemble-compatible observations {pins, unresolved}.

    Only image-pixel sign evidence can produce +/−. Each sign must bind a unique
    nearest observed wire anchor with a distance margin. Output requires a
    distinct observed triangular apex, never box orientation or model topology.
    Existing physical terminal coordinates are preserved; none are invented.
    """
    if semantic_pin_policy == 'observation_consensus_v1':
        return _infer_consensus(image_path,components,pins)
    if semantic_pin_policy != 'frozen':
        raise ValueError('Unknown semantic pin policy: '+str(semantic_pin_policy))
    image=Image.open(image_path).convert('L');gray=np.asarray(image);h,w=gray.shape
    if h*w>2_000_000:raise ValueError('Semantic pin observer image budget exceeded')
    source=hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    result={'pins':[],'unresolved':[]}
    for c in components:
        if c['type'] not in ('voltage_source','dependent_voltage_source','op_amp'):continue
        b=c['bbox'];box=[max(0,int(math.floor(b[0]))),max(0,int(math.floor(b[1]))),min(w,int(math.ceil(b[2]))),min(h,int(math.ceil(b[3])))]
        if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Invalid component bbox')
        if (box[2]-box[0])*(box[3]-box[1])>65536:continue
        observed=_observe(gray,box,48)
        stable=observed is not None and observed==_observe(gray,box,60)
        candidates=[p for p in pins if p['component_id']==c['id'] and p.get('anchor_px') is not None
                    and p.get('state') not in ('ambiguous','unattached','unsupported','unresolved')]
        if not stable or len(candidates)<2:
            result['unresolved'].append({'component_id':c['id'],'reason':'interior_polarity_marks_not_uniquely_observed'});continue
        signs,vertices=observed;chosen=[]
        for sign,center in signs:
            ranked=sorted((math.dist(center,p['anchor_px']),p['id'],p) for p in candidates)
            if ranked[1][0]-ranked[0][0]<3:break
            chosen.append((sign,center,ranked[0][2]))
        if len(chosen)!=2 or chosen[0][2]['id']==chosen[1][2]['id']:
            result['unresolved'].append({'component_id':c['id'],'reason':'polarity_anchor_ownership_ambiguous'});continue
        for sign,center,pin in chosen:
            result['pins'].append({'component_id':c['id'],'role':sign,'anchor_px':pin['anchor_px'],
                'terminal_px':pin.get('terminal_px'),'evidence':[f'pixel_sign:{sign}:center:{center}:image:{source}', 'threshold48_60_agreement']})
        if c['type']=='op_amp' and vertices:
            # The apex lies farthest from both observed signed input anchors.
            ordered=sorted((sum(math.dist(v,p[2]['anchor_px']) for p in chosen),i,v) for i,v in enumerate(vertices))
            if ordered[-1][0]-ordered[-2][0]<5:continue
            apex=ordered[-1][2];used={p[2]['id'] for p in chosen}
            remaining=[p for p in candidates if p['id'] not in used]
            matches=[p for p in remaining if math.dist(apex,p['anchor_px'])<=max(8,.12*math.hypot(box[2]-box[0],box[3]-box[1]))]
            if len(matches)==1:
                pin=matches[0]
                result['pins'].append({'component_id':c['id'],'role':'output','anchor_px':pin['anchor_px'],
                    'terminal_px':pin.get('terminal_px'),'evidence':[f'observed_triangle_apex:{apex}:image:{source}','distinct_from_signed_input_anchors']})
    return result


def _assign_signs(observation,candidates):
    """Use the frozen unique-anchor margin independently for one observation."""
    if observation is None or len(candidates)<2:
        return None
    chosen=[]
    for sign,center in observation[0]:
        ranked=sorted((math.dist(center,p['anchor_px']),p['id'],p) for p in candidates)
        if ranked[1][0]-ranked[0][0]<3:
            return None
        chosen.append((sign,center,ranked[0][2]))
    if len(chosen)!=2 or chosen[0][2]['id']==chosen[1][2]['id']:
        return None
    return chosen


def _assign_apex(observation,chosen,candidates,box):
    """Output needs an observed triangle and a unique unused anchor, as before."""
    vertices=observation[1]
    if not vertices:
        return None
    ordered=sorted((sum(math.dist(v,p[2]['anchor_px']) for p in chosen),i,v)
                   for i,v in enumerate(vertices))
    if ordered[-1][0]-ordered[-2][0]<5:
        return None
    apex=ordered[-1][2];used={p[2]['id'] for p in chosen}
    matches=[p for p in candidates if p['id'] not in used and
             math.dist(apex,p['anchor_px'])<=max(8,.12*math.hypot(box[2]-box[0],box[3]-box[1]))]
    return (matches[0],apex) if len(matches)==1 else None


def _infer_consensus(image_path,components,pins):
    """Require independent threshold observations to agree on pin identity.

    No coordinate averaging or single-threshold role claims.
    Sign roles may survive an uncertain output apex; supply roles stay unknown.
    """
    gray=np.asarray(Image.open(image_path).convert('L'));h,w=gray.shape
    if h*w>2_000_000:
        raise ValueError('Semantic pin observer image budget exceeded')
    source=hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    result={'pins':[],'unresolved':[]}
    for c in components:
        if c['type'] not in ('voltage_source','dependent_voltage_source','op_amp'):
            continue
        b=c['bbox'];box=[max(0,int(math.floor(b[0]))),max(0,int(math.floor(b[1]))),
                       min(w,int(math.ceil(b[2]))),min(h,int(math.ceil(b[3])))]
        if box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid component bbox')
        if (box[2]-box[0])*(box[3]-box[1])>65536:
            continue
        observations=[_observe(gray,box,delta) for delta in (48,60)]
        if any(obs is None for obs in observations):
            result['unresolved'].append({'component_id':c['id'],'reason':'interior_polarity_marks_not_uniquely_observed'})
            continue
        candidates=[p for p in pins if p['component_id']==c['id'] and p.get('anchor_px') is not None
                    and p.get('state') not in ('ambiguous','unattached','unsupported','unresolved')]
        assignments=[_assign_signs(obs,candidates) for obs in observations]
        if any(a is None for a in assignments):
            result['unresolved'].append({'component_id':c['id'],'reason':'polarity_anchor_ownership_ambiguous'})
            continue
        identities=[{sign:pin['id'] for sign,_,pin in a} for a in assignments]
        if identities[0]!=identities[1]:
            result['unresolved'].append({'component_id':c['id'],'reason':'polarity_anchor_identity_disagrees_between_thresholds'})
            continue
        for sign,_,pin in assignments[0]:
            role={'+':'noninverting','-':'inverting'}[sign] if c['type']=='op_amp' else sign
            observed=[{'threshold_delta':delta,'sign':sign,
                       'glyph_center_px':next(center for s,center,_ in assignment if s==sign),
                       'triangle_vertices_px':obs[1],'assigned_pin_id':pin['id']}
                      for delta,assignment,obs in zip((48,60),assignments,observations)]
            result['pins'].append({'component_id':c['id'],'role':role,'anchor_px':pin['anchor_px'],
                'terminal_px':pin.get('terminal_px'),'semantic_observations':observed,
                'evidence':[f'pixel_sign:{sign}:image:{source}','threshold48_60_anchor_identity_consensus',
                            'semantic_pin_policy:observation_consensus_v1']})
        if c['type']!='op_amp':
            continue
        apexes=[_assign_apex(obs,assignment,candidates,box) for obs,assignment in zip(observations,assignments)]
        if any(a is None for a in apexes) or apexes[0][0]['id']!=apexes[1][0]['id']:
            result['unresolved'].append({'component_id':c['id'],'reason':'triangle_output_anchor_not_uniquely_consistent'})
            continue
        pin=apexes[0][0]
        result['pins'].append({'component_id':c['id'],'role':'output','anchor_px':pin['anchor_px'],
            'terminal_px':pin.get('terminal_px'),
            'semantic_observations':[{'threshold_delta':delta,'triangle_vertices_px':obs[1],
                'apex_px':apex[1],'assigned_pin_id':pin['id']} for delta,obs,apex in zip((48,60),observations,apexes)],
            'evidence':[f'observed_triangle_apex:image:{source}','threshold48_60_output_anchor_identity_consensus',
                        'distinct_from_signed_input_anchors','semantic_pin_policy:observation_consensus_v1']})
    return result
