"""Source-observed polarity pairs. No text detector, OCR or guessed net geometry."""
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image



from ._graph.semantic_pins import _sign


def _source_sign(bits):
    # External voltage minus marks follow the electrode axis horizontally;
    # isolated vertical text strokes are not minus signs.
    h,w=bits.shape
    if w>=5 and w>=3*h:
        # A straight bar is defined by its continuous central band, not the
        # occupancy of its antialiased bounding rectangle. Edge pixels may
        # add a sparse row without changing the observed stroke.
        rows=np.flatnonzero(bits.sum(1)>.5*w)
        if len(rows) and rows[-1]-rows[0]+1==len(rows) and bits.sum(1).max()>=.9*w:
            core=np.zeros_like(bits);core[rows,:]=1
            halo=cv2.dilate(core,np.ones((3,3),np.uint8))
            if np.all(bits.sum(0)>0) and not np.any(bits & (1-halo)):
                return '-'
    sign=_sign(bits)
    return None if sign=='-' and bits.shape[0]>bits.shape[1] else sign


def _marks(candidate):
    gray=cv2.cvtColor(np.asarray(Image.open(candidate['image']['path']).convert('RGB')),cv2.COLOR_RGB2GRAY);paper=float(np.percentile(gray,90));observations=[]
    for threshold in (48,60):
        ink=(gray<=paper-threshold).astype(np.uint8);n,labels,stats,_=cv2.connectedComponentsWithStats(ink,8);marks=[]
        for k in range(1,n):
            x,y,w,h,area=map(int,stats[k])
            if area<5 or area>1000:continue
            center=[x+(w-1)/2,y+(h-1)/2]
            if any(c['bbox'][0]<=center[0]<=c['bbox'][2] and c['bbox'][1]<=center[1]<=c['bbox'][3] for c in candidate['components']):continue
            sign=_source_sign((labels[y:y+h,x:x+w]==k).astype(np.uint8))
            if sign:marks.append({'sign':sign,'center_px':center,'bbox_px':[x,y,x+w,y+h]})
        observations.append(marks)
    return [a for a in observations[0] if sum(b['sign']==a['sign'] and math.dist(a['center_px'],b['center_px'])<=max(1.,.15*max(a['bbox_px'][2]-a['bbox_px'][0],a['bbox_px'][3]-a['bbox_px'][1])) for b in observations[1])==1]


def polarity_pairs(candidate):
    from .measurements import sites
    observed=sites(candidate);owned={p['id']:p for p in observed};components={c['id']:c for c in candidate['components']};marks=_marks(candidate);proposals=[]
    for c in components.values():
        pins=[p for p in candidate['pins'] if p['component_id']==c['id'] and p['id'] in owned]
        if len(pins)==2:proposals.append((owned[pins[0]['id']],owned[pins[1]['id']],c['id']))
    ports={}
    for p in observed:
        if p['kind']=='external_terminal':ports.setdefault((p['net_id'],tuple(p['point_px'])),p)
    ports=list(ports.values())
    for i,a in enumerate(ports):
        for b in ports[i+1:]:proposals.append((a,b,None))
    result=[];seen=set()
    def add(a,b,kind,component_id,proof):
        key=(a['net_id'],b['net_id'],tuple(a['point_px']),tuple(b['point_px']))
        if key in seen:return
        seen.add(key);result.append({'id':'vp'+str(len(result)+1),'kind':kind,'component_id':component_id,'positive_site_id':a['id'],'negative_site_id':b['id'],'positive_net_id':a['net_id'],'negative_net_id':b['net_id'],'evidence':proof})
    for a,b,cid in proposals:
        for p,n in ((a,b),(b,a)):
            start=np.array(p['point_px']);end=np.array(n['point_px']);v=end-start;length=float(np.linalg.norm(v))
            if length<5:continue
            u=v/length;normal=np.array([-u[1],u[0]]);matches=[]
            for plus in (m for m in marks if m['sign']=='+'):
                for minus in (m for m in marks if m['sign']=='-'):
                    pa=np.array(plus['center_px']);ma=np.array(minus['center_px']);axis=ma-pa;span=float(np.linalg.norm(axis))
                    if span==0 or float(axis@u)/span<.95:continue
                    left=float((pa-start)@u)/length;right=float((ma-start)@u)/length
                    if not (-.5<=left<=.5 and .5<=right<=1.5):continue
                    offsets=[float((m-start)@normal) for m in (pa,ma)];glyph=max(plus['bbox_px'][2]-plus['bbox_px'][0],plus['bbox_px'][3]-plus['bbox_px'][1])
                    if max(map(abs,offsets))>.75*length or abs(offsets[0]-offsets[1])>glyph:continue
                    matches.append({'positive_mark':plus,'negative_mark':minus,'method':'source_sign_pair_two_thresholds_and_owned_electrode_axis'})
            if len(matches)==1:add(p,n,'native_polarity_marks',cid,matches[0])
        if cid and components[cid]['type'] in ('voltage_source','dependent_voltage_source'):
            role={pin.get('role'):pin['id'] for pin in candidate['pins'] if pin['component_id']==cid}
            if 'positive' in role and 'negative' in role:add(owned[role['positive']],owned[role['negative']],'source_symbol',cid,{'method':'owned_source_polarity_contacts'})
    return result
