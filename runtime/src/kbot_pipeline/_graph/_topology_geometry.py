"""Bounded actual canonical-pixel wire/contact observations. No model or solver.

Legacy contact/crossover primitives are reused narrowly. Pixel connectivity is a
fallible producer observation; masks, open ends and unresolved events stay visible.
"""
import hashlib,math,resource,sys,time
from collections import deque
import cv2
import numpy as np
import json
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"))
from ._topology_observers import _contact,_diagonal_contacts,_diagonal_masks,_local_crossings,_possible_bridges,_ground_body_box,earth_ground_shape
from ._topology_bridges import verified_crossover
from .wire_junctions import collar_arms,junction_kind,CONVENTIONS,within_complete_junction
from .wire_contacts import contact_endpoint_proof,observed_multiarm_endpoint
from .stroke_scene import conductor_appearance,collar_contacts

VERSION='kbot.topology-geometry/5'
LIMITS={'pixels':2000000,'components':256,'skeleton_pixels':100000,'fragments':4096,'terminals':2048,'thinning_iterations':128,'seconds':45,'owned_RSS_bytes':512*1024**2,'reserve_available_bytes':4*1024**3}
TWO_PIN={'resistor','capacitor','inductor','diode','switch','voltage_source','current_source','dependent_voltage_source','dependent_current_source'}
KNOWN=TWO_PIN|{'ground','junction','op_amp','impedance_block','ideal_transformer','transistor_bjt','two_port_block','meter'}
DIRECTIONS={'left':[-1.,0.],'right':[1.,0.],'top':[0.,-1.],'bottom':[0.,1.]}


def _guard(start):
    import psutil
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)
    if peak>LIMITS['owned_RSS_bytes']or time.monotonic()-start>LIMITS['seconds']:raise ValueError('Topology CPU memory/time budget exceeded')
    if psutil.virtual_memory().available<LIMITS['reserve_available_bytes']:raise ValueError('Topology stopped to preserve system reserve')
    return int(peak)


def _id(prefix,value):return prefix+hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def _thin(mask):
    """Zhang–Suen thinning, bounded, retaining loops and original-mask provenance."""
    image=np.pad(mask.astype(np.uint8),1);done=False
    for iteration in range(LIMITS['thinning_iterations']):
        changed=False
        for phase in (0,1):
            p=image[1:-1,1:-1];a=[image[:-2,1:-1],image[:-2,2:],image[1:-1,2:],image[2:,2:],image[2:,1:-1],image[2:,:-2],image[1:-1,:-2],image[:-2,:-2]]
            count=sum(a);turns=sum((a[i]==0)&(a[(i+1)%8]!=0)for i in range(8))
            if phase==0:condition=(a[0]*a[2]*a[4]==0)&(a[2]*a[4]*a[6]==0)
            else:condition=(a[0]*a[2]*a[6]==0)&(a[0]*a[4]*a[6]==0)
            remove=(p!=0)&(count>=2)&(count<=6)&(turns==1)&condition
            if remove.any():p[remove]=0;changed=True
        if not changed:done=True;break
    return image[1:-1,1:-1],done,iteration+1


def _adjacency(skeleton,regions):
    ys,xs=np.nonzero(skeleton)
    if len(xs)>LIMITS['skeleton_pixels']:raise ValueError('Topology skeleton budget exceeded')
    height,width=skeleton.shape;result={}
    for x,y in zip(xs.tolist(),ys.tolist()):
        neighbors=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                xx,yy=x+dx,y+dy
                if not(dx or dy)or not(0<=xx<width and 0<=yy<height)or not skeleton[yy,xx]:continue
                if regions[yy,xx]!=regions[y,x]:continue
                if dx and dy and(skeleton[y,xx]or skeleton[yy,x]):continue
                neighbors.append((xx,yy))
        result[(x,y)]=sorted(neighbors)
    return result


def _trace(skeleton,regions):
    neighbors=_adjacency(skeleton,regions);used=set();paths=[]
    def edge(a,b):return tuple(sorted((a,b)))
    starts=sorted(p for p,n in neighbors.items()if len(n)!=2)
    def walk(start,other):
        points=[start];previous,current=start,other;used.add(edge(previous,current))
        while True:
            points.append(current)
            if len(neighbors[current])!=2:break
            following=next(p for p in neighbors[current]if p!=previous)
            if edge(current,following)in used:break
            used.add(edge(current,following));previous,current=current,following
        return points
    for p in starts:
        if not neighbors[p]:paths.append([p])
        for q in neighbors[p]:
            if edge(p,q)not in used:paths.append(walk(p,q))
    for p in sorted(neighbors):
        for q in neighbors[p]:
            if edge(p,q)not in used:paths.append(walk(p,q))
    if len(paths)>LIMITS['fragments']:raise ValueError('Topology fragment budget exceeded')
    fragments=[];by_region={}
    for path in paths:
        if len(path)>2:
            keep=[path[0]]
            for a,b,c in zip(path,path[1:],path[2:]):
                if(b[0]-a[0],b[1]-a[1])!=(c[0]-b[0],c[1]-b[1]):keep.append(b)
            keep.append(path[-1])
        else:keep=path
        poly=[[x+.5,y+.5]for x,y in keep];uid=_id('wire-',poly);region=int(regions[path[0][1],path[0][0]])
        row={'id':uid,'polyline_px':poly,'observed_skeleton_samples':len(path),'closed_loop':len(path)>2 and path[0]==path[-1],
             'incidence_complete':True,'support_refs':['occupied_eight_neighbor_region:'+str(region)],'verified':False}
        fragments.append(row);by_region.setdefault(region,[]).append(uid)
    endpoints=[p for p,n in neighbors.items()if len(n)==1]
    return fragments,by_region,endpoints,neighbors


def _orth_arms(horizontal,vertical,x,y,reach=9):
    h,w=horizontal.shape;out=[]
    for name,dx,dy in [('left',-1,0),('right',1,0),('top',0,-1),('bottom',0,1)]:
        points=[(x+dx*k,y+dy*k)for k in range(2,reach+2)]
        if not all(1<=xx<w-1 and 1<=yy<h-1 for xx,yy in points):continue
        signal=horizontal if dx else vertical
        hits=sum(bool(signal[yy-1:yy+2,xx-1:xx+2].any())for xx,yy in points)
        if hits==len(points):out.append(name)
    return out


def _box(box,w,h):return [max(0,int(math.floor(box[0]))),max(0,int(math.floor(box[1]))),min(w,int(math.ceil(box[2]))),min(h,int(math.ceil(box[3])))]


def _body_witness(raw,box,contact,comparison=None,forbidden=None):
    """Source lead -> measured footprint -> owned nonstraight body.

    A contact coordinate represents a raster band, not a guaranteed ink pixel.
    Quantization expands that observed band only; no pixel or anchor is moved.
    """
    direction=contact.get('direction_from_body',DIRECTIONS.get(contact['side']))
    if direction is None:return None
    direction=np.asarray(direction,dtype=float);norm=float(np.linalg.norm(direction))
    if not norm:return None
    dx,dy=direction/norm;x,y=map(float,contact['point_px']);x0,y0,x1,y1=box
    width=max(1.,float(contact.get('edge_contact_width_px',1)))
    # The anchor band and source raster each contribute one half-cell diagonal.
    quantum=2*math.sqrt(.5);half=width/2+quantum;reach=max(3.,3*width)
    if contact.get('consecutive_wire_support_px') is not None:
        reach=min(reach,float(contact['consecutive_wire_support_px']))
    if reach<half:return None
    h,w=raw.shape;a=max(0,math.floor(min(x0,x-reach-half)));b=max(0,math.floor(min(y0,y-reach-half)))
    c=min(w,math.ceil(max(x1,x+reach+half))+1);d=min(h,math.ceil(max(y1,y+reach+half))+1)
    yy,xx=np.mgrid[b:d,a:c];along=(xx-x)*dx+(yy-y)*dy;across=(xx-x)*dy-(yy-y)*dx
    inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    lead=(along>=-quantum)&(along<=reach+quantum)&(np.abs(across)<=half)
    boundary_gap=max(0.,float(contact.get('boundary_gap_px',0)))
    gate=(np.hypot(xx-x,yy-y)<=half+boundary_gap)
    outside=(~inside)&lead&(along>=half)
    allowed=inside|lead
    if forbidden is not None:allowed&=forbidden[b:d,a:c]==0
    proofs=[]
    for observed in (raw,) if comparison is None else (raw,comparison):
        occupied=(observed[b:d,a:c]>0)&allowed
        _,labels=cv2.connectedComponents(occupied.astype(np.uint8),connectivity=8)
        owners=set(map(int,labels[occupied&gate]))-{0}
        if len(owners)!=1:return None
        component=labels==next(iter(owners))
        if not np.any(component&outside):return None
        body=component&inside
        nonstraight=body&(np.abs(across)>width+1)
        if not np.any(nonstraight):return None
        # The entry is actual occupied ink crossing the owned boundary, not a
        # nearby body pixel selected because it is closest to the anchor.
        entries=component&gate
        if not np.any(entries&inside) or not np.any(component&lead&(~inside)):return None
        proofs.append({'inward_connected_pixels':int(body.sum()),
            'off_lead_axis_body_pixels':int(nonstraight.sum()),
            'outside_lead_pixels':int((component&outside).sum()),
            'entry_footprint_pixels':int(entries.sum())})
    return {**proofs[0],'body_roi_px':box,'source_threshold_proofs':proofs,
        'contact_footprint_half_width_px':half,'quantization_radius_px':quantum,
        'observed_boundary_gap_px':boundary_gap,
        'method':'native_connected_outside_lead_footprint_owned_body',
        'anchor_is_representative_not_assumed_source_occupied':True,'verified':False}



def _trace_body_terminal(raw, box, contact):
    """Trace native axis-aligned lead ink to its first body-width transition.

    A box edge never supplies the terminal coordinate. Missing/gapped ink,
    diagonal leads, or a stroke that never enters wider body ink abstain.
    """
    side=contact['side']
    if side not in DIRECTIONS:return None
    outward=DIRECTIONS[side];dx,dy=[-int(v) for v in outward]
    x,y=map(int,contact['point_px']);x0,y0,x1,y1=box
    baseline=max(1,int(contact.get('edge_contact_width_px',1)))
    if baseline>12:return None
    path=[];height,width=raw.shape;reach=max(x1-x0,y1-y0)+4
    for step in range(reach):
        xx,yy=x+dx*step,y+dy*step
        if not (0<=xx<width and 0<=yy<height) or not raw[yy,xx]:return None
        # Cross-sections use only the connected run containing the lead axis.
        lo=hi=0
        for sign in (-1,1):
            extent=0
            for distance in range(1,65):
                a,b=xx+dy*distance*sign,yy-dx*distance*sign
                if not (0<=a<width and 0<=b<height) or not raw[b,a]:break
                extent=distance
            if sign<0:lo=extent
            else:hi=extent
        span=lo+hi+1
        inside=x0<=xx<x1 and y0<=yy<y1
        if inside and span>=baseline+max(3,baseline) and lo>=1 and hi>=1:
            if not path:return None
            # Require visible off-axis body support, beyond a fringe/noise pixel.
            body_pixels=sum(bool(raw[b,a]) for a,b in ((xx+dy*k,yy-dx*k) for k in range(-lo,hi+1))
                            if abs((a-xx)*dy-(b-yy)*dx)>baseline/2)
            if body_pixels<4:return None
            return {'point_px':[xx+.5,yy+.5],'lead_pixels':path,
                    'wire_anchor_px':[x+.5,y+.5],'axis_width_px':baseline,
                    'body_cross_section_px':span,'localization_radius_px':max(1,baseline/2),
                    'method':'native_contiguous_lead_to_body_width_transition','verified':False}
        if inside and span>baseline+2:return None
        path.extend([[xx+dy*k,yy-dx*k] for k in range(-lo,hi+1)])
        if step>2 and not inside:return None
    return None


def _native_short_contacts(side,box,mask,regions,raw,*,require_body=True):
    """One-to-eight real mask pixels touching the box; no gap/coordinate snapping.

    Native RGB must witness entry into the actual body, independently of the box.
    This only attaches the observed fragment; disconnected outer ink stays apart.
    """
    x0,y0,x1,y1=box;along=side in ('left','right')
    first={'left':x0-1,'right':x1,'top':y0-1,'bottom':y1}[side]
    sign=-1 if side in ('left','top') else 1
    low,high=(y0,y1) if along else (x0,x1)
    scan=mask.T if along else mask;labels=regions.T if along else regions
    if not 0<=first<scan.shape[0]:return []
    line=scan[first,low:high]>0
    changes=np.flatnonzero(np.diff(np.pad(line.astype(np.int8),1)))
    found=[]
    for lo,hi in zip(changes[::2],changes[1::2]):
        if hi-lo>12:continue
        center=low+(int(lo)+int(hi)-1)//2;region=int(labels[first,center])
        if not region:continue
        length=0
        for step in range(9):
            at=first+sign*step
            if not 0<=at<scan.shape[0] or not scan[at,center] or int(labels[at,center])!=region:break
            length+=1
        if not 1<=length<=8:continue
        point=[first,center] if along else [center,first]
        contact={'side':side,'point_px':point,'ink_region':region,'edge_contact_width_px':int(hi-lo),
                 'boundary_gap_px':0,'consecutive_wire_support_px':length,'contact_policy':'native_short_lead_v1'}
        if require_body and (_trace_body_terminal(raw,box,contact) is None or _body_witness(raw,box,contact) is None):continue
        found.append(contact)
    return found


def _departing_body_terminal(raw,box,contact,forbidden):
    """Observe an off-axis symbol entry; single wire bends remain ambiguous."""
    side=contact['side'];ox,oy=map(int,DIRECTIONS[side]);dx,dy=-ox,-oy
    x,y=map(int,contact['point_px']);x0,y0,x1,y1=box
    width=max(1,int(contact.get('edge_contact_width_px',1)))
    a,b,c,d=max(0,min(x0,x)),max(0,min(y0,y)),min(raw.shape[1],max(x1,x+1)),min(raw.shape[0],max(y1,y+1))
    patch=raw[b:d,a:c];_,labels=cv2.connectedComponents(patch,connectivity=8)
    label=labels[y-b,x-a]
    if not label:return None
    connected=labels==label;yy,xx=np.nonzero(connected);xx=xx+a;yy=yy+b
    inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    bx,by=xx[inside],yy[inside]
    if not len(bx) or forbidden[by,bx].any():return None
    along=(bx-x)*dx+(by-y)*dy;across=(bx-x)*dy-(by-y)*dx
    if int(along.max()-along.min())<4*width:return None
    centers=[]
    for step in range(int(along.min()),int(along.max())+1):
        values=across[along==step]
        if not len(values):return None
        centers.append((step,float(np.median(values))))
    excursion=max(3,width);pivot=centers[0][1];direction=0;runs=0
    for _,value in centers[1:]:
        change=value-pivot
        if not direction and abs(change)>=excursion:
            direction=1 if change>0 else -1;runs=1;pivot=value
        elif direction and change*direction>=0:pivot=value
        elif direction and abs(change)>=excursion:
            direction=-direction;runs+=1;pivot=value
    if runs<3:return None
    # Locate the first actual narrow lead departure, not a bbox projection.
    path=[];terminal=None
    for step in range(max(0,int(along.max()))+1):
        px,py=x+dx*step,y+dy*step
        if not (a<=px<c and b<=py<d) or not connected[py-b,px-a]:break
        coords=np.flatnonzero(connected[:,px-a]) + b if dx else np.flatnonzero(connected[py-b,:]) + a
        centre=float(np.median(coords));baseline=y if dx else x
        if abs(centre-baseline)>1:
            if not path:return None
            terminal=path[-1];break
        path.append([px,py])
    if terminal is None:return None
    tx,ty=terminal
    if not (x0<=tx<x1 and y0<=ty<y1):return None
    return {'point_px':[tx+.5,ty+.5],'lead_pixels':path[:-1],
        'wire_anchor_px':[x+.5,y+.5],'axis_width_px':width,'localization_radius_px':max(1,width/2),
        'method':'native_connected_body_axis_departure','body_shape_evidence':{'signed_transverse_runs':runs,'minimum_excursion_px':excursion,'connected_body_pixels':int(len(bx))},'verified':False}


def _transition_interval(supports,masks,contact):
    """Intersect existing native uncertainty; never invent a precise terminal."""
    direction=np.array(DIRECTIONS[contact['side']],dtype=float);inward=-direction
    a,b=[np.array(pair[0]['point_px'],dtype=float) for pair in supports]
    delta=b-a
    if abs(float(direction[0]*delta[1]-direction[1]*delta[0]))>1e-9:return None
    radii=[float(pair[0].get('localization_radius_px',0)) for pair in supports]
    if min(radii)<=0:return None
    values=[float(np.dot(point,inward)) for point in (a,b)]
    lower=max(t-r for t,r in zip(values,radii));upper=min(t+r for t,r in zip(values,radii))
    if lower>upper:return None
    steps=int(round(float(np.linalg.norm(b-a))))
    for step in range(steps+1):
        point=a+(b-a)*(step/max(1,steps));x,y=[int(round(v-.5)) for v in point]
        if not all(0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x] for mask in masks):return None
    chosen=int(np.argmax(values));reference=[a,b][chosen]
    start=reference+inward*(lower-values[chosen]);end=reference+inward*(upper-values[chosen])
    return {'coordinate_state':'interval_only','candidate_terminals_px':[a.tolist(),b.tolist()],
        'support_interval_px':[start.tolist(),end.tolist()],'native_localization_radii_px':radii,
        'path_reference_px':reference.tolist(),'rule':'overlap_of_existing_native_localization_supports_same_source_lead',
        'both_source_paths_present':True,'body_witnesses':[pair[1] for pair in supports],'verified':False}


def _contrast_body_support(gray,box,contact,mask,*,departure_fallback=False,forbidden=None,body_local=False,interval_agreement=False):
    """Local two-threshold RGB corroboration of actual mask pixels, never a fill.

    Weak leads need >=3 contiguous axial observations and contrast >=12 gray
    levels AND >=6 robust local-paper noise sigmas. Strong native strokes keep
    support for one-pixel stubs. Geometry must agree at 35%/65% of observed contrast.
    """
    side=contact['side']
    if side not in DIRECTIONS:return None
    dx,dy=map(int,DIRECTIONS[side]);x,y=map(int,contact['point_px']);h,w=gray.shape
    values=[]
    for step in range(12):
        a,b=x+dx*step,y+dy*step
        if not (0<=a<w and 0<=b<h) or not mask[b,a]:break
        values.append(float(gray[b,a]))
    if not values:return None
    x0,y0,x1,y1=box;left,top,right,bottom=max(0,min(x0,x)-12),max(0,min(y0,y)-12),min(w,max(x1,x)+13),min(h,max(y1,y)+13)
    roi=gray[top:bottom,left:right].astype(float)
    if not roi.size:return None
    paper_pixels=roi[roi>=np.percentile(roi,60)];paper=float(np.median(paper_pixels))
    noise=1.4826*float(np.median(np.abs(paper_pixels-paper)));lead=float(np.median(values));contrast=paper-lead
    if contrast<max(12.,6.*noise):return None
    if contrast<60 and len(values)<3:return None
    if sum(abs(v-lead)<=max(4.,2.*noise,.25*contrast) for v in values)<.8*len(values):return None
    supports=[];corroboration_masks=[]
    thresholds=[paper-.35*contrast,paper-.65*contrast]
    body_thresholds=None;body_contrast=None;body_gray=None
    if body_local:
        # Body colour is measured from connected off-axis native ink, not from
        # the darker lead or from all pixels in a mostly white detector box.
        provisional=np.zeros_like(mask);provisional[top:bottom,left:right]=(roi<=paper-max(12.,6.*noise)).astype(np.uint8)
        _,labels=cv2.connectedComponents(provisional,connectivity=8);label=labels[y,x]
        if not label:return None
        yy,xx=np.nonzero(labels==label)
        transverse=np.abs((xx-x)*dy-(yy-y)*dx)
        eligible=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)&(transverse>contact.get('edge_contact_width_px',1)+1)
        bx,by=xx[eligible],yy[eligible]
        if len(bx)<4 or forbidden[by,bx].any():return None
        body_gray=float(np.median(gray[by,bx]));body_contrast=paper-body_gray
        if body_contrast<max(12.,6.*noise):return None
        body_thresholds=[paper-.35*body_contrast,paper-.65*body_contrast]
    for index,threshold in enumerate(thresholds):
        weak=np.zeros_like(mask);weak[top:bottom,left:right]=(roi<=threshold).astype(np.uint8)
        if body_thresholds is not None:
            weak[y0:y1,x0:x1]=(gray[y0:y1,x0:x1]<=body_thresholds[index]).astype(np.uint8)
        witness=_trace_body_terminal(weak,box,contact);body=_body_witness(weak,box,contact)
        if witness is None and departure_fallback:
            witness=_departing_body_terminal(weak,box,contact,forbidden)
        if witness is None or body is None:return None
        supports.append((witness,body));corroboration_masks.append(weak)
    interval=None
    if supports[0][0]['point_px']!=supports[1][0]['point_px']:
        if not interval_agreement:return None
        interval=_transition_interval(supports,corroboration_masks,contact)
        if interval is None:return None
    witness,body=supports[0]
    if interval is not None:
        witness=next(pair[0] for pair in supports if pair[0]['point_px']==interval['path_reference_px'])
        witness=dict(witness,transition_uncertainty=interval)
    evidence={'method':'local_contrast_two_threshold_body_witness','paper_gray':paper,'lead_gray':lead,
              'contrast_gray':contrast,'paper_noise_sigma':noise,'minimum_contrast_gray':12.,'noise_multiplier':6.,
              'axial_mask_samples':len(values),'relative_thresholds_gray':thresholds,'roi_px':[left,top,right,bottom],
              'no_mask_pixels_added':True}
    if body_local:evidence.update(body_gray=body_gray,body_contrast_gray=body_contrast,body_relative_thresholds_gray=body_thresholds,body_threshold_scope='component_interior_only',method='separate_body_and_lead_local_contrast')
    witness=dict(witness,contrast_support=evidence)
    return {'terminal':witness,'body':dict(body,contrast_support=evidence)}

def _source_resistor_terminal(gray,box,contact,mask,regions,forbidden):
    """Localize one existing contact against a typed source body, never a box face."""
    if contact['side'] not in DIRECTIONS:return None
    proof=_contrast_body_support(gray,box,contact,mask,departure_fallback=True,
        forbidden=forbidden,body_local=True,interval_agreement=True)
    if proof is None:return None
    evidence=proof['terminal']['contrast_support'];left,top,right,bottom=evidence['roi_px']
    x0,y0,x1,y1=box;x,y=map(int,contact['point_px']);region=int(regions[y,x])
    if not region or forbidden[y,x]:return None
    supports=[];masks=[];shapes=[]
    for lead_threshold,body_threshold in zip(evidence['relative_thresholds_gray'],evidence['body_relative_thresholds_gray']):
        weak=np.zeros_like(mask);weak[top:bottom,left:right]=(gray[top:bottom,left:right]<=lead_threshold).astype(np.uint8)
        weak[y0:y1,x0:x1]=(gray[y0:y1,x0:x1]<=body_threshold).astype(np.uint8)
        body=_body_witness(weak,box,contact)
        terminal=_departing_body_terminal(weak,box,contact,forbidden)
        shape='repeated_transverse_body'
        if terminal is None:
            # Closed rectangle identity is independent of a lead widening. Never
            # apply the widening heuristic to circular or unknown source shapes.
            _,labels=cv2.connectedComponents(weak,connectivity=8);label=labels[y,x]
            if not label:return None
            connected=(labels==label).astype(np.uint8)
            contours,hierarchy=cv2.findContours(connected,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            rectangles=[]
            if hierarchy is not None:
                for i,contour in enumerate(contours):
                    if hierarchy[0,i,3]<0:continue
                    hull=cv2.convexHull(contour);poly=cv2.approxPolyDP(hull,.03*cv2.arcLength(hull,True),True)
                    if len(poly)!=4:continue
                    pts=poly[:,0,:]
                    if not all(x0<=a<x1 and y0<=b<y1 for a,b in pts):continue
                    edges=np.roll(pts,-1,axis=0)-pts
                    if any(min(abs(int(a)),abs(int(b)))>1 for a,b in edges):continue
                    if np.ptp(pts[:,0])<=2*contact.get('edge_contact_width_px',1) or np.ptp(pts[:,1])<=2*contact.get('edge_contact_width_px',1):continue
                    rectangles.append(poly)
            if len(rectangles)!=1:return None
            terminal=_trace_body_terminal(weak,box,contact);shape='closed_axis_aligned_rectangle'
        if terminal is None or body is None:return None
        pixels=terminal['lead_pixels']+[[int(v-.5) for v in terminal['point_px']]]
        for a,b in pixels:
            if forbidden[b,a]:return None
            if not (x0<=a<x1 and y0<=b<y1) and int(regions[b,a])!=region:return None
        supports.append((terminal,body));masks.append(weak);shapes.append(shape)
    if shapes[0]!=shapes[1]:return None
    interval=None
    if supports[0][0]['point_px']!=supports[1][0]['point_px']:
        interval=_transition_interval(supports,masks,contact)
        if interval is None:return None
    terminal=dict(supports[0][0])
    if interval is not None:
        terminal=dict(next(t for t,_ in supports if t['point_px']==interval['path_reference_px']),transition_uncertainty=interval)
    terminal.update(contrast_support=evidence,source_body_shape=shapes[0],
        localization_policy='source_body_v1',existing_contact_only=True,no_conductor_pixels_changed=True)
    return terminal


def _source_fill_group_proof(members,side,plane,low,upper,region,gray,probability,high,screened,regions,forbidden):
    """Inventory evidence only: no interior model-confidence or net union."""
    along=side in ('left','right');dx,dy=map(int,DIRECTIONS[side]);h,w=gray.shape
    def point(step,t):return (plane+dx*step,t) if along else (t,plane+dy*step)
    collars=[[point(step,t) for step in range(6) for t in ts] for ts in (range(low-5,low-1),range(upper+1,upper+5))]
    if not all(0<=x<w and 0<=y<h for collar in collars for x,y in collar):return None
    evidence=[];cutoffs=[]
    for collar in collars:
        values=np.array([float(gray[y,x]) for x,y in collar]);paper=float(np.median(values));noise=1.4826*float(np.median(np.abs(values-paper)));margin=max(12.,6*noise)
        if any(forbidden[y,x] for x,y in collar) or np.max(np.abs(values-paper))>margin:return None
        cutoffs.append(paper-margin);evidence.append({'background_gray':paper,'noise_sigma':noise,'required_contrast_gray':margin})
    for step in range(6):
        for t in range(low,upper):
            x,y=point(step,t)
            if forbidden[y,x] or gray[y,x]>min(cutoffs) or int(regions[y,x]) not in (0,region):return None
    for member in members:
        x,y=member['point_px']
        for step in range(6):
            a,b=x+dx*step,y+dy*step
            if probability[b,a]<high or not screened[b,a] or int(regions[b,a])!=region:return None
    return {'policy':'same_lead_v2','same_ink_region':region,'cross_sections':6,'observed_width_px':upper-low,
        'bilateral_background_evidence':evidence,'source_fill_cutoffs_gray':cutoffs,'member_high_threshold':high,
        'interior_probability_required':False,'operation':'existing_contact_inventory_collapse_only',
        'mask_pixels_added':0,'net_regions_unioned':0,'verified':False}


def _owned_body_intervals(intervals,main_pixels,raw,strict,body_box,contact,occupied,regions,forbidden,origin,axis,reach,half):
    """Classify actual weak body ink, never erase an exterior lead hypothesis."""
    if body_box is None or _body_witness(raw,body_box,contact,strict,forbidden) is None:return None
    x0,y0,x1,y1=body_box;extras={tuple(p) for interval in intervals for p in interval}
    if any(not(x0<=x-.5 and x+.5<=x1 and y0<=y-.5 and y+.5<=y1) or strict[y,x] for x,y in extras):return None
    # A box alone cannot own ink: require one actual nonstraight native body
    # component, corroborated by the stricter source and the independent entry.
    body_raw=raw[y0:y1,x0:x1].copy();body_raw&=(forbidden[y0:y1,x0:x1]==0)
    _,labels=cv2.connectedComponents(body_raw,connectivity=8)
    owners={int(labels[y-y0,x-x0]) for x,y in extras}-{0}
    if len(owners)!=1:return None
    body=labels==next(iter(owners));yy,xx=np.nonzero(body);xx+=x0;yy+=y0
    across=(xx-origin[0])*axis[1]-(yy-origin[1])*axis[0]
    if not np.any((np.abs(across)>contact.get('edge_contact_width_px',1)+1)&(strict[yy,xx]>0)):return None
    # Remove the persistent main interval only in this reachability probe. An
    # extra interval with ANY independent exterior source support must abstain,
    # even when that support is short, weak, or on the same electrical net.
    removed=set(map(tuple,main_pixels));h,w=raw.shape
    for x,y in extras:
        for a in range(x-1,x+2):
            for b in range(y-1,y+2):
                if not(0<=a<w and 0<=b<h) or (a,b) in extras or (a,b) in removed or not raw[b,a] or forbidden[b,a]:continue
                if not(x0<=a<x1 and y0<=b<y1):return None
                # Body traversal cannot supply an exterior route on behalf of
                # the fringe; only the classified interval itself is a seed.
    # The persistent main interval independently reaches the observed outside
    # stem at the stricter threshold with the fringe absent by construction.
    normal=np.array([-axis[1],axis[0]]);corners=np.array([[x0,y0],[x1,y1],origin-normal*half,origin+normal*half,origin+axis*(reach+2)-normal*half,origin+axis*(reach+2)+normal*half])
    a,b=np.maximum(0,np.floor(corners.min(axis=0)).astype(int)-1);c,d=np.minimum([w,h],np.ceil(corners.max(axis=0)).astype(int)+2)
    yy,xx=np.mgrid[b:d,a:c];along=(xx-origin[0])*axis[0]+(yy-origin[1])*axis[1];trans=(xx-origin[0])*normal[0]+(yy-origin[1])*normal[1]
    inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1);tube=(along>=-math.sqrt(2))&(along<=reach+math.sqrt(2))&(np.abs(trans)<=half)
    native=((strict[b:d,a:c]>0)&(inside|tube)&(forbidden[b:d,a:c]==0)).astype(np.uint8)
    _,components=cv2.connectedComponents(native,connectivity=8);ids={int(components[y-b,x-a]) for x,y in main_pixels if a<=x<c and b<=y<d}-{0}
    if len(ids)!=1:return None
    component=components==next(iter(ids));main_body=component&inside
    if not np.any(main_body&(np.abs(trans)>contact.get('edge_contact_width_px',1)+1)):return None
    by,bx=np.nonzero(main_body);bx+=a;by+=b
    main_body_owners=set(map(int,labels[by-y0,bx-x0]))-{0}
    if main_body_owners!=owners:return None
    # Tie the weak fringe to the SAME persistent entry source component. A
    # separate interior glyph cannot borrow a valid body's proof elsewhere.
    weak_native=((raw[b:d,a:c]>0)&(inside|tube)&(forbidden[b:d,a:c]==0)).astype(np.uint8)
    _,weak_components=cv2.connectedComponents(weak_native,connectivity=8)
    weak_main={int(weak_components[y-b,x-a]) for x,y in main_pixels if a<=x<c and b<=y<d}-{0}
    weak_extra={int(weak_components[y-b,x-a]) for x,y in extras}-{0}
    if len(weak_main)!=1 or weak_extra!=weak_main:return None
    stem=component&(~inside)&(along>=reach-1)&(occupied[b:d,a:c]>0)&(regions[b:d,a:c]==contact['ink_region'])
    if not stem.any():return None
    return {'classification':'owned_body_only_weak_fringe','source_pixels_px':[list(p) for p in sorted(extras)],
        'body_component_pixels':int(body.sum()),'strict_nonstraight_body_support':True,
        'main_entry_body_component_correspondence':'same_weak_source_component_and_same_strict_main_owned_body',
        'independent_exterior_source_pixels':0,'main_reaches_strict_occupied_outer_stem':True,
        'source_pixels_removed':0,'net_regions_unioned':0}


def _native_lead_group(members,raw,raw2,occupied,regions,forbidden,body_box=None):
    """Prove duplicate contact bands on one native oblique lead; inventory only."""
    guides=[c for c in members if c.get('contact_axis')=='continuous_source_collar' and c.get('direction_from_body')]
    if not guides:return None
    axis=np.asarray(guides[0]['direction_from_body'],float);length=float(np.linalg.norm(axis))
    if not length:return None
    axis/=length
    if any(float(axis@np.asarray(c['direction_from_body']))<.98 for c in guides):return None
    normal=np.asarray([-axis[1],axis[0]]);side=members[0]['side'];region=members[0]['ink_region'];h,w=raw.shape
    # Point anchors have finite observed contact bands. Every band must witness
    # the same source stroke, rather than relying on anchor proximity or net ID.
    band_witnesses=[]
    for member in members:
        x,y=map(int,member['point_px']);width=int(member.get('edge_contact_width_px',1));low=-((width-1)//2)
        pixels=[(x,y+t) if side in ('left','right') else (x+t,y) for t in range(low,low+width)]
        # Two half-cell raster quantizations have a fixed Minkowski footprint:
        # at most one native pixel per axis, never an expanding nearest search.
        footprint={(a+dx,b+dy) for a,b in pixels for dx in (-1,0,1) for dy in (-1,0,1)
            if 0<=a+dx<w and 0<=b+dy<h}
        if any(forbidden[b,a] and (raw[b,a] or raw2[b,a]) for a,b in footprint):return None
        shared=[(a,b) for a,b in sorted(footprint) if raw[b,a] and raw2[b,a] and occupied[b,a] and int(regions[b,a])==region and not forbidden[b,a]]
        if not shared:return None
        band_witnesses.append({'contact_point_px':[x,y],'band_width_px':width,
            'quantization_footprint_max_offset_xy_px':[1,1],
            'source_footprint_pixels_px':[[a,b] for a,b in sorted(footprint) if raw[b,a] or raw2[b,a]],
            'shared_native_occupied_pixels_px':[list(p) for p in shared]})
    representatives=[c for c in members if raw[c['point_px'][1],c['point_px'][0]] and raw2[c['point_px'][1],c['point_px'][0]] and occupied[c['point_px'][1],c['point_px'][0]] and not forbidden[c['point_px'][1],c['point_px'][0]]]
    if not representatives:return None
    chosen=max(representatives,key=lambda c:(c.get('contact_axis')=='continuous_source_collar',c.get('consecutive_wire_support_px',0),tuple(c['point_px'])))
    origin=np.asarray(chosen['point_px'],float);scale=max(1.,float(np.median([c.get('edge_contact_width_px',1) for c in members])))
    # Inspect the entire already measured collar span, rather than stopping
    # before a shallow fork becomes distinguishable from one thick band.
    reach=max(4,int(math.floor(min(c.get('consecutive_wire_support_px',0) for c in guides))));half=max(4.,4*scale)
    if any(c.get('consecutive_wire_support_px',0)<reach for c in guides):return None
    sections=[];widths=[];centres=[];witness_pixels=set()
    for depth in range(reach+1):
        center=origin+axis*depth
        samples=np.unique(np.rint(center+np.arange(-half,half+.25,.5)[:,None]*normal).astype(int),axis=0)
        projection=(samples-center)@normal;order=np.argsort(projection);samples=samples[order];projection=projection[order]
        if not all(0<=x<w and 0<=y<h for x,y in samples):return None
        if any(forbidden[y,x] for x,y in samples):return None
        layers=[];native_sets=[];interval_classifications=[]
        strict_values=raw2[samples[:,1],samples[:,0]].astype(bool)
        strict_changes=np.flatnonzero(np.diff(np.pad(strict_values.astype(np.int8),1)));strict_runs=list(zip(strict_changes[::2],strict_changes[1::2]))
        if len(strict_runs)!=1:return None
        strict_lo,strict_hi=strict_runs[0];main_pixels=samples[strict_lo:strict_hi]
        for layer in (raw,raw2):
            values=layer[samples[:,1],samples[:,0]].astype(bool)
            changes=np.flatnonzero(np.diff(np.pad(values.astype(np.int8),1)));runs=list(zip(changes[::2],changes[1::2]))
            if not runs or runs[0][0]==0 or runs[-1][1]==len(values):return None
            if len(runs)>1:
                main=[run for run in runs if run[0]<=strict_lo and strict_hi<=run[1]]
                if len(main)!=1:return None
                extras=[samples[lo:hi].tolist() for lo,hi in runs if (lo,hi)!=main[0]]
                classification=_owned_body_intervals(extras,main_pixels.tolist(),raw,raw2,body_box,chosen,occupied,regions,forbidden,origin,axis,reach,half)
                if classification is None:return None
                interval_classifications.append(classification);lo,hi=main[0]
            else:lo,hi=runs[0]
            points=samples[lo:hi];support=occupied[points[:,1],points[:,0]]>0
            seen=set(map(int,regions[points[:,1],points[:,0]]))-{0}
            if not support.any() or seen!={region}:return None
            # Source background must bound the filled band on BOTH sides.
            if not interval_classifications and (values[:lo].any() or values[hi:].any()):return None
            layers.append({'first_offset_px':float(projection[lo]),'last_offset_px':float(projection[hi-1]),'filled_sample_count':int(hi-lo),'background_samples_each_side':[int(lo),int(len(values)-hi)]})
            native_sets.append({tuple(p) for p in points})
        if not native_sets[0]&native_sets[1]:return None
        common=native_sets[0]&native_sets[1];witness_pixels.update(common)
        widths.append(layers[0]['last_offset_px']-layers[0]['first_offset_px']+1)
        centres.append((layers[0]['last_offset_px']+layers[0]['first_offset_px'])/2)
        sections.append({'outward_depth_px':depth,'source_thresholds':[48,60],'layers':layers,
            'interval_domain_classifications':interval_classifications})
    if max(widths)>2*min(widths) or max(centres)-min(centres)>scale:return None
    # A shallow fork can still look like one filled transverse band. Check the
    # native tube's interior skeleton as well, with its sampling caps excluded.
    corners=np.asarray([origin-normal*half,origin+normal*half,
        origin+axis*reach+normal*half,origin+axis*reach-normal*half])
    left,top=np.floor(corners.min(axis=0)).astype(int)-1
    right,bottom=np.ceil(corners.max(axis=0)).astype(int)+2
    if not (0<=left<right<=w and 0<=top<bottom<=h):return None
    tube=np.zeros((bottom-top,right-left),np.uint8)
    cv2.fillConvexPoly(tube,np.rint(corners-[left,top]).astype(np.int32),1)
    for source in (raw,raw2):
        native=tube&source[top:bottom,left:right]
        skeleton,converged,_=_thin(native)
        if not converged:return None
        _,native_regions=cv2.connectedComponents(native,8)
        for point,neighbours in _adjacency(skeleton,native_regions).items():
            depth=float((np.asarray(point)+[left,top]-origin)@axis)
            if len(neighbours)>2 and .5*scale<depth<reach-.5*scale:return None
    # Every original band must belong to the measured source-band envelope at
    # the boundary, not merely be connected elsewhere on the same electrical net.
    first=sections[0]['layers'];low=max(x['first_offset_px'] for x in first)-.75;high=min(x['last_offset_px'] for x in first)+.75
    body_fringe={tuple(p) for section in sections for classification in section['interval_domain_classifications']
        for p in classification['source_pixels_px']}
    for witness in band_witnesses:
        if not any(low<=float((np.asarray(p)-origin)@normal)<=high for p in witness['shared_native_occupied_pixels_px']):return None
        if not all(tuple(p) in body_fringe or low<=float((np.asarray(p)-origin)@normal)<=high for p in witness['source_footprint_pixels_px']):return None
    proof={'policy':'native_single_lead_cross_sections','operation':'existing_contact_inventory_collapse_only',
        'members':[dict(c) for c in members],'representative_anchor_px':[float(v)+.5 for v in chosen['point_px']],
        'measured_outward_unit_vector':axis.tolist(),'same_ink_region':int(region),'band_witnesses':band_witnesses,
        'cross_sections':sections,'native_witness_pixels_sha256':hashlib.sha256(canonical([list(map(int,p)) for p in sorted(witness_pixels)]).encode()).hexdigest(),
        'bilateral_background':True,'threshold_partition_agreement':True,'native_interior_branch_check':'no_third_arm_at_both_thresholds','mask_pixels_added':0,'net_regions_unioned':0,
        'anchor_invented':False,'verified':False}
    return dict(chosen,contact_group_evidence=proof)


def _group_native_lead_contacts(contacts,raw,raw2,occupied,regions,forbidden,body_box=None):
    """Collapse only after a source-geometry proof; never force expected arity."""
    groups={};result=[]
    for contact in contacts:
        side=contact['side']
        if side not in DIRECTIONS:result.append(contact);continue
        plane=contact['point_px'][0 if side in ('left','right') else 1]
        groups.setdefault((side,plane,contact['ink_region']),[]).append(contact)
    for members in groups.values():
        merged=_native_lead_group(members,raw,raw2,occupied,regions,forbidden,body_box) if len(members)>1 else None
        if merged is None:result.extend(members)
        else:result.append(merged)
    return result


def _group_same_lead(contacts,gray,probability,high,screened,regions,forbidden,*,policy='same_lead_v1'):
    """Collapse only threshold-split bands of one filled native physical lead."""
    groups={};result=[];h,w=gray.shape
    for contact in contacts:
        side=contact['side'];point=contact['point_px']
        if side not in DIRECTIONS:result.append(contact);continue
        axis=0 if side in ('left','right') else 1
        groups.setdefault((side,point[axis],contact['ink_region']),[]).append(contact)
    for (side,plane,region),members in groups.items():
        if len(members)<2:result.extend(members);continue
        along=side in ('left','right');cross=1 if along else 0
        low=min(c['point_px'][cross]-(c['edge_contact_width_px']-1)//2 for c in members)
        high_cross=max(c['point_px'][cross]-(c['edge_contact_width_px']-1)//2+c['edge_contact_width_px'] for c in members)
        if high_cross-low>12:result.extend(members);continue
        dx,dy=map(int,DIRECTIONS[side]);points=[]
        for step in range(6):
            for t in range(low,high_cross):points.append((plane+dx*step,t) if along else (t,plane+dy*step))
        if not all(2<=x<w-2 and 2<=y<h-2 for x,y in points):result.extend(members);continue
        if policy=='same_lead_v2':
            proof=_source_fill_group_proof(members,side,plane,low,high_cross,region,gray,probability,high,screened,regions,forbidden)
            if proof is None:result.extend(members);continue
            centre=(low+high_cross-1)/2;chosen=min(members,key=lambda c:(abs(c['point_px'][cross]-centre),tuple(c['point_px'])))
            merged=dict(chosen,edge_contact_width_px=high_cross-low)
            merged['contact_group_evidence']=dict(proof,members=[dict(c) for c in members],representative_anchor_px=[v+.5 for v in chosen['point_px']],source_stroke_center_px=[plane+.5,centre+.5] if along else [centre+.5,plane+.5])
            result.append(merged);continue
        xs,ys=zip(*points);left,top,right,bottom=max(0,min(xs)-12),max(0,min(ys)-12),min(w,max(xs)+13),min(h,max(ys)+13)
        roi=gray[top:bottom,left:right].astype(float);paper_pixels=roi[roi>=np.percentile(roi,60)];paper=float(np.median(paper_pixels));noise=1.4826*float(np.median(np.abs(paper_pixels-paper)))
        values=[float(gray[c['point_px'][1]+dy*k,c['point_px'][0]+dx*k]) for c in members for k in range(6)]
        contrast=paper-float(np.median(values));cutoff=paper-.65*contrast
        if contrast<max(12.,6*noise):result.extend(members);continue
        valid=all(not forbidden[y,x] and gray[y,x]<=cutoff and probability[y,x]>=(2./3.)*high and (probability[y,x]<high or screened[y,x] and int(regions[y,x])==region) for x,y in points)
        # A transverse branch or a wider neighbouring stroke cannot hide inside
        # the grouped lead envelope. No cross-region equivalence is introduced.
        for step in range(6):
            for t in (low-2,high_cross+1):
                x,y=(plane+dx*step,t) if along else (t,plane+dy*step)
                if gray[y,x]<=paper-.35*contrast:valid=False
        if not valid:result.extend(members);continue
        centre=(low+high_cross-1)/2
        chosen=min(members,key=lambda c:(abs(c['point_px'][cross]-centre),tuple(c['point_px'])))
        merged=dict(chosen,edge_contact_width_px=high_cross-low)
        merged['contact_group_evidence']={'policy':'same_lead_v1','members':[dict(c) for c in members],
            'representative_anchor_px':[v+.5 for v in chosen['point_px']],
            'source_stroke_center_px':[plane+.5,centre+.5] if along else [centre+.5,plane+.5],
            'same_ink_region':region,'cross_sections':6,'observed_width_px':high_cross-low,
            'source_relative_thresholds_gray':[paper-.35*contrast,cutoff],
            'auxiliary_probability_floor':(2./3.)*high,'mask_pixels_added':0,'net_regions_unioned':0,'verified':False}
        result.append(merged)
    return result


def _existing_body_local_proof(gray,box,contact,probability,high,screened,regions,forbidden,*,policy='body_local_v1'):
    """Refine an existing high anchor using independent native body evidence."""
    side=contact['side']
    if side not in DIRECTIONS:return None
    h,w=gray.shape;x0,y0,x1,y1=box;left,top,right,bottom=max(0,x0-24),max(0,y0-24),min(w,x1+24),min(h,y1+24)
    roi=gray[top:bottom,left:right].astype(float);paper_pixels=roi[roi>=np.percentile(roi,60)];paper=float(np.median(paper_pixels));noise=1.4826*float(np.median(np.abs(paper_pixels-paper)))
    source=np.zeros_like(screened);source[top:bottom,left:right]=(roi<=paper-max(12.,6*noise)).astype(np.uint8)
    sh=cv2.morphologyEx(source,cv2.MORPH_OPEN,np.ones((1,9),np.uint8));sv=cv2.morphologyEx(source,cv2.MORPH_OPEN,np.ones((9,1),np.uint8));_,labels=cv2.connectedComponents(source,connectivity=8)
    native=_contact(side,box,sh,sv,labels)
    if len(native)!=1:return None
    native=native[0];width=native['edge_contact_width_px'];dx,dy=map(int,DIRECTIONS[side]);x,y=map(int,contact['point_px']);nx,ny=map(int,native['point_px'])
    if width>12 or abs((x-nx)*dx+(y-ny)*dy)>2 or abs((x-nx)*dy-(y-ny)*dx)>width/2+1:return None
    proof=_contrast_body_support(gray,box,native,source,departure_fallback=True,forbidden=forbidden,body_local=True,interval_agreement=policy=='body_local_v2')
    if proof is None:return None
    tx,ty=[int(v-.5) for v in proof['terminal']['point_px']]
    distance=(nx-tx)*dx+(ny-ty)*dy
    if not 0<=distance<=12:return None
    cutoff=min(proof['terminal']['contrast_support']['relative_thresholds_gray']);region=contact['ink_region'];weak=[]
    for step in range(distance):
        a,b=nx-dx*step,ny-dy*step
        if forbidden[b,a] or gray[b,a]>cutoff or probability[b,a]<(2./3.)*high:return None
        if probability[b,a]<high:weak.append([a,b])
    if len(weak)>8:return None
    for step in range(6):
        a,b=x+dx*step,y+dy*step
        if not (0<=a<w and 0<=b<h) or forbidden[b,a] or gray[b,a]>cutoff or probability[b,a]<high or not screened[b,a] or int(regions[b,a])!=region:return None
        # Retained anchor and source centre must belong to one filled lead.
        points=[(a,t) for t in range(min(y,ny),max(y,ny)+1)] if dx else [(t,b) for t in range(min(x,nx),max(x,nx)+1)]
        if any(forbidden[v,u] or gray[v,u]>cutoff or probability[v,u]<(2./3.)*high for u,v in points):return None
    evidence={'policy':policy,'kind':'retained_high_contact_native_body_refinement','high_anchor_px':[x+.5,y+.5],'source_face_contact_px':[nx+.5,ny+.5],'body_terminal_px':proof['terminal']['point_px'],'weak_path_px':[[a+.5,b+.5] for a,b in weak],'high_threshold':high,'auxiliary_probability_floor':(2./3.)*high,'mask_pixels_added':0,'verified':False}
    if proof['terminal'].get('transition_uncertainty'):
        evidence['path_reference_px']=evidence.pop('body_terminal_px');evidence['terminal_uncertainty']=proof['terminal']['transition_uncertainty']
    proof['terminal']=dict(proof['terminal'],wire_anchor_px=[x+.5,y+.5],pin_path_evidence=evidence)
    return proof


def _supported_single_gap_links(screened,original,probability,raw,raw2,gray,regions,bodymask,cut,events,spans,by_region):
    """Certify graph edges across one observed weak pixel; never alter mask pixels."""
    blocked=(bodymask|cut).copy();h,w=screened.shape;radius=4
    for span in spans:
        x0,y0,x1,y1=_box(span['bbox_px'],w,h);blocked[y0:y1,x0:x1]=1
    for event in events:
        if event.get('proof_box_px'):
            x0,y0,x1,y1=event['proof_box_px'];blocked[y0:y1,x0:x1]=1
        else:
            x,y=[int(v-.5) for v in event['point_px']]
            cv2.circle(blocked,(x,y),max(4,event.get('cut_radius_px',4)),1,-1)
    weak=(screened==0)&(original==0)&(probability>=.5)&(probability<.75)&(raw!=0)&(raw2!=0)
    horizontal=np.zeros_like(weak);vertical=np.zeros_like(weak)
    horizontal[1:-1,1:-1]=(screened[1:-1,:-2]!=0)&(screened[1:-1,2:]!=0)&(screened[:-2,1:-1]==0)&(screened[2:,1:-1]==0)
    vertical[1:-1,1:-1]=(screened[:-2,1:-1]!=0)&(screened[2:,1:-1]!=0)&(screened[1:-1,:-2]==0)&(screened[1:-1,2:]==0)
    ys,xs=np.nonzero(weak&(horizontal^vertical));links=[];records=[]
    for x,y in zip(xs.tolist(),ys.tolist()):
        if not radius<=x<w-radius or not radius<=y<h-radius:continue
        area=(slice(y-radius,y+radius+1),slice(x-radius,x+radius+1))
        if blocked[area].any():continue
        neighbors=[(x-1,y),(x+1,y)] if horizontal[y,x] else [(x,y-1),(x,y+1)]
        groups=[int(regions[b,a]) for a,b in neighbors]
        if 0 in groups or groups[0]==groups[1]:continue
        if set(map(int,regions[area].ravel()))-{0} != set(groups):continue
        supported=True
        for ink in (raw,raw2):
            local=ink[area];_,lab=cv2.connectedComponents(local,connectivity=8)
            center=int(lab[radius,radius])
            if not center or any(not ink[b,a] for a,b in neighbors):supported=False;break
            component=(lab==center).astype(np.uint8);boundary=component.copy();boundary[1:-1,1:-1]=0
            arms=cv2.connectedComponents(boundary,connectivity=8)[0]-1
            if arms!=2 or (local & (1-component)).any():supported=False;break
        if not supported:continue
        ids=sorted({f for group in groups for f in by_region.get(group,[])})
        if not all(by_region.get(group) for group in groups):continue
        uid=_id('single-gap-',[x,y,ids])
        record={'id':uid,'point_px':[x+.5,y+.5],'neighbor_points_px':[[a+.5,b+.5] for a,b in neighbors],
                'fragment_ids':ids,'probability':float(probability[y,x]),'high_threshold':.75,'auxiliary_probability_floor':.5,
                'source_gray':int(gray[y,x]),'rgb_threshold_offsets':[48,60],'collar_radius_px':radius,
                'evidence':['one_original_background_pixel','two_opposite_high_mask_neighbors_in_distinct_regions',
                            'two_rgb_supported_arms_at_both_thresholds','no_body_text_or_intersection_roi'],
                'mask_pixels_added':0,'state':'supported_graph_link','verified':False}
        records.append(record);links.append({'id':uid,'fragment_ids':ids,'state':'supported','evidence_refs':[uid,'supported_single_gap_v1']})
    return links,records


def _probability_pin_contacts(gray,box,probability,high,screened,regions,forbidden,*,policy='probability_pin_path_v1'):
    """New pin evidence from a bounded observed source/probability lead path.

    Existing hard-mask contacts remain untouched. No mask or region is modified.
    A terminal is a native RGB body transition; an anchor is an existing high
    region pixel, never a box-face snap or an interpolated conductor coordinate.
    """
    h,w=gray.shape;x0,y0,x1,y1=box;pad=24
    left,top,right,bottom=max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad)
    roi=gray[top:bottom,left:right].astype(float)
    paper_pixels=roi[roi>=np.percentile(roi,60)];paper=float(np.median(paper_pixels))
    noise=1.4826*float(np.median(np.abs(paper_pixels-paper)))
    source=np.zeros_like(screened);source[top:bottom,left:right]=(roi<=paper-max(12.,6*noise)).astype(np.uint8)
    sh=cv2.morphologyEx(source,cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
    sv=cv2.morphologyEx(source,cv2.MORPH_OPEN,np.ones((9,1),np.uint8))
    _,source_regions=cv2.connectedComponents(source,connectivity=8)
    contacts=[];proofs={};low=(2./3.)*high
    for side in DIRECTIONS:
        found=_contact(side,box,sh,sv,source_regions)
        if len(found)!=1:continue
        native=found[0];width=native['edge_contact_width_px']
        if width>12:continue
        proof=_contrast_body_support(gray,box,native,source,departure_fallback=policy in ('probability_pin_path_v2','body_local_v1','body_local_v2'),forbidden=forbidden,body_local=policy in ('body_local_v1','body_local_v2'),interval_agreement=policy=='body_local_v2')
        if proof is None:continue
        terminal=proof['terminal']['point_px'];face=native['point_px']
        if math.dist([v-.5 for v in terminal],face)>12:continue
        dx,dy=map(int,DIRECTIONS[side]);tx,ty=[int(v-.5) for v in terminal]
        thresholds=proof['terminal']['contrast_support']['relative_thresholds_gray']
        max_gray=min(thresholds);weak=[];anchor=None;region=None;path=[];valid=True
        for distance in range(1,15):
            x,y=tx+dx*distance,ty+dy*distance
            if not (0<=x<w and 0<=y<h) or forbidden[y,x] or gray[y,x]>max_gray:valid=False;break
            if probability[y,x]>=high and (policy not in ('body_local_v1','body_local_v2') or screened[y,x]):
                if (not weak and policy not in ('body_local_v1','body_local_v2')) or len(weak)>8 or not screened[y,x] or not regions[y,x]:valid=False;break
                anchor=[x,y];region=int(regions[y,x]);break
            if probability[y,x]<low or len(weak)>=8:valid=False;break
            weak.append([x,y]);path.append([x,y])
        if not valid or anchor is None:continue
        for step in range(6):
            x,y=anchor[0]+dx*step,anchor[1]+dy*step
            if not (0<=x<w and 0<=y<h) or forbidden[y,x] or probability[y,x]<high or not screened[y,x] or int(regions[y,x])!=region or gray[y,x]>max_gray:
                valid=False;break
            path.append([x,y])
        if not valid:continue
        # Every lead cross-section stays narrow until the body transition. No
        # branch or parallel competing high-region may be concealed by a path.
        half=max(2,int(math.ceil(width/2))+2)
        for x,y in path:
            # Exclusions cover the observed stroke footprint, not only its
            # centreline, so adjacent text cannot share a certified lead band.
            stroke_half=max(0,int(math.ceil((width-1)/2)))
            if dx:
                excluded=forbidden[max(0,y-stroke_half):min(h,y+stroke_half+1),x:x+1]
            else:excluded=forbidden[y:y+1,max(0,x-stroke_half):min(w,x+stroke_half+1)]
            if excluded.any():valid=False;break
            extent=[]
            for sign in (-1,1):
                n=0
                for k in range(1,15):
                    a,b=x+dy*k*sign,y-dx*k*sign
                    if not (0<=a<w and 0<=b<h) or gray[b,a]>max_gray:break
                    n=k
                extent.append(n)
            if sum(extent)+1>width+2:valid=False;break
            if dx:
                a=regions[max(0,y-half):min(h,y+half+1),x:x+1]
            else:a=regions[y:y+1,max(0,x-half):min(w,x+half+1)]
            if set(map(int,a.ravel()))-{0,region}:valid=False;break
        if not valid:continue
        # The weak corridor and high evidence are retained as graph observations.
        contact=dict(native,point_px=anchor,ink_region=region,boundary_gap_px=None,
                     consecutive_wire_support_px=6,contact_policy=policy)
        evidence={'policy':policy,'source_face_contact_px':[v+.5 for v in face],
                  'body_terminal_px':terminal,'high_anchor_px':[v+.5 for v in anchor],
                  'weak_path_px':[[a+.5,b+.5] for a,b in weak],
                  'weak_probabilities':[float(probability[b,a]) for a,b in weak],
                  'high_support_path_px':[[a+.5,b+.5] for a,b in path[len(weak):]],
                  'high_threshold':float(high),'auxiliary_probability_floor':float(low),
                  'max_weak_pixels':8,'minimum_high_pixels':6,'body_inset_limit_px':12,
                  'mask_pixels_added':0,'source_contrast':proof['terminal']['contrast_support'],'verified':False}
        if proof['terminal'].get('transition_uncertainty'):
            evidence['path_reference_px']=evidence.pop('body_terminal_px');evidence['terminal_uncertainty']=proof['terminal']['transition_uncertainty']
        contact['pin_path_evidence']=evidence
        witness=dict(proof['terminal'],wire_anchor_px=[v+.5 for v in anchor],pin_path_evidence=evidence)
        contacts.append(contact);proofs[tuple(anchor)]={'terminal':witness,'body':proof['body']}
    return contacts,proofs


def _source_outline_fringe(image,parts,boxes,nested,base_body,foreground):
    """Typed closed triangles plus native AA evidence; never expand a box blindly."""
    rgb=np.asarray(image);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    paper=float(np.percentile(gray,90));extra=np.zeros_like(base_body);records=[]
    for part in parts:
        if part['class_name']!='op_amp' or part['id'] in nested:continue
        uid=part['id'];x0,y0,x1,y1=boxes[uid];a,b,c,d=max(0,x0-8),max(0,y0-8),min(w,x1+8),min(h,y1+8)
        roi=gray[b:d,a:c];native_rgb=rgb[b:d,a:c].astype(float);area=(x1-x0)*(y1-y0)
        record={'component_id':uid,'policy':'source_outline_fringe_v1','observation_roi_px':[a,b,c,d],'state':'abstained'};records.append(record)
        if area<=0 or area>65536:record['reason']='body_roi_budget';continue
        shapes=[];cores=[]
        for darkness in (48,60):
            strong=(roi<=paper-darkness).astype(np.uint8)
            contours,hierarchy=cv2.findContours(strong,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);choices=[]
            if hierarchy is not None:
                for index,contour in enumerate(contours):
                    if hierarchy[0,index,3]<0:continue
                    size=cv2.contourArea(contour)
                    if not .15*area<=size<=.9*area:continue
                    hull=cv2.convexHull(contour);poly=cv2.approxPolyDP(hull,.035*cv2.arcLength(hull,True),True)
                    if len(poly)!=3:continue
                    pts=poly[:,0,:];center=pts.mean(axis=0)
                    if not (x0<=center[0]+a<x1 and y0<=center[1]+b<y1):continue
                    if np.any(pts[:,0]<=0) or np.any(pts[:,1]<=0) or np.any(pts[:,0]>=roi.shape[1]-1) or np.any(pts[:,1]>=roi.shape[0]-1):continue
                    fill=np.zeros_like(strong);cv2.drawContours(fill,[contour],-1,1,-1);choices.append((fill,poly))
            if len(choices)!=1:break
            fill,poly=choices[0];shapes.append(fill);cores.append(strong)
        if len(shapes)!=2:record['reason']='unique_closed_native_triangle_not_observed_twice';continue
        union=shapes[0]|shapes[1];intersection=shapes[0]&shapes[1];iou=float(intersection.sum()/max(1,union.sum()))
        if iou<.9:record.update(reason='native_outline_threshold_disagreement',interior_iou=iou);continue
        ring=cv2.dilate(union,np.ones((7,7),np.uint8))&(1-cv2.erode(intersection,np.ones((7,7),np.uint8)))
        core=(cores[0]|cores[1])&ring
        envelope=union|core
        near=cv2.dilate(core,np.ones((7,7),np.uint8))&(1-envelope)
        background=np.percentile(native_rgb.reshape(-1,3),90,axis=0)
        fringe=np.zeros_like(core)
        for yy,xx in zip(*np.nonzero(near & (roi<=paper-12))):
            ly,hy=max(0,yy-3),min(core.shape[0],yy+4);lx,hx=max(0,xx-3),min(core.shape[1],xx+4)
            cy,cx=np.nonzero(core[ly:hy,lx:hx]);cx=cx+lx;cy=cy+ly
            if not len(cx):continue
            distances=(cx-xx)**2+(cy-yy)**2;within=distances<=9
            if not within.any():continue
            # A multi-level AA boundary can itself cross the strong threshold.
            # Use the darkest local core witness, not the nearest fringe sample.
            candidates=native_rgb[cy[within],cx[within]]
            strength=np.sum((background-candidates)**2,axis=1)
            boundary=candidates[int(np.argmax(strength))];direction=background-boundary;denominator=float(direction@direction)
            if denominator<=0:continue
            alpha=float((background-native_rgb[yy,xx])@direction/denominator)
            residual=float(np.max(np.abs(native_rgb[yy,xx]-(background-alpha*direction))))
            if .08<=alpha<=.65 and residual<=12:fringe[yy,xx]=1
        envelope|=fringe
        # Preserve native external strokes, including near parallel wires. If
        # a very short lead cannot be distinguished from the body, abstain for
        # the entire component rather than erasing its possible contact.
        source=(roi<=paper-12).astype(np.uint8);external=source&(1-envelope)
        horizontal_external=cv2.morphologyEx(external,cv2.MORPH_OPEN,np.ones((1,3),np.uint8))
        vertical_external=cv2.morphologyEx(external,cv2.MORPH_OPEN,np.ones((3,1),np.uint8))
        directional=horizontal_external|vertical_external
        guard=cv2.dilate(directional,np.ones((7,7),np.uint8))&envelope
        # A native lead may enter an inset detector box before its body edge.
        # Protect its contiguous source run, rather than stopping at an arbitrary
        # dilation radius. Only external three-pixel axial witnesses can seed a
        # corridor; paper gaps terminate it and no conductor pixels are added.
        for transpose,seeds in ((False,horizontal_external),(True,vertical_external)):
            scan=source.T if transpose else source
            starts=seeds.T if transpose else seeds
            protected=guard.T if transpose else guard
            interior=union.T if transpose else union
            for row in range(scan.shape[0]):
                changes=np.flatnonzero(np.diff(np.pad(scan[row].astype(np.int8),1)))
                for lo,hi in zip(changes[::2],changes[1::2]):
                    if starts[row,lo:hi].any() and interior[row,lo:hi].any():protected[row,lo:hi]=1
        guard&=envelope
        count,_,stats,_=cv2.connectedComponentsWithStats(guard,connectivity=8)
        groups=sum(int(stats[i,cv2.CC_STAT_AREA])>0 for i in range(1,count))
        record.update(interior_iou=iou,body_core_pixels=int(core.sum()),aa_fringe_pixels=int(fringe.sum()),protected_entry_groups=groups,protected_body_pixels=int(guard.sum()))
        if groups<3:record['reason']='fewer_than_three_distinct_native_lead_corridors';continue
        proposed=envelope&(1-guard)
        # Overlapping component hypotheses cannot supply exclusive body identity.
        if any(other['id']!=uid and other['class_name']!='junction' and max(x0,boxes[other['id']][0])<min(x1,boxes[other['id']][2]) and max(y0,boxes[other['id']][1])<min(y1,boxes[other['id']][3]) for other in parts):
            record['reason']='overlapping_component_body_hypothesis';continue
        additions=proposed&(1-base_body[b:d,a:c]);extra[b:d,a:c]|=additions
        record.update(state='applied',reason='dual_closed_triangle_and_correlated_native_fringe_with_protected_leads',additional_exclusion_pixels=int(additions.sum()),removed_foreground_pixels=int((additions&foreground[b:d,a:c]).sum()),source_patch_sha256=hashlib.sha256(rgb[b:d,a:c].tobytes()).hexdigest(),exclusion_sha256=hashlib.sha256(additions.tobytes()).hexdigest(),verified=False)
    return extra,records


def observe_topology_geometry(image,inventory,*,canonical_rgb_sha256, mask_path=None, conductor_mask=None, conductor_evidence=None, mask_contact_policy="frozen", conductor_probability=None, conductor_gap_policy="frozen", terminal_evidence_policy="frozen", contact_group_policy="frozen", body_exclusion_policy="frozen", terminal_localization_policy="frozen", junction_convention="explicit_marks"):
    start=time.monotonic();peak=_guard(start)
    if image.mode!='RGB' or list(image.size)!=[inventory['observations']['image'][k]for k in ('width','height')]or hashlib.sha256(image.tobytes()).hexdigest()!=canonical_rgb_sha256:raise ValueError('Topology requires exact canonical RGB frame')
    w,h=image.size;parts=inventory['observations']['detections'];spans=inventory['observations']['ocr']
    if w*h>LIMITS['pixels']or len(parts)>LIMITS['components']:raise ValueError('Topology pixel/component budget exceeded')
    if conductor_mask is not None:
        if conductor_mask.shape!=(h,w) or conductor_mask.dtype!=np.uint8 or not np.isin(conductor_mask,[0,1]).all() or not conductor_evidence:
            raise ValueError('Conductor mask requires binary original-frame pixels and checked evidence')
    if mask_contact_policy not in ('frozen','native_short_lead_v1','contrast_supported_v2','contrast_fallback_v3','probability_pin_path_v1','probability_pin_path_v2'):raise ValueError('Unknown mask contact policy')
    if mask_contact_policy!='frozen' and conductor_mask is None:raise ValueError('Native mask contact policy requires explicit conductor mask')
    if mask_contact_policy in ('probability_pin_path_v1','probability_pin_path_v2'):
        if conductor_probability is None or conductor_probability.shape!=(h,w) or conductor_evidence.get('format')!='probability_u8' or not np.isfinite(conductor_probability).all():raise ValueError('Pin path policy requires original finite probability_u8 frame')
        if not np.array_equal(conductor_mask,conductor_probability>=conductor_evidence['threshold']):raise ValueError('Pin path probability and high-mask frame differ')
    if conductor_gap_policy not in ('frozen','supported_single_gap_v1'):raise ValueError('Unknown graph gap policy')
    if conductor_gap_policy!='frozen':
        if conductor_mask is None or conductor_probability is None or conductor_probability.shape!=(h,w) or not np.isfinite(conductor_probability).all():raise ValueError('Gap policy requires finite native probability frame')
        if conductor_evidence.get('format')!='probability_u8' or conductor_evidence.get('threshold')!=.75:raise ValueError('Single-gap policy requires unchanged high threshold0.75 probability_u8 evidence')
        if not np.array_equal(conductor_mask,conductor_probability>=.75):raise ValueError('Probability and high-mask frame differ')
    if terminal_evidence_policy not in ('frozen','body_local_v1','body_local_v2') or contact_group_policy not in ('frozen','same_lead_v1','same_lead_v2'):raise ValueError('Unknown contact evidence policy')
    if terminal_evidence_policy!='frozen' or contact_group_policy!='frozen':
        if conductor_mask is None or conductor_probability is None or conductor_probability.shape!=(h,w) or not np.isfinite(conductor_probability).all() or conductor_evidence.get('format')!='probability_u8':raise ValueError('Contact evidence policy requires checked native probabilities')
        if not np.array_equal(conductor_mask,conductor_probability>=conductor_evidence['threshold']):raise ValueError('Contact evidence probability and mask differ')
    if terminal_localization_policy not in ('frozen','source_body_v1'):raise ValueError('Unknown terminal localization policy')
    if terminal_localization_policy!='frozen' and conductor_mask is None:raise ValueError('Source body localization requires explicit conductor mask')
    if body_exclusion_policy not in ('frozen','source_outline_fringe_v1'):raise ValueError('Unknown body exclusion policy')
    if body_exclusion_policy!='frozen' and conductor_mask is None:raise ValueError('Source outline fringe policy requires explicit conductor foreground')
    if junction_convention not in CONVENTIONS:raise ValueError('Unknown junction convention')
    previous=cv2.getNumThreads();cv2.setNumThreads(0)
    try:
        gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY);paper=float(np.percentile(gray,90));raw=(gray<=paper-48).astype(np.uint8);raw2=(gray<=paper-60).astype(np.uint8)
        bodymask=np.zeros_like(raw);masks=[];boxes={};nested=set();models=[]
        for part in parts:
            b=_box(part['bbox_px'],w,h);boxes[part['id']]=b
            if any(other['id']!=part['id']and b[0]<=other['bbox_px'][0]<other['bbox_px'][2]<=b[2]and b[1]<=other['bbox_px'][1]<other['bbox_px'][3]<=b[3]for other in parts):nested.add(part['id'])
            if part['class_name']=='junction':continue
            if part['class_name']=='ground':b=_ground_body_box(tuple(b),raw,9);boxes[part['id']]=list(b)
            x0,y0,x1,y1=b
            # A competing enclosing detection is not a proven body boundary.
            # Retain its interior; the region blocks graph claims below, while
            # independently observed child contacts remain available.
            if part['id'] not in nested:bodymask[y0:y1,x0:x1]=1
            masks.append({'source_ref':part['id'],'region_px':list(b),'kind':'component_body_exclusion','state':'retained_ambiguous_parent'if part['id']in nested else'uncertain_exclusion'if part['class_name']not in TWO_PIN|{'ground','op_amp'}else'observed_body_boundary_hypothesis','applied':part['id']not in nested,'original_ink_pixels':int(raw[y0:y1,x0:x1].sum()),'original_roi_sha256':hashlib.sha256(raw[y0:y1,x0:x1].tobytes()).hexdigest()})
        # Preserve demonstrably straight component leads inside overinclusive
        # detection boxes; restore only exact foreground pixels before the body.
        lead_witnesses={}
        rawh=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
        rawv=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((9,1),np.uint8))
        _,rawregions=cv2.connectedComponents(raw,connectivity=8)
        for part in parts:
            uid=part['id']
            if uid in nested or part['class_name'] not in TWO_PIN|{'ground','op_amp'}:continue
            box=boxes[uid]
            for contact in [c for side in DIRECTIONS for c in _contact(side,box,rawh,rawv,rawregions)]:
                witness=_trace_body_terminal(raw,box,contact)
                other=_trace_body_terminal(raw2,box,contact)
                if witness is None or other is None or witness['point_px']!=other['point_px']:continue
                lead_witnesses[(uid,tuple(contact['point_px']))]=witness
                for x,y in witness['lead_pixels']:
                    # Never restore through a competing symbol exclusion.
                    if any(o['id']!=uid and o['class_name']!='junction' and boxes[o['id']][0]<=x<boxes[o['id']][2] and boxes[o['id']][1]<=y<boxes[o['id']][3] for o in parts):continue
                    bodymask[y,x]=0
        # Keep the original RGB observer's body mask fixed for crossing semantics.
        effective_bodymask=bodymask;body_fringe=np.zeros_like(bodymask);body_fringe_records=[]
        if body_exclusion_policy=='source_outline_fringe_v1':
            body_fringe,body_fringe_records=_source_outline_fringe(image,parts,boxes,nested,bodymask,conductor_mask)
            effective_bodymask=bodymask|body_fringe
        rgb_working=raw&(1-bodymask);rgb_working2=raw2&(1-bodymask)
        # Joint source/SAM wire observations retain the model mask unchanged.
        # Continuous native line tracks may corroborate gaps between supports;
        # blank source pixels and component bodies never supply connectivity.
        scene_mask,scene_evidence=conductor_appearance(image,conductor_mask if conductor_mask is not None else raw,effective_bodymask,[{'component_id':uid,'point_px':list(point),'body_entry_proof':proof['method']} for (uid,point),proof in lead_witnesses.items()])
        working=scene_mask&(1-effective_bodymask)
        working2=working.copy()
        # Geometry and occupancy share the same source-appearance population.
        # The RGB raster remains the authority for bodies and terminal marks.
        crossing_ink=rgb_working&(scene_mask if conductor_mask is not None else 1)
        _,labels_before=cv2.connectedComponents(crossing_ink,connectivity=8)
        horizontal=cv2.morphologyEx(crossing_ink,cv2.MORPH_OPEN,np.ones((1,9),np.uint8));vertical=cv2.morphologyEx(crossing_ink,cv2.MORPH_OPEN,np.ones((9,1),np.uint8))
        dot_records=_local_crossings(crossing_ink,horizontal,vertical,labels_before,bodymask,9);skeleton,converged,iterations=_thin(crossing_ink)
        adjacency=_adjacency(skeleton,labels_before);branch=np.zeros_like(raw)
        for (x,y),neighbors in adjacency.items():
            if len(neighbors)>=3:branch[y,x]=1
        count,_,stats,centers=cv2.connectedComponentsWithStats(branch,connectivity=8);events=[];cut=np.zeros_like(raw);dist=cv2.distanceTransform(crossing_ink,cv2.DIST_L2,3)
        for index in range(1,count):
            x,y=map(lambda v:int(round(v)),centers[index]);arms=collar_arms(crossing_ink,x,y,float(dist[y,x]))
            near=next((r for r in dot_records if abs(r['point_px'][0]-x)<=6 and abs(r['point_px'][1]-y)<=6),None)
            overlaps=[s['id']for s in spans if s['bbox_px'][0]<=x<=s['bbox_px'][2]and s['bbox_px'][1]<=y<=s['bbox_px'][3]]
            kind=junction_kind(arms,filled_dot=bool(near and near['kind']=='joined_filled_dot'),convention=junction_convention) if not overlaps else None
            event={'id':_id('event-',[x,y]),'point_px':[x+.5,y+.5],
                'state':'joined' if kind else 'unresolved','kind':kind or 'ambiguous_contact',
                'observed_arms':arms,'local_arm_partitions':[],
                'evidence_refs':['original_ink','eight_neighbor_stroke','rotation_independent_collar'],
                'drawing_convention':junction_convention,'conflicts':overlaps}
            # Explicit geometric bridge evidence has precedence over conventions.
            proof=verified_crossover(crossing_ink,bodymask,(x,y)) if len(arms)>=3 and not overlaps else None
            if proof is not None:
                bbox,keep,arm_points,cut_disk=proof;left,top,right,bottom=bbox
                cut[top:bottom,left:right]|=(~keep).astype(np.uint8)
                event['cut_disk_px']=cut_disk
                event.update(state='separate',kind='traced_overpass',arm_points_px=(arm_points+.5).round(3).tolist(),proof_box_px=list(bbox))
            elif kind is None:
                radius=max(3,int(math.ceil(float(dist[y,x])*2))+1)
                cv2.circle(cut,(x,y),radius,1,-1);event['cut_radius_px']=radius
            events.append(event)
        if mask_path is not None:
            from pathlib import Path
            stage_dir=Path(mask_path).parent
            for filename,bits in [('wire-input-mask.png',conductor_mask if conductor_mask is not None else raw),('source-conditioned-conductor-mask.png',scene_mask),('wire-after-body-mask.png',working),('component-body-exclusion-mask.png',effective_bodymask),('crossing-cut-mask.png',cut)]:
                cv2.imwrite(str(stage_dir/filename),bits.astype(np.uint8)*255)
            if body_exclusion_policy!='frozen':
                cv2.imwrite(str(stage_dir/'body-outline-fringe-mask.png'),body_fringe*255)
                cv2.imwrite(str(stage_dir/'frozen-body-exclusion-mask.png'),bodymask*255)
        screened=working&(1-cut);_,regions=cv2.connectedComponents(screened,connectivity=8);skeleton,converged2,iterations2=_thin(screened);fragments,by_region,endpoints,path_neighbors=_trace(skeleton,regions)
        # Wire-side contacts use predicted foreground, while body/terminal
        # witnesses and sign observations continue to use the original RGB.
        horizontal=cv2.morphologyEx(working,cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
        vertical=cv2.morphologyEx(working,cv2.MORPH_OPEN,np.ones((9,1),np.uint8))
        connections=[];issues=[];ports=[];terminals=[];unconfirmed_contacts=[]
        for region,ids in by_region.items():
            if len(ids)>1:connections.append({'id':_id('continuation-',ids),'fragment_ids':sorted(ids),'state':'supported','evidence_refs':['occupied_eight_neighbor_path:'+str(region)]})
        gap_records=[]
        if conductor_gap_policy=='supported_single_gap_v1':
            gap_links,gap_records=_supported_single_gap_links(screened,conductor_mask,conductor_probability,raw,raw2,gray,regions,bodymask,cut,events,spans,by_region)
            connections.extend(gap_links)
        def region_ids_near(x,y,radius=2):
            a=regions[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)];return sorted(set(map(int,a.ravel()))-{0})
        for event in events:
            x,y=[int(v-.5)for v in event['point_px']];near_regions=[]
            if event['state']=='separate':
                groups=[]
                for point in event['arm_points_px']:
                    a,b=[int(round(v-.5))for v in point];choices=region_ids_near(a,b,1);groups.append(choices[0]if len(choices)==1 else None)
                if len(groups)==4 and all(x is not None for x in groups):
                    event['local_arm_partitions']=[[0,1],[2,3]]
                    event['bound_arm_regions']=groups
                    for pair in ((0,1),(2,3)):
                        ids=sorted({f for i in pair for f in by_region.get(groups[i],[])})
                        if len(ids)>1:connections.append({'id':_id('bridge-pair-',ids),'fragment_ids':ids,'state':'supported','evidence_refs':[event['id'],'source_traced_curve_four_arms_clear_collar']})
                else:event['state']='unresolved';event['conflicts'].append('arm_fragment_binding_not_unique')
            if event['state']=='unresolved':
                radius=event.get('cut_radius_px',8)+2;a=regions[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)];near_regions=sorted(set(map(int,a.ravel()))-{0});ids=sorted({f for r in near_regions for f in by_region.get(r,[])})
                if len(ids)>1:connections.append({'id':event['id']+':alternatives','fragment_ids':ids,'state':'unresolved','evidence_refs':[event['id']]})
                issues.append({'id':event['id']+':unresolved','fragment_ids':ids,'blocking':True,'affects_equivalence':True,'reason':'join_vs_separate_not_observed'})
        diag=_diagonal_masks(screened,11);h2=cv2.morphologyEx(working2,cv2.MORPH_OPEN,np.ones((1,9),np.uint8));v2=cv2.morphologyEx(working2,cv2.MORPH_OPEN,np.ones((9,1),np.uint8));_,labels2=cv2.connectedComponents(working2,connectivity=8)
        contact_points=[];owned_contact_leads={};component_lead_endpoints=[]
        for part in parts:
            uid=part['id'];kind=part['class_name'];box=boxes[uid]
            if kind=='junction':models.append({'component_id':uid,'terminal_model':'junction_marker','expected_arity':0,'state':'supported'});continue
            contacts=[c for side in DIRECTIONS for c in _contact(side,box,horizontal,vertical,regions)]
            if mask_contact_policy in ('native_short_lead_v1','contrast_supported_v2','contrast_fallback_v3'):
                native=[c for side in DIRECTIONS for c in _native_short_contacts(side,box,screened,regions,raw,require_body=mask_contact_policy not in ('contrast_supported_v2','contrast_fallback_v3'))]
                contacts.extend(c for c in native if not any(c['side']==p['side'] and math.dist(c['point_px'],p['point_px'])<=2 for p in contacts))
            if kind in TWO_PIN:
                contacts.extend(collar_contacts(box,working,regions,raw))
            elif not contacts:contacts=_diagonal_contacts(tuple(box),regions,diag,single=kind=='ground')
            compare=[c for side in DIRECTIONS for c in _contact(side,box,h2,v2,labels2)]
            if mask_contact_policy in ('native_short_lead_v1','contrast_supported_v2','contrast_fallback_v3'):
                native=[c for side in DIRECTIONS for c in _native_short_contacts(side,box,screened,regions,raw2,require_body=mask_contact_policy not in ('contrast_supported_v2','contrast_fallback_v3'))]
                compare.extend(c for c in native if not any(c['side']==p['side'] and math.dist(c['point_px'],p['point_px'])<=2 for p in compare))
            if kind in TWO_PIN:compare.extend(collar_contacts(box,working2,labels2,raw2))
            elif not compare:compare=_diagonal_contacts(tuple(box),labels2,_diagonal_masks(working2,11),single=kind=='ground')
            # Native contact identity is established before inventory stability
            # or arity. This consumes no probabilities and never changes a net.
            if uid not in nested:
                native_forbidden=cut.copy()
                for other in parts:
                    if other['id']!=uid:
                        a,b,c,d=boxes[other['id']];native_forbidden[b:d,a:c]=1
                for span in spans:
                    a,b,c,d=_box(span['bbox_px'],w,h);native_forbidden[b:d,a:c]=1
                for event in events:
                    if event.get('proof_box_px'):
                        a,b,c,d=event['proof_box_px'];native_forbidden[b:d,a:c]=1
                    else:
                        a,b=[int(v-.5) for v in event['point_px']]
                        cv2.circle(native_forbidden,(a,b),max(2,event.get('cut_radius_px',2)),1,-1)
                contacts=_group_native_lead_contacts(contacts,raw,raw2,screened,regions,native_forbidden,box)
                compare=_group_native_lead_contacts(compare,raw,raw2,working2,labels2,native_forbidden,box)
            stable=sorted((c['side'],c['point_px'])for c in contacts)==sorted((c['side'],c['point_px'])for c in compare)
            pin_path_proofs={}
            if (mask_contact_policy in ('probability_pin_path_v1','probability_pin_path_v2') or terminal_evidence_policy!='frozen' or contact_group_policy!='frozen') and kind in TWO_PIN|{'ground'} and uid not in nested:
                forbidden=cut.copy()
                for other in parts:
                    if other['id']==uid:continue
                    a,b,c,d=boxes[other['id']];forbidden[b:d,a:c]=1
                for span in spans:
                    a,b,c,d=_box(span['bbox_px'],w,h);forbidden[b:d,a:c]=1
                for event in events:
                    if event.get('proof_box_px'):
                        a,b,c,d=event['proof_box_px'];forbidden[b:d,a:c]=1
                    else:
                        a,b=[int(v-.5) for v in event['point_px']];cv2.circle(forbidden,(a,b),max(2,event.get('cut_radius_px',2)),1,-1)
                forbidden=cv2.dilate(forbidden,np.ones((5,5),np.uint8))
                if contact_group_policy in ('same_lead_v1','same_lead_v2'):
                    contacts=_group_same_lead(contacts,gray,conductor_probability,conductor_evidence['threshold'],screened,regions,forbidden,policy=contact_group_policy)
                    compare=_group_same_lead(compare,gray,conductor_probability,conductor_evidence['threshold'],screened,regions,forbidden,policy=contact_group_policy)
                proposed=[]
                if terminal_evidence_policy in ('body_local_v1','body_local_v2') or mask_contact_policy in ('probability_pin_path_v1','probability_pin_path_v2'):
                    policy=terminal_evidence_policy if terminal_evidence_policy in ('body_local_v1','body_local_v2') else mask_contact_policy
                    proposed,pin_path_proofs=_probability_pin_contacts(gray,box,conductor_probability,conductor_evidence['threshold'],screened,regions,forbidden,policy=policy)
                if terminal_evidence_policy in ('body_local_v1','body_local_v2'):
                    for contact in contacts:
                        proof=_existing_body_local_proof(gray,box,contact,conductor_probability,conductor_evidence['threshold'],screened,regions,forbidden,policy=terminal_evidence_policy)
                        if proof is not None:
                            contact['contact_policy']=terminal_evidence_policy;pin_path_proofs[tuple(contact['point_px'])]=proof
                existing_sides={c['side'] for c in contacts}
                additions=[c for c in proposed if c['side'] not in existing_sides]
                contacts.extend(additions);compare.extend(additions)
                stable=sorted((c['side'],c['point_px'])for c in contacts)==sorted((c['side'],c['point_px'])for c in compare)
            if kind=='op_amp':
                rejected=[c for c in contacts if _body_witness(raw,box,c) is None]
                unconfirmed_contacts.extend({'component_id':uid,'point_px':[v+.5 for v in c['point_px']],
                    'reason':'no_native_component_body_entry','state':'unconfirmed_contact_hypothesis'} for c in rejected)
                contacts=[c for c in contacts if _body_witness(raw,box,c) is not None]
                compare=[c for c in compare if _body_witness(raw2,box,c) is not None]
                stable=sorted((c['side'],c['point_px']) for c in contacts)==sorted((c['side'],c['point_px']) for c in compare)
            expected=2 if kind in TWO_PIN else 1 if kind=='ground'else None
            model_kind='two_terminal'if kind in TWO_PIN else'reference_marker'if kind=='ground'else'unsupported_multi'
            opposed=len(contacts)==2 and (set(c['side']for c in contacts)in ({'left','right'},{'top','bottom'})or all('direction_from_body'in c for c in contacts))
            body_forbidden=cut.copy()
            for other in parts:
                if other['id']!=uid:
                    a,b,c,d=boxes[other['id']];body_forbidden[b:d,a:c]=1
            for span in spans:
                a,b,c,d=_box(span['bbox_px'],w,h);body_forbidden[b:d,a:c]=1
            contrast_proofs={tuple(c['point_px']):_contrast_body_support(gray,box,c,screened) for c in contacts} if mask_contact_policy in ('contrast_supported_v2','contrast_fallback_v3') else {}
            def body_proof(c):
                if c.get('contact_policy') in ('probability_pin_path_v1','probability_pin_path_v2','body_local_v1','body_local_v2'):return pin_path_proofs[tuple(c['point_px'])]['body']
                if mask_contact_policy not in ('contrast_supported_v2','contrast_fallback_v3'):return _body_witness(raw,box,c,raw2,body_forbidden)
                if mask_contact_policy=='contrast_fallback_v3':
                    strong=_body_witness(raw,box,c)
                    a,b=map(int,c['point_px']);side=c['side'];half=max(0,int(c.get('edge_contact_width_px',1))//2)
                    if side in ('left','right'):
                        observed=bool(raw[max(0,b-half):min(h,b+half+1),a].any())
                    elif side in ('top','bottom'):
                        observed=bool(raw[b,max(0,a-half):min(w,a+half+1)].any())
                    else:observed=bool(raw[b,a])
                    if strong is not None and observed:
                        return dict(strong,contact_witness='retained_strong_rgb_body_and_native_contact_band',contrast_policy='contrast_fallback_v3')
                proof=contrast_proofs.get(tuple(c['point_px']))
                if proof is not None:return proof['body']
                # Existing diagonal strong-ink witnesses remain unchanged; no
                # local contrast inference is claimed for their unsupported axis.
                if c['side'] not in DIRECTIONS:
                    a,b=map(int,c['point_px'])
                    return _body_witness(raw,box,c) if raw[b,a] else None
                return None
            body_proofs=[body_proof(c)for c in contacts]
            supported=kind in TWO_PIN and opposed and stable and all(body_proofs)and uid not in nested
            if kind=='op_amp':supported=len(contacts) in (3,5) and stable and all(body_proofs) and uid not in nested
            if kind=='ground':supported=len(contacts)==1 and stable and uid not in nested and earth_ground_shape(image,box)is not None
            models.append({'component_id':uid,'terminal_model':model_kind,'expected_arity':expected,'state':'supported'if supported else'unsupported','observed_contacts':len(contacts),'threshold_agreement':stable if conductor_mask is None else None,'wire_contact_source':'rgb_threshold_pair'if conductor_mask is None else'external_conductor_mask'})
            if not supported:issues.append({'id':_id('component-scope-',uid),'fragment_ids':sorted({f for c in contacts for f in by_region.get(c['ink_region'],[])}),'component_ids':[uid],'blocking':True,'reason':'unsupported_arity_role_body_or_contact_geometry'})
            source_terminal_proofs={}
            if terminal_localization_policy=='source_body_v1' and kind=='resistor' and supported:
                forbidden=cut.copy()
                for other in parts:
                    if other['id']==uid:continue
                    a,b,c,d=boxes[other['id']];forbidden[b:d,a:c]=1
                for span in spans:
                    a,b,c,d=_box(span['bbox_px'],w,h);forbidden[b:d,a:c]=1
                forbidden=cv2.dilate(forbidden,np.ones((5,5),np.uint8))
                observed=[_source_resistor_terminal(gray,box,c,screened,regions,forbidden) for c in contacts]
                if all(observed) and len({p['source_body_shape'] for p in observed})==1:
                    source_terminal_proofs={tuple(c['point_px']):p for c,p in zip(contacts,observed)}
            for index,c in enumerate(sorted(contacts,key=lambda r:(r['point_px'][0],r['point_px'][1],r['side']))):
                choices=by_region.get(c['ink_region'],[]);direction=c.get('direction_from_body',DIRECTIONS.get(c['side']));point=[float(v)+.5 for v in c['point_px']];contact_points.append((point,c['ink_region']))
                state='supported'if choices and stable and body_proof(c)is not None else'ambiguous';terminal={'id':_id('pin-',[uid,c['point_px']]),'component_id':uid,'point_px':point,'outward_unit_vector':direction,'geometric_role':c['side'],'semantic_role':None,'observed_contact':c,'body_support':body_proof(c),'attachment':{'state':state,'fragment_id':sorted(choices)[0]if state=='supported'else None,'candidate_fragment_ids':sorted(choices),'evidence_refs':(['native_occupied_lead_contact','threshold48_60_agreement']if conductor_mask is None else['external_mask_wire_contact','native_rgb_body_support'])if state=='supported'else[],'reasons':[]if state=='supported'else['contact_not_uniquely_supported']}}
                if mask_contact_policy=='contrast_fallback_v3' and terminal['body_support'] is not None and terminal['body_support'].get('contact_witness'):
                    terminal['attachment']['evidence_refs'].append('contrast_fallback_v3_retained_strong_rgb_body_and_observed_contact_band')
                witness=lead_witnesses.get((uid,tuple(c['point_px'])))
                if c.get('contact_policy') in ('probability_pin_path_v1','probability_pin_path_v2','body_local_v1','body_local_v2'):
                    witness=pin_path_proofs[tuple(c['point_px'])]['terminal']
                    terminal['attachment']['evidence_refs'].append(c['contact_policy']+'_bounded_weak_path_native_body_terminal')
                if c.get('contact_group_evidence'):terminal['attachment']['evidence_refs'].append(c['contact_group_evidence']['policy']+'_native_cross_sections_same_region');terminal['contact_group_evidence']=c['contact_group_evidence']
                if mask_contact_policy in ('contrast_supported_v2','contrast_fallback_v3') and contrast_proofs.get(tuple(c['point_px'])) is not None:
                    witness=contrast_proofs[tuple(c['point_px'])]['terminal']
                    terminal['attachment']['evidence_refs'].append(mask_contact_policy+'_mask_continuity_local_noise_margin_two_threshold_geometry')
                if witness is None and c.get('contact_policy')=='native_short_lead_v1':
                    witness=_trace_body_terminal(raw,box,c);other=_trace_body_terminal(raw2,box,c)
                    if witness is None or other is None or witness['point_px']!=other['point_px']:witness=None
                    if witness is not None:terminal['attachment']['evidence_refs'].append('native_short_lead_v1_no_gap_observed_fragment_only')
                if witness is None and state=='supported' and tuple(c['point_px']) in source_terminal_proofs:
                    witness=source_terminal_proofs[tuple(c['point_px'])]
                    terminal['attachment']['evidence_refs'].append('source_body_v1_existing_contact_localization_only')
                if witness is not None and witness.get('localization_policy')!='source_body_v1':contact_points.append((witness['point_px'],c['ink_region']))
                if witness is not None and state=='supported' and uid not in nested:
                    for a,b in witness.get('lead_pixels',[]):
                        if int(regions[b,a])==c['ink_region']:
                            owned_contact_leads.setdefault((a,b,c['ink_region']),set()).add(uid)
                terminal['body_terminal_px']=witness['point_px'] if witness is not None and not witness.get('transition_uncertainty') else None
                terminal['terminal_localization']=({k:v for k,v in witness.items() if k!='lead_pixels' and not (k=='point_px' and witness.get('transition_uncertainty'))} if witness else None)
                if witness is not None:terminal['attachment']['evidence_refs'].append('native_lead_body_transition_support_overlap' if witness.get('transition_uncertainty') else 'native_lead_body_transition_threshold_agreement')
                terminals.append(terminal)
            if expected is not None:
                for index in range(max(0,expected-len(contacts))):terminals.append({'id':_id('missing-pin-',[uid,index]),'component_id':uid,'point_px':None,'outward_unit_vector':None,'geometric_role':None,'semantic_role':None,'attachment':{'state':'unsupported','fragment_id':None,'candidate_fragment_ids':[],'evidence_refs':[],'reasons':['missing_observed_terminal']}})
        if len(terminals)>LIMITS['terminals']:raise ValueError('Topology terminal budget exceeded')
        contact_endpoint_matches=[];source_footprint_endpoints=[]
        wire_distance=cv2.distanceTransform(screened,cv2.DIST_L2,3)
        for x,y in endpoints:
            region=int(regions[y,x]);point=[x+.5,y+.5]
            owners=owned_contact_leads.get((x,y,region),set())
            if len(owners)==1:
                component_lead_endpoints.append({'point_px':point,'component_id':next(iter(owners)),
                    'evidence':'native_source_lead_to_body_transition_in_owned_component_interior'})
                continue
            proof=contact_endpoint_proof(point,region,contact_points,regions,path_neighbors,wire_distance)
            if proof is not None:
                contact_endpoint_matches.append(proof);continue
            footprint=observed_multiarm_endpoint(point,region,rgb_working&(1-cut),regions,float(wire_distance[y,x]))
            if footprint is not None:
                source_footprint_endpoints.append(footprint);continue
            if within_complete_junction(point,region,events,regions):continue
            registered=False
            for event in events:
                disk=event.get('cut_disk_px')
                if event['state']=='separate' and disk and region in event.get('bound_arm_regions',[]):
                    if disk['inner_radius']-disk['stroke_radius']-math.sqrt(2)<=math.dist(point,disk['center'])<=disk['outer_radius']+math.sqrt(2):registered=True
                elif event['state']=='unresolved' and math.dist(point,event['point_px'])<=event.get('cut_radius_px',0)+2:registered=True
            if registered:continue
            ids=by_region.get(region,[])
            if ids:ports.append({'id':_id('open-end-',point),'point_px':point,'fragment_id':sorted(ids)[0],'state':'unknown_external_or_open_stub'})
        for operation in masks:
            if operation['state']!='retained_ambiguous_parent':continue
            x0,y0,x1,y1=operation['region_px'];seen=set(map(int,regions[y0:y1,x0:x1].ravel()))-{0}
            ids=sorted({f for region in seen for f in by_region.get(region,[])})
            issues.append({'id':_id('retained-parent-',operation['source_ref']),'fragment_ids':ids,'component_ids':[operation['source_ref']],'blocking':True,'affects_equivalence':True,'reason':'ambiguous_parent_interior_retained_not_proven_conductor'})
        # No OCR rectangle is erased. Any text/wire overlap keeps its ambiguity.
        for span in spans:
            x0,y0,x1,y1=_box(span['bbox_px'],w,h);seen=set(map(int,regions[y0:y1,x0:x1].ravel()))-{0};ids=sorted({f for r in seen for f in by_region.get(r,[])})
            if ids:issues.append({'id':_id('text-overlap-',span['id']),'fragment_ids':ids,'blocking':True,'affects_equivalence':True,'reason':'retained_text_or_false_text_may_overlap_conductor','ocr_span_id':span['id']})
        if not converged or not converged2:issues.append({'id':'thinning-budget','fragment_ids':[f['id']for f in fragments],'blocking':True,'reason':'thinning_iteration_budget'})
        peak=_guard(start)
        if mask_path is not None: cv2.imwrite(str(mask_path), screened * 255)
        return {'schema_version':VERSION,'status':'partial'if issues else'observed','frame':{'width':w,'height':h,'canonical_png_sha256':inventory['observations']['image']['sha256'],'canonical_rgb_sha256':canonical_rgb_sha256,'coordinate_convention':'canonical pixel centers; x right,y down','analysis_to_canonical_matrix':[[1,0,0],[0,1,0],[0,0,1]]},'wire_fragments':fragments,'connections':connections,'gap_connections':gap_records,'terminals':terminals,'component_models':models,'intersections':events,'issues':issues,'external_ports':ports,'mask_operations':masks,'unresolved_regions':[m for m in masks if m['state']in ('uncertain_exclusion','retained_ambiguous_parent')],'evidence':{**({'terminal_localization_policy':terminal_localization_policy} if terminal_localization_policy!='frozen' else {}),**({'body_exclusion_policy':body_exclusion_policy,'body_outline_fringe':body_fringe_records} if body_exclusion_policy!='frozen' else {}),**({'terminal_evidence_policy':terminal_evidence_policy,'contact_group_policy':contact_group_policy} if terminal_evidence_policy!='frozen' or contact_group_policy!='frozen' else {}),'mask_contact_policy':mask_contact_policy,'conductor_gap_policy':conductor_gap_policy,'stroke_scene':scene_evidence,'supported_single_gap_links':len(gap_records),'wire_foreground_source':'external_conductor_mask'if conductor_mask is not None else'rgb_threshold','conductor_mask':conductor_evidence,'intersection_evidence_source':'native_rotation_independent_collar','junction_convention':junction_convention,'original_ink_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'working_ink_sha256':hashlib.sha256(screened.tobytes()).hexdigest(),'thresholds':[48,60],'line_run':9,'connectivity':8,'contact_endpoint_matches':contact_endpoint_matches,'component_lead_endpoints':component_lead_endpoints,'source_footprint_endpoints':source_footprint_endpoints,'unconfirmed_contact_hypotheses':unconfirmed_contacts,'thinning':'Zhang-Suen','iterations':[iterations,iterations2],'opencv_threads':1,'peak_owned_RSS_bytes':peak,'seconds':time.monotonic()-start},'limits':dict(LIMITS),'verified':False}
    finally:cv2.setNumThreads(0 if previous==1 else previous)
