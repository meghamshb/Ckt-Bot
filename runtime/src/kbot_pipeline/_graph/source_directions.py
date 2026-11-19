"""Observe arrow instances independently of annotation/text grouping."""
import cv2
import numpy as np


def continuous_wire_context(box, stroke, wire_support, occupied, source_ink, bodies, *, electrical=True):
    """Witness two collinear flanks and an already occupied, nonwhite local path.

    Returned pixels are a subset of original source AND SAM support. No white
    pixel, nearby endpoint, component body or third wire arm can supply a join.
    """
    import math,hashlib
    x0,y0,x1,y1=map(int,box);h,w=occupied.shape
    extent=max(8,math.ceil(max(x1-x0,y1-y0)*1.25));a=max(0,x0-extent);b=max(0,y0-extent);c=min(w,x1+extent);d=min(h,y1+extent)
    local=wire_support[b:d,a:c].copy().astype(np.uint8)
    pad=max(1,round(stroke));local[max(0,y0-b-pad):min(d-b,y1-b+pad),max(0,x0-a-pad):min(c-a,x1-a+pad)]=0
    lines=cv2.HoughLinesP(local,1,np.pi/360,max(4,round(3*stroke)),minLineLength=max(5,round(4*stroke)),maxLineGap=0)
    if lines is None:return None
    segments=[]
    for row in lines.reshape(-1,4):
        u=np.asarray(row[:2],float)+[a,b];v=np.asarray(row[2:],float)+[a,b];axis=v-u;length=float(np.linalg.norm(axis))
        if length:segments.append((u,v,axis/length))
    center=np.asarray([(x0+x1-1)/2,(y0+y1-1)/2]);corners=np.asarray([[x0,y0],[x1-1,y0],[x0,y1-1],[x1-1,y1-1]],float)
    candidates=[]
    for i,(u,v,axis) in enumerate(segments):
        normal=np.asarray([-axis[1],axis[0]]);bounds=(corners-center)@axis;lo,hi=float(bounds.min()),float(bounds.max())
        for p,q,other in segments[i+1:]:
            if abs(float(axis@other))<.985:continue
            cross=np.asarray([u,v,p,q])@normal
            if float(np.ptp(cross))>max(1.5,1.5*stroke):continue
            normal_bounds=(corners-center)@normal
            offset=float(np.median(cross)-center@normal)
            if not float(normal_bounds.min())<=offset<=float(normal_bounds.max()):continue
            first=np.asarray([u,v]);second=np.asarray([p,q]);t1=(first-center)@axis;t2=(second-center)@axis
            if max(t1)<=lo+pad and min(t2)>=hi-pad:left,right=first,second
            elif max(t2)<=lo+pad and min(t1)>=hi-pad:left,right=second,first
            else:continue
            start=left[np.argmax((left-center)@axis)]-axis*max(2,2*stroke)
            end=right[np.argmin((right-center)@axis)]+axis*max(2,2*stroke)
            if not(a<=start[0]<c and b<=start[1]<d and a<=end[0]<c and b<=end[1]<d):continue
            corridor=np.zeros((d-b,c-a),np.uint8)
            cv2.line(corridor,tuple(np.rint(start-[a,b]).astype(int)),tuple(np.rint(end-[a,b]).astype(int)),1,max(1,round(stroke)))
            if np.any(corridor&bodies[b:d,a:c]):continue
            # A visual shaft may follow original source ink through an observed
            # head. This is separate from the stricter electrical witness.
            allowed=corridor.copy()
            if not electrical:
                allowed[max(0,y0-b):min(d-b,y1-b),max(0,x0-a):min(c-a,x1-a)]=1
            if np.any(allowed&bodies[b:d,a:c]):continue
            supported=allowed&source_ink[b:d,a:c]
            if electrical:supported &= occupied[b:d,a:c]
            if not np.any(supported[max(0,y0-b):min(d-b,y1-b),max(0,x0-a):min(c-a,x1-a)]):continue
            _,labels=cv2.connectedComponents(supported,8)
            def sample(point):
                x,y=np.rint(point-[a,b]).astype(int);values=set(labels[max(0,y-1):y+2,max(0,x-1):x+2].ravel())-{0}
                return next(iter(values)) if len(values)==1 else None
            one,two=sample(start),sample(end)
            if one is None or one!=two:continue
            # A third source-supported arm at the object boundary is a junction
            # or crossing hypothesis, not a simple overprinted lead.
            _,arms=cv2.connectedComponents(local,8)
            yy,xx=np.mgrid[b:d,a:c]
            ring=((xx>=x0-2*pad)&(xx<x1+2*pad)&(yy>=y0-2*pad)&(yy<y1+2*pad))&~((xx>=x0-pad)&(xx<x1+pad)&(yy>=y0-pad)&(yy<y1+pad))
            ids=set(arms[ring&(local>0)].ravel())-{0}
            if len(ids)>2:continue
            ys,xs=np.nonzero(labels==one);pixels=np.column_stack((xs+a,ys+b)).astype(int).tolist()
            record={'axis':axis.tolist(),'source_extent_px':list(box),'flank_segments_px':[[u.tolist(),v.tolist()],[p.tolist(),q.tolist()]],'support_type':'electrical_source_and_SAM' if electrical else 'visual_source_only','coordinate_frame':'source_image_pixels','source_ink_sha256':hashlib.sha256(source_ink.tobytes()).hexdigest(),'proof':'two_collinear_source_wire_flanks_and_continuous_original_SAM_source_ink_path_no_body_or_third_arm' if electrical else 'two_collinear_observed_flanks_and_continuous_original_source_ink_through_head_no_body_or_third_arm'}
            if electrical:record['retained_pixels_px']=pixels
            record['_support_pixels']=len(pixels)
            candidates.append(record)
    if not candidates:return None
    axes=[]
    for item in candidates:
        axis=np.asarray(item['axis'])
        if not any(abs(float(axis@old))>.985 for old in axes):axes.append(axis)
    if len(axes)!=1:return None
    chosen=max(candidates,key=lambda item:item['_support_pixels']);chosen.pop('_support_pixels')
    return chosen


def arrow_objects(ink, stroke, *, wire_support=None, occupied=None, source_ink=None, bodies=None, include_rejected=False):
    """One authoritative arrow identity, including rejected/ambiguous proposals."""
    import hashlib,json,math
    from .stroke_scene import arrow_shape_evidence
    ink=ink.astype(np.uint8);count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    proposals=[]
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i])
        if area>=8:proposals.append(([x,y,x+w,y+h],(labels[y:y+h,x:x+w]==i).astype(np.uint8),'source_ink_object'))
    contextual=all(x is not None for x in (wire_support,occupied,source_ink,bodies))
    if contextual:
        native=(source_ink&occupied&(1-bodies)).astype(np.uint8)
        distance=cv2.distanceTransform(native,cv2.DIST_L2,3)
        peaks=((distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))-1e-5)&(distance>=1.6*stroke)).astype(np.uint8)
        n,_,_,centers=cv2.connectedComponentsWithStats(peaks,8)
        for center in centers[1:]:
            x,y=np.rint(center).astype(int);radius=max(4,math.ceil(4*float(distance[y,x])));a=max(0,x-radius);b=max(0,y-radius);c=min(native.shape[1],x+radius+1);d=min(native.shape[0],y+radius+1)
            proposals.append(([a,b,c,d],native[b:d,a:c].copy(),'source_width_lobe'))
    records=[]
    for box,bits,proposal in proposals:
        box=list(map(int,box))
        context=continuous_wire_context(box,stroke,wire_support,occupied,source_ink,bodies) if contextual else None
        visual=context or (continuous_wire_context(box,stroke,wire_support,occupied,source_ink,bodies,electrical=False) if contextual else None)
        shape=arrow_shape_evidence(bits,stroke,visual['axis'] if visual else None,wire_context=visual is not None)
        uid='arrow-'+hashlib.sha256(json.dumps(box).encode()+bits.tobytes()).hexdigest()[:20]
        record={'id':uid,'bbox_px':box,'source_extent_px':box,'proposal':proposal,**shape}
        if context:record['shared_wire_context']=context
        if visual:record['visual_shaft_context']=visual
        records.append(record)
    # Competing representations of the same arrow share one accepted identity.
    accepted=sorted([r for r in records if r['status']=='accepted'],key=lambda r:(r['bbox_px'][2]-r['bbox_px'][0])*(r['bbox_px'][3]-r['bbox_px'][1]))
    chosen=[]
    for record in accepted:
        a,b,c,d=record['bbox_px'];area=(c-a)*(d-b);duplicate=None
        for other in chosen:
            e,f,g,h=other['bbox_px'];overlap=max(0,min(c,g)-max(a,e))*max(0,min(d,h)-max(b,f))
            if overlap>=.7*min(area,(g-e)*(h-f)) and float(np.dot(record['direction_unit'],other['direction_unit']))>.98:duplicate=other;break
        if duplicate:
            record.update(status='rejected',direction_unit=None,reason='duplicate_representation',canonical_object_id=duplicate['id'])
        else:chosen.append(record)
    return records if include_rejected else [r for r in records if r['status']=='accepted']


def source_roles(image, components, pins, stroke):
    """Bind an isolated arrow enclosed by a source outline to observed pins.

    A closed background hole witnesses the source interior. The arrow must be
    unique inside it, oppose the two existing lead anchors, and agree under
    both source thresholds. No new pin or terminal position is constructed.
    """
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    paper = float(np.percentile(gray, 90)); result = []
    for component in components:
        if component['type'] not in ('current_source', 'dependent_current_source'):
            continue
        owned = [p for p in pins if p['component_id'] == component['id']]
        if len(owned) != 2 or any(not p.get('anchor_px') for p in owned):
            continue
        x0, y0, x1, y1 = map(int, component['bbox']); observations = []
        for contrast in (48, 60):
            ink = (gray[y0:y1, x0:x1] <= paper-contrast).astype(np.uint8)
            contours, hierarchy = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            holes = []
            if hierarchy is not None:
                for i, contour in enumerate(contours):
                    depth = 0; parent = int(hierarchy[0][i][3])
                    while parent >= 0:
                        depth += 1; parent = int(hierarchy[0][parent][3])
                    if depth % 2:
                        holes.append(contour)
            candidates = []
            for arrow in arrow_objects(ink, stroke):
                a, b, c, d = arrow['bbox_px']
                corners = [(a, b), (c-1, b), (a, d-1), (c-1, d-1)]
                if not any(all(cv2.pointPolygonTest(hole, tuple(map(float, p)), False) > 0 for p in corners) for hole in holes):
                    continue
                axis = np.array(arrow['direction_unit']); center = np.array([(a+c)/2+x0, (b+d)/2+y0])
                anchors = [np.array(p['anchor_px']) for p in owned]
                projections = [float((p-center) @ axis) for p in anchors]
                delta = anchors[1]-anchors[0]; length = float(np.linalg.norm(delta))
                transverse = np.array([-axis[1], axis[0]])
                if not length or projections[0]*projections[1] >= 0 or abs(float(delta @ axis))/length < .95:
                    continue
                if max(abs(float((p-center) @ transverse)) for p in anchors) > max(2*stroke, min(c-a, d-b)):
                    continue
                order = np.argsort(projections)
                candidates.append({'from': owned[int(order[0])]['id'], 'to': owned[int(order[1])]['id'],
                    'arrow_bbox_px': [a+x0, b+y0, c+x0, d+y0], 'direction_unit': axis.tolist()})
            if len(candidates) != 1:
                break
            observations.append(candidates[0])
        if len(observations) == 2 and all(observations[0][k] == observations[1][k] for k in ('from', 'to')):
            result.append({'component_id': component['id'], **observations[0],
                'evidence': 'unique_isolated_arrow_in_closed_source_interior_two_threshold_agreement'})
    return result
