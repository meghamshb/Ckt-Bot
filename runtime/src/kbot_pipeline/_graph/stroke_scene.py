"""Image-wide stroke evidence shared by wire, port and junction observation.

SAM proposes occupied pixels. Appearance proposes source-object classes from long strokes;
compact marks and unclassified ink are retained. No colour name or textbook palette is assumed and no missing ink is painted.
Ports are fitted from occupied collars in continuous orientation space.
"""
import hashlib
import cv2
import numpy as np


def arrow_shape_evidence(bits, stroke, axis_hint=None, wire_context=False):
    """Cross-section evidence for a pointed head, with its own or witnessed shaft.

    Width changes are measured along observed axes, not a global aspect ratio.
    A head-only mark needs an independently observed wire-context shaft.
    """
    ys,xs=np.nonzero(bits)
    rejected={'status':'rejected','direction_unit':None,'reason':'no_supported_pointed_head_and_shaft'}
    if len(xs)<8:return rejected
    points=np.column_stack((xs,ys)).astype(float)
    axes=[]
    if axis_hint is not None:axes.append(np.asarray(axis_hint,float))
    else:
        _,vectors=np.linalg.eigh(np.cov(points.T));axes.append(vectors[:,-1])
        lines=cv2.HoughLinesP(bits.astype(np.uint8),1,np.pi/360,max(4,round(3*stroke)),minLineLength=max(5,round(3*stroke)),maxLineGap=0)
        if lines is not None:
            line=max(lines.reshape(-1,4),key=lambda p:(p[2]-p[0])**2+(p[3]-p[1])**2)
            axes.append(np.asarray(line[2:],float)-line[:2])
    accepted=[];metrics=[]
    for raw_axis in axes:
        length=float(np.linalg.norm(raw_axis))
        if not length:continue
        axis=raw_axis/length;normal=np.array([-axis[1],axis[0]])
        along=points@axis;across=points@normal;bins=np.rint(along-along.min()).astype(int)
        widths=[];centers=[];positions=[]
        for k in range(int(bins.max())+1):
            values=across[bins==k]
            if not len(values):continue
            positions.append(k);widths.append(float(np.ptp(values)+1));centers.append(float((values.min()+values.max())/2))
        widths=np.asarray(widths);centers=np.asarray(centers);positions=np.asarray(positions)
        if len(widths)<5:continue
        baseline=max(1.,float(np.percentile(widths,20)));peak=float(widths.max())
        if peak<max(2.2*baseline,2.2*stroke):continue
        broad=np.flatnonzero(widths>=max(1.8*baseline,.38*peak))
        if len(broad)<max(3,round(1.5*stroke)):continue
        lo,hi=int(broad[0]),int(broad[-1]);section=widths[lo:hi+1]
        if len(section)<3 or hi-lo>0.85*len(widths):continue
        slope=float(np.polyfit(positions[lo:hi+1],section,1)[0])
        # The broad band locates a head; it is not its complete taper. Include
        # the actually observed tip-facing transition to the shaft width.
        # Otherwise low-resolution triangular heads lose their narrow tip
        # before variation is measured. Both measured axes still compete.
        sign=-1 if slope>0 else 1
        if sign>0:
            while hi+1<len(widths) and widths[hi]>baseline:
                hi+=1
        else:
            while lo>0 and widths[lo]>baseline:
                lo-=1
        section=widths[lo:hi+1]
        # A taper has intermediate native-cell widths. A perpendicular arm is
        # an abrupt narrow/wide/narrow spike, even if a line fit has a slope.
        cell_widths=np.rint(section)
        tip_order=cell_widths if sign>0 else cell_widths[::-1]
        taper=tip_order[int(np.argmax(tip_order)):]
        if len(np.unique(taper))<3 or np.any(np.diff(taper)>1):continue
        taper_fit=np.polyval(np.polyfit(np.arange(len(taper)),taper,1),np.arange(len(taper)))
        # Each of the two measured silhouette edges occupies a native pixel
        # cell. Their projected cell extents bound width-fit uncertainty.
        if np.max(abs(taper-taper_fit))>2*float(np.abs(normal).sum()):continue
        slope=float(np.polyfit(positions[lo:hi+1],section,1)[0])
        variation=float(np.ptp(section))
        if variation<max(stroke,.25*peak) or abs(slope)<.15:continue
        sign=-1 if slope>0 else 1
        # The head must be centered on one straight shaft, not a curved glyph.
        center=float(np.median(centers[lo:hi+1]));drift=float(np.max(abs(centers[lo:hi+1]-center)))
        if drift>max(stroke,baseline):continue
        behind=np.flatnonzero(positions<positions[lo]) if sign>0 else np.flatnonzero(positions>positions[hi])
        shaft=[i for i in behind if widths[i]<=max(1.7*baseline,1.5*stroke) and abs(centers[i]-center)<=max(1.,stroke)]
        intrinsic=len(shaft)>=max(4,round(3*stroke))
        # A colored triangular/chevron head has no same-color shaft. A continuous
        # source/SAM wire with two flanks supplies that evidence, never proximity.
        if not intrinsic and not wire_context:continue
        direction=(axis*sign).round(5).tolist()
        accepted.append(direction);metrics.append({'head_peak_width_px':peak,'shaft_width_px':baseline,'head_width_slope':slope,'centerline_drift_px':drift,'shaft_basis':'intrinsic_straight_shaft' if intrinsic else 'independently_observed_source_shaft_context'})
    unique=[]
    for direction,metric in zip(accepted,metrics):
        if not any(np.dot(direction,d)>.98 for d,_ in unique):unique.append((direction,metric))
    if len(unique)==1:return {'status':'accepted','direction_unit':unique[0][0],'reason':'pointed_width_profile_with_supported_shaft','geometry':unique[0][1]}
    if len(unique)>1:return {'status':'ambiguous','direction_unit':None,'reason':'competing_arrow_directions'}
    return rejected


def shaft_head_direction(bits, stroke):
    """Compatibility entry point; all callers share the same arrow observer."""
    return arrow_shape_evidence(bits,stroke).get('direction_unit')


def annotation_ink(direction,norm,similarity,stroke):
    """Compatibility view of the one authoritative arrow-object observer."""
    from .source_directions import arrow_objects
    eligible=((norm>60)&(similarity<.97)).astype(np.uint8)
    records=arrow_objects(eligible,stroke)
    result=np.zeros(norm.shape,np.uint8)
    for record in records:
        x,y,a,b=record['bbox_px'];result[y:b,x:a]|=eligible[y:b,x:a]
    return result,records


def observed_line_tracks(native,proposed,bodies,stroke,body_contacts=()):
    """Corroborate SAM supports with continuous original-image wire tracks.

    Each track needs SAM occupancy plus another mask or source-verified body entry, and contains
    a continuous source-ink path between them. Never cross white paper or a
    symbol body. Length/spacing does not decide electrical connection.
    """
    source=(native&(1-bodies)).astype(np.uint8);base=(proposed&(1-bodies)).astype(np.uint8)
    _,regions=cv2.connectedComponents(base,8);added=np.zeros_like(base);proofs=[]
    run=max(5,round(7*stroke));lines=cv2.HoughLinesP(source,1,np.pi/360,max(4,round(4*stroke)),minLineLength=run,maxLineGap=0)
    if lines is None:return proposed,proofs
    radius=max(1,round(stroke));
    for x0,y0,x1,y1 in lines.reshape(-1,4):
        a,b=max(0,min(x0,x1)-radius),max(0,min(y0,y1)-radius);c,d=min(source.shape[1],max(x0,x1)+radius+1),min(source.shape[0],max(y0,y1)+radius+1)
        band=np.zeros((d-b,c-a),np.uint8);cv2.line(band,(int(x0-a),int(y0-b)),(int(x1-a),int(y1-b)),1,2*radius+1);band&=source[b:d,a:c]
        n,labels=cv2.connectedComponents(band,8)
        for k in range(1,n):
            bits=labels==k;ids=np.unique(regions[b:d,a:c][bits]);ids=ids[ids>0]
            contacts=[contact for contact in body_contacts if a<=contact['point_px'][0]<c and b<=contact['point_px'][1]<d and bits[contact['point_px'][1]-b,contact['point_px'][0]-a]]
            if not len(ids) or (len(ids)<2 and not contacts):continue
            if np.count_nonzero(base[b:d,a:c][bits])<.5*np.count_nonzero(bits):continue
            missing=bits&(base[b:d,a:c]==0)
            if not missing.any():continue
            added[b:d,a:c][bits]=1
            proofs.append({'segment_px':[[int(x0),int(y0)],[int(x1),int(y1)]],'existing_mask_regions':ids.tolist(),'native_added_pixels':int(missing.sum()),'body_entry_contacts':contacts,'proof':'continuous_source_ink_track_between_independent_mask_or_body_witnesses'})
    return proposed|added,proofs


def _guard_annotation_deletions(proposed, staged, bodies, body_contacts, objects):
    """Decline combined annotation deletions that cut a physical lead region.

    This is abstention before committing a conditioning decision. Only original
    occupied pixels survive; the retained, disputed region is explicitly
    ambiguous rather than certified as a conductor by a visual arrow label.
    """
    before=proposed&(1-bodies);after=staged&(1-bodies)
    _,original=cv2.connectedComponents(before.astype(np.uint8),8)
    _,descendants=cv2.connectedComponents(after.astype(np.uint8),8)
    boundary=cv2.dilate(bodies.astype(np.uint8),np.ones((3,3),np.uint8))>0
    _,boundary_bands=cv2.connectedComponents((before&boundary).astype(np.uint8),8)
    removed=(before>0)&(after==0);declined=[]
    for region in np.unique(original[removed]):
        if not region:continue
        support=original==region
        children=sorted(int(v) for v in np.unique(descendants[support]) if v>0)
        contacts=[]
        for contact in body_contacts:
            x,y=contact['point_px']
            # Preserve exact observed occupied incidence; never snap a missing
            # contact to a nearby surviving pixel or another component's lead.
            if (int(x)!=x or int(y)!=y or not(0<=x<proposed.shape[1] and 0<=y<proposed.shape[0])):continue
            if original[int(y),int(x)]==region:contacts.append(contact)
        ids=sorted({c['component_id'] for c in contacts})
        lost_contacts=[dict(c) for c in contacts if not after[int(c['point_px'][1]),int(c['point_px'][0])]]
        lost_bands=[]
        for band in np.unique(boundary_bands[support]):
            if not band:continue
            footprint=boundary_bands==band
            if np.any(after[footprint]):continue
            by,bx=np.nonzero(footprint)
            lost_bands.append({'source_extent_px':[int(bx.min()),int(by.min()),int(bx.max()+1),int(by.max()+1)],
                'occupied_pixels':int(len(bx)),
                'source_footprint_sha256':hashlib.sha256(footprint.tobytes()).hexdigest(),
                'component_ids':sorted({c['component_id'] for c in contacts if footprint[int(c['point_px'][1]),int(c['point_px'][0])]})})
        if len(children)==1 and not lost_contacts and not lost_bands:continue
        if not ids and not np.any(support&boundary):continue
        ys,xs=np.nonzero(removed&support)
        box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        object_ids=[o['id'] for o in objects if o['bbox_px'][0]<box[2] and o['bbox_px'][2]>box[0]
                    and o['bbox_px'][1]<box[3] and o['bbox_px'][3]>box[1]]
        uid='declined-annotation-deletion-'+hashlib.sha256(str((int(region),box,object_ids)).encode()).hexdigest()[:20]
        declined.append({'id':uid,'object_ids':object_ids,'source_extent_px':box,
            'affected_before_regions':[int(region)],'proposed_after_regions':children,
            'affected_component_ids':ids,'affected_fragment_ids':[],
            'component_bearing':True,'coordinate_frame':'source_image_pixels',
            'proposed_removed_pixels':int(len(xs)),'removed_pixels':0,
            'before_support_pixels_sha256':hashlib.sha256(support.tobytes()).hexdigest(),
            'lost_body_contacts':lost_contacts,'lost_body_boundary_footprints':lost_bands,
            'reason':'annotation_deletion_would_erase_observed_component_incidence' if lost_contacts or lost_bands else 'annotation_deletion_would_change_component_bearing_occupied_region_connectivity',
            'action':'declined_annotation_deletion','status':'ambiguous','blocking':True})
        staged[support]=proposed[support]
    return staged,declined,original


def conductor_appearance(image, proposed, bodies, body_contacts=()):
    rgb=np.asarray(image).astype(np.float32);paper=np.percentile(rgb.reshape(-1,3),95,axis=0)
    ink=np.maximum(0,paper-rgb);norm=np.linalg.norm(ink,axis=2);direction=ink/np.maximum(norm[...,None],1)
    outside=proposed.astype(np.uint8)&(1-bodies)
    widths=cv2.distanceTransform(outside,cv2.DIST_L2,3)
    stroke=max(1.,float(np.percentile(widths[widths>0],65))*2) if outside.any() else 1.
    run=max(5,round(7*stroke));horizontal=cv2.morphologyEx(outside,cv2.MORPH_OPEN,np.ones((1,run),np.uint8));vertical=cv2.morphologyEx(outside,cv2.MORPH_OPEN,np.ones((run,1),np.uint8))
    anchors=(horizontal|vertical).astype(bool)&(norm>48)
    if not anchors.any():return proposed.copy(),{'state':'appearance_not_identifiable','stroke_width_px':stroke}
    codes=np.rint(direction[anchors]*12).astype(int);unique,counts=np.unique(codes,axis=0,return_counts=True);mode=unique[np.argmax(counts)]/12.;mode/=np.linalg.norm(mode)
    similarity=direction@mode;support=similarity[anchors]>.985
    # A palette is usable only when its image-wide straight-wire population
    # agrees. Ambiguous/multicolour drawings retain the model's proposals.
    agreement=float(np.mean(support));identifiable=agreement>=.75
    annotation=np.zeros(proposed.shape,np.uint8);objects=[];arrows=[]
    strong=(norm>48).astype(np.uint8)
    wire_support=(anchors&(similarity>.985)&(bodies==0)).astype(np.uint8)
    if identifiable:
        from .source_directions import arrow_objects
        mismatch=(((similarity<.97)|(proposed==0))&(norm>48)&(bodies==0)).astype(np.uint8)
        objects=arrow_objects(mismatch,stroke,wire_support=wire_support,occupied=proposed,
            source_ink=strong,bodies=bodies,include_rejected=True)
        for obj in objects:
            x,y,a,b=obj['source_extent_px'];bits=mismatch[y:b,x:a]
            if obj['status']=='accepted':
                if obj.get('visual_shaft_context',{}).get('support_type')=='visual_source_only':
                    # Visual head identity is not a conductor decision. Without
                    # an electrical preservation witness it authorizes neither
                    # annotation deletion nor restoration of occupied pixels.
                    continue
                # Full source object extent includes shared conductor ink. Its
                # occupied line is retained below only with a two-flank proof.
                annotation[y:b,x:a]|=strong[y:b,x:a]&(1-bodies[y:b,x:a])
            elif obj.get('reason')!='duplicate_representation':
                # Retain the established nonconductor proposal screening,
                # without treating its failure as an accepted arrow identity.
                high,wide=bits.shape
                if min(wide,high)<2 or max(wide,high)/min(wide,high)<1.5:continue
                radius=cv2.distanceTransform(bits,cv2.DIST_L2,3);peak=float(radius.max())
                occupied=radius[radius>0]
                if peak<stroke or max(wide,high)<6*peak or len(occupied)<8 or float(np.mean(occupied<peak*.55))<.4:continue
                contours,_=cv2.findContours(bits,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                perimeter=sum(cv2.arcLength(c,True) for c in contours)
                if 4*np.pi*int(bits.sum())/max(perimeter*perimeter,1)>.65:continue
                annotation[y:b,x:a]|=bits
        arrows=[obj for obj in objects if obj['status']=='accepted']
        _,nearest=cv2.distanceTransformWithLabels((strong==0).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        classes=np.ones(int(nearest.max())+1,bool);classes[nearest[strong>0]]=annotation[strong>0]==0
        accepted=classes[nearest]
    else:accepted=np.ones(proposed.shape,bool)
    result=proposed&accepted.astype(np.uint8)
    shared=[]
    for obj in arrows:
        context=obj.get('shared_wire_context')
        if not context:continue
        pixels=context['retained_pixels_px']
        restored=[]
        for x,y in pixels:
            # This is preservation of occupied evidence, not gap completion.
            if proposed[y,x] and strong[y,x] and not bodies[y,x]:
                if not result[y,x]:restored.append([x,y])
                result[y,x]=1
        if restored:shared.append({'object_id':obj['id'],**context,
            'preserved_proposed_pixels_px':restored,'preserved_pixels':len(restored)})
    result,declined,original_regions=_guard_annotation_deletions(proposed,result,bodies,body_contacts,objects)
    native=((norm>60)&(similarity>.985)&accepted).astype(np.uint8) if identifiable else np.zeros_like(proposed)
    result,line_proofs=observed_line_tracks(native,result,bodies,stroke,body_contacts)
    # Audit every deletion which changes a source/SAM component's connectivity.
    before=proposed&(1-bodies);after=result&(1-bodies)
    _,before_labels=cv2.connectedComponents(before.astype(np.uint8),8)
    _,after_labels=cv2.connectedComponents(after.astype(np.uint8),8)
    removed=(before>0)&(after==0);removals=[];uncertain=[]
    for entry in declined:
        support=np.isin(original_regions,entry['affected_before_regions'])
        entry['after_regions']=sorted(int(v) for v in np.unique(after_labels[support]) if v>0)
        uncertain.append(entry)
    boundary=cv2.dilate(bodies.astype(np.uint8),np.ones((3,3),np.uint8))>0
    for region in np.unique(before_labels[removed]):
        if not region:continue
        support=before_labels==region;descendants=sorted(int(x) for x in np.unique(after_labels[support]) if x>0)
        if len(descendants)==1:continue
        ys,xs=np.nonzero(removed&support)
        box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        ids=sorted({c['component_id'] for c in body_contacts if 0<=c['point_px'][0]<proposed.shape[1] and 0<=c['point_px'][1]<proposed.shape[0] and before_labels[c['point_px'][1],c['point_px'][0]]==region})
        bearing=bool(ids) or bool(np.any(support&boundary))
        object_ids=[o['id'] for o in objects if o['bbox_px'][0]<box[2] and o['bbox_px'][2]>box[0] and o['bbox_px'][1]<box[3] and o['bbox_px'][3]>box[1]]
        uid='appearance-removal-'+hashlib.sha256(str((int(region),box,object_ids)).encode()).hexdigest()[:20]
        record={'id':uid,'object_ids':object_ids,'source_extent_px':box,
            'affected_before_regions':[int(region)],'after_regions':descendants,
            'affected_component_ids':ids,'affected_fragment_ids':[],
            'component_bearing':bearing,'relevance_proof':'original_region_reaches_body_boundary_or_body_entry' if bearing else 'isolated_original_region_has_no_body_boundary_or_body_entry',
            'removed_pixels':int(len(xs)),'before_support_pixels_sha256':hashlib.sha256(support.tobytes()).hexdigest(),
            'reason':'annotation_conditioning_changes_original_occupied_component_connectivity'}
        removals.append(record)
        if bearing:uncertain.append({**record,'status':'ambiguous','blocking':True})
    return result,{'state':'observed_straight_wire_appearance' if identifiable else 'appearance_not_identifiable',
        'reference_ink_direction':mode.tolist(),'anchor_agreement':agreement,
        'annotation_objects':objects,'arrow_objects':objects,'reference_arrow_objects':arrows,
        'shared_ink_continuations':shared,'connectivity_removals':removals,'uncertain_overlaps':uncertain,
        'declined_annotation_deletions':declined,
        'anchor_pixels':int(anchors.sum()),'removed_proposed_pixels':int((proposed&(1-result)).sum()),
        'stroke_width_px':stroke,'source_line_tracks':line_proofs,
        'policy':'layered_source_SAM_annotation_evidence_with_shared_occupied_tracks',
        'input_mask_pixels_sha256':hashlib.sha256(proposed.tobytes()).hexdigest(),
        'observed_mask_pixels_sha256':hashlib.sha256(result.tobytes()).hexdigest()}


def collar_contacts(box, mask, regions, source):
    """Observe lead segments around a body with no discrete angle catalogue."""
    x0,y0,x1,y1=box;h,w=mask.shape
    radius=max(5,round(min(x1-x0,y1-y0)*.24));left,top=max(0,x0-radius),max(0,y0-radius);right,bottom=min(w,x1+radius),min(h,y1+radius)
    yy,xx=np.mgrid[top:bottom,left:right]
    outside=(xx<x0)|(xx>=x1)|(yy<y0)|(yy>=y1)
    local=(mask[top:bottom,left:right]&outside.astype(np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(local,8);out=[]
    for i in range(1,count):
        ys,xs=np.nonzero(labels==i);points=np.column_stack((xs+left,ys+top)).astype(float)
        if len(points)<4:continue
        delta=np.maximum.reduce([x0-points[:,0],points[:,0]-(x1-1),y0-points[:,1],points[:,1]-(y1-1),np.zeros(len(points))])
        near=delta<=max(2,radius*.3)
        if not near.any():continue
        center=points.mean(axis=0);cov=np.cov(points.T);values,vectors=np.linalg.eigh(cov);axis=vectors[:,-1]
        if values[-1]<4*max(values[0],.3):continue
        if min(abs(axis))<.2:continue
        if axis@(center-np.array([(x0+x1-1)/2,(y0+y1-1)/2]))<0:axis=-axis
        projection=points@axis;length=float(projection.max()-projection.min())
        if length<radius*.65:continue
        boundary=points[near];projection=boundary@axis;edge=boundary[projection<=projection.min()+max(1.,2*np.sqrt(max(values[0],.25)))];normal=np.array([-axis[1],axis[0]])
        point=edge[np.argmin(abs((edge-center)@normal))].astype(int);x,y=map(int,point)
        region=int(regions[y,x])
        if not region:continue
        # Straight lead evidence must continue into native component ink.
        inward=[]
        for t in np.arange(1,2*radius,.5):
            a,b=np.rint(point-axis*t).astype(int)
            if x0<=a<x1 and y0<=b<y1 and source[b,a]:inward.append((int(a),int(b)))
        if len(inward)<2:continue
        side=('right' if axis[0]>0 else 'left') if abs(axis[0])>=abs(axis[1]) else ('bottom' if axis[1]>0 else 'top')
        out.append({'side':side,'point_px':[x,y],'ink_region':region,'edge_contact_width_px':max(1,round(2*np.sqrt(max(values[0],.25)))),'boundary_gap_px':float(delta.min()),'consecutive_wire_support_px':length,'direction_from_body':axis.tolist(),'contact_axis':'continuous_source_collar','contact_policy':'native_continuous_collar'})
    return sorted(out,key=lambda c:(c['point_px'][0],c['point_px'][1]))
