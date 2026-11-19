"""Opt-in current-source roles from the existing isolated printed-arrow observer.

No model, head heuristic, box orientation or new pin coordinate is introduced.
A printed arrow is a reference direction, independent of a signed current value.
"""
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from reader_pipeline.rev3.printed_binding_pixels import acquire_printed_binding_pixels

POLICY='printed_source_arrow_v1'
SOURCE_ALIASES={'current_source':'current_source','independent_current_source':'current_source',
    'dependent_current_source':'dependent_current_source','current.dc':'current_source',
    'current.ac':'current_source','current.dependent':'dependent_current_source'}
MIN_OWNERSHIP_MARGIN_PX=3.
MAX_AXIS_ANGLE_DEGREES=15.


def _source_path(anchor,center,body,gray,threshold):
    """Only witness existing source ink from anchor to the observed closed body."""
    a=np.floor(anchor).astype(int);b=np.floor(center).astype(int)
    steps=int(np.max(np.abs(b-a)))
    if steps==0:return None
    pixels=[];previous=None
    for t in np.linspace(0,1,steps+1):
        point=np.rint(a+(b-a)*t).astype(int);x,y=map(int,point)
        if previous==(x,y):continue
        previous=(x,y)
        if not(0<=x<gray.shape[1] and 0<=y<gray.shape[0]):return None
        inside=cv2.pointPolygonTest(body,(x+.5,y+.5),False)>=0
        if inside:
            if pixels:return pixels
            return [[x,y]] if gray[y,x]<=threshold else None
        if gray[y,x]>threshold:return None
        pixels.append([x,y])
    return None


def infer_current_source_roles(image_path,components,pins,current_role_policy='off'):
    """Return assemble-compatible {pins, unresolved} and retained pixel evidence.

    Preregistered binding: exactly two supported anchors; endpoint ownership
    margin >=3px; opposed projected anchor axis within15deg of observed arrow;
    each anchor within max(3px, observed shaft thickness) of that arrow axis and
    outside the closed body's interior. Real source ink must connect each anchor
    to that body. Tail binds from, tip binds to. Default off performs no I/O.
    """
    result={'pins':[],'unresolved':[],'evidence':{'current_role_policy':current_role_policy}}
    if current_role_policy=='off':return result
    if current_role_policy!=POLICY:raise ValueError('Unknown current-source role policy')
    image=Image.open(image_path).convert('RGB');w,h=image.size
    source_sha=hashlib.sha256(Path(image_path).read_bytes()).hexdigest();rgb_sha=hashlib.sha256(image.tobytes()).hexdigest()
    selected=[];ids=set()
    for c in components:
        if c['id'] in ids:raise ValueError('Duplicate component id')
        ids.add(c['id'])
        if c['type'] not in SOURCE_ALIASES:continue
        box=c['bbox']
        if len(box)!=4 or not all(math.isfinite(v) for v in box) or not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):raise ValueError('Invalid source bbox')
        selected.append({'detector_id':c['id'],'class_name':SOURCE_ALIASES[c['type']],'bbox_px':list(box)})
    inventory={'observations':{'image':{'width':w,'height':h,'sha256':source_sha}},'components':selected}
    pixels=acquire_printed_binding_pixels(image,inventory,canonical_rgb_sha256=rgb_sha)
    result['pixel_observations']=pixels
    result['evidence'].update(image_sha256=source_sha,rgb_sha256=rgb_sha,
        observer_policy=pixels['policy_version'],ownership_margin_px=MIN_OWNERSHIP_MARGIN_PX,
        axis_angle_limit_degrees=MAX_AXIS_ANGLE_DEGREES,
        implementation_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'reader_pipeline/rev3/printed_binding_pixels.py',Path(__file__).resolve().parents[1]/'reader_pipeline/rev3/legacy/visual_annotations.py')})
    gray=np.asarray(image.convert('L'));threshold=pixels['thresholding']['level']
    for c in selected:
        uid=c['detector_id']
        def reject(reason):result['unresolved'].append({'component_id':uid,'reason':reason,'policy':POLICY})
        candidates=[p for p in pins if p['component_id']==uid]
        if len(candidates)!=2 or any(p.get('state')!='supported' or p.get('anchor_px') is None for p in candidates):
            reject('current_source_requires_exactly_two_supported_anchors');continue
        if len({p['id'] for p in candidates})!=2:raise ValueError('Duplicate source pin id')
        if any(len(p['anchor_px'])!=2 or not all(math.isfinite(v) for v in p['anchor_px']) or not(0<=p['anchor_px'][0]<w and 0<=p['anchor_px'][1]<h) for p in candidates):raise ValueError('Source anchor outside image')
        witnesses=[v for v in pixels['body_arrow_witnesses'] if v['component_id']==uid]
        summary=next((v for v in pixels['source_observations'] if v['component_id']==uid),{})
        if pixels['status']!='observed' or len(witnesses)!=1 or summary.get('ambiguous_body_interiors',0):
            reject('isolated_closed_body_current_arrow_not_unique');continue
        witness=witnesses[0];ink=witness['ink_bbox_px']
        if any(other['id']!=uid and other['bbox'][0]<=ink[0] and other['bbox'][1]<=ink[1] and other['bbox'][2]>=ink[2] and other['bbox'][3]>=ink[3] for other in components):
            reject('current_arrow_component_ownership_competing');continue
        arrow=witness['pixel_arrow'];tail=np.array(arrow['tail_px']);tip=np.array(arrow['tip_px']);axis=np.array(arrow['unit_vector_xy']);axis/=np.linalg.norm(axis)
        body=np.array(witness['body_contour_px'],dtype=np.float32).reshape(-1,1,2)
        ranked_tail=sorted((math.dist(p['anchor_px'],tail),p['id'],p) for p in candidates)
        ranked_tip=sorted((math.dist(p['anchor_px'],tip),p['id'],p) for p in candidates)
        margins=[ranked_tail[1][0]-ranked_tail[0][0],ranked_tip[1][0]-ranked_tip[0][0]]
        if min(margins)<MIN_OWNERSHIP_MARGIN_PX or ranked_tail[0][1]==ranked_tip[0][1]:
            reject('current_arrow_endpoint_anchor_ownership_ambiguous');continue
        source_pin,target_pin=ranked_tail[0][2],ranked_tip[0][2]
        a=np.array(source_pin['anchor_px']);b=np.array(target_pin['anchor_px']);delta=b-a;length=float(np.linalg.norm(delta))
        if length==0:reject('current_source_anchor_axis_degenerate');continue
        cosine=float((delta/length)@axis)
        perpendicular=lambda p:abs(float((p-tail)[0]*axis[1]-(p-tail)[1]*axis[0]))
        band=max(3.,float(arrow['evidence']['tail_thickness_px']))
        if (cosine<math.cos(math.radians(MAX_AXIS_ANGLE_DEGREES)) or
            float((a-tail)@axis)>=0 or float((b-tip)@axis)<=0 or
            max(perpendicular(a),perpendicular(b))>band or
            any(cv2.pointPolygonTest(body,tuple(map(float,p['anchor_px'])),False)>0 for p in candidates)):
            reject('current_source_anchors_not_opposed_on_observed_arrow_axis');continue
        paths=[_source_path(p['anchor_px'],(tail+tip)/2,body,gray,threshold) for p in (source_pin,target_pin)]
        if not all(paths):reject('current_source_anchor_to_body_source_ink_disconnected');continue
        assignments=[('from',source_pin),('to',target_pin)]
        if any(p.get('role') not in (None,role) for role,p in assignments):reject('current_source_existing_role_conflict');continue
        binding={'policy':POLICY,'witness_id':witness['id'],'image_sha256':source_sha,
            'body_mask_sha256':witness['body_mask_sha256'],'ink_mask_sha256':witness['ink_mask_sha256'],
            'pixel_arrow':arrow,'from_pin_id':source_pin['id'],'to_pin_id':target_pin['id'],
            'axis_cosine':cosine,'perpendicular_band_px':band,'endpoint_ownership_margins_px':margins,
            'source_anchor_to_body_paths_px':paths,'physical_terminal_localization':'unchanged',
            'meaning':'printed_reference_direction_not_solved_current','verified':False}
        for role,pin in assignments:
            result['pins'].append({'component_id':uid,'pin_id':pin['id'],'role':role,
                'anchor_px':list(pin['anchor_px']),'terminal_px':pin.get('terminal_px'),
                'evidence':[f'printed_current_arrow:{witness["id"]}:image:{source_sha}',
                    'existing_isolated_shaft_and_asymmetric_arrowhead','unique_opposed_supported_anchor_binding'],
                'semantic_observations':binding})
    return result
