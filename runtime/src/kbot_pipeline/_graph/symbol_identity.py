"""Physical glyph identity before component/terminal assembly.

Two detector proposals are one earth-ground object only when the existing
native recognizer witnesses the identical three bar footprints. Overlapping
boxes, shared nets or similar electrical values are never identity evidence.
"""
import copy
import numpy as np
from ._topology_observers import _earth_ground_down


def earth_ground_observation(image,bounds):
    raster=np.asarray(image);height,width=raster.shape[:2];box=list(bounds);matches=[]
    def original(point,rotation):
        x,y=point
        return [(x,y),(width-y,x),(width-x,height-y),(y,height-x)][rotation]
    for rotation in range(4):
        observed=_earth_ground_down(raster,box,observe=True)
        if observed is not None:
            bars=[]
            for a,b,c,d in observed['bar_boxes_px']:
                points=np.array([original(p,rotation) for p in ((a,b),(c,b),(c,d),(a,d))])
                bars.append([int(points[:,0].min()),int(points[:,1].min()),int(points[:,0].max()),int(points[:,1].max())])
            matches.append({'kind':'earth_ground','rotation_quarters':rotation,'bar_boxes_px':bars,'shape_signature':observed['signature'],
                'evidence':'identical_native_three_bar_footprints_with_stem_and_two_contrasts'})
        current_width=raster.shape[1];a,b,c,d=box;box=[b,current_width-c,d,current_width-a];raster=np.rot90(raster)
    return matches[0] if len(matches)==1 else None


def physical_components(image,components):
    """Retain raw proposals as evidence; coalesce only proved glyph identity."""
    result=[];groups={};events=[]
    for item in components:
        component=copy.deepcopy(item)
        proof=earth_ground_observation(image,component['bbox']) if component['type']=='ground' else None
        if proof is None:result.append(component);continue
        # Detector extents may include the adjacent rail/stem. The component
        # body is the witnessed bars, with one native stroke of margin. Keeping
        # the rail outside the body prevents it becoming a second terminal.
        bars=proof['bar_boxes_px'];stroke=max(min(b[2]-b[0],b[3]-b[1]) for b in bars)
        height,width=np.asarray(image).shape[:2]
        component['bbox']=[max(0,min(b[0] for b in bars)-stroke),max(0,min(b[1] for b in bars)-stroke),
            min(width,max(b[2] for b in bars)+stroke),min(height,max(b[3] for b in bars)+stroke)]
        component['body_bbox_evidence']='native_bar_footprint_with_one_observed_stroke_margin'
        identity=tuple(tuple(b) for b in proof['bar_boxes_px'])
        if identity not in groups:
            component['physical_symbol_observation']=proof
            component['detector_proposals']=[copy.deepcopy(item)];result.append(component);groups[identity]=component;continue
        existing=groups[identity];existing['detector_proposals'].append(copy.deepcopy(item))
        existing['confidence']=max(existing.get('confidence',0.),component.get('confidence',0.))
        events.append({'kept_id':existing['id'],'duplicate_id':component['id'],'physical_symbol_observation':proof})
    return result,events
