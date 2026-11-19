"""Subprocess stages release model memory before the next stage starts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from .io import load, save, sha256


def detect(config, root, source, output):
    from ultralytics import YOLO
    detector=config['detector']
    weights=root/detector['weights']
    if sha256(weights)!=detector['sha256']:
        raise ValueError('Our selected YOLO checkpoint changed')
    model=YOLO(str(weights))
    result=model.predict(str(source), **detector['predict_kwargs'])[0]
    rows=[]
    for box in result.boxes:
        cls=int(box.cls.item());bbox=box.xyxy[0].tolist()
        rows.append({'type':detector['class_map'][str(cls)],'bbox':bbox,
            'confidence':float(box.conf.item())})
    rows.sort(key=lambda d:(d['bbox'][1],d['bbox'][0],d['type']))
    components=[];junctions=[]
    for row in rows:
        if row['type']=='junction':
            junctions.append({'id':f'j{len(junctions)+1}', **row})
        else:
            components.append({'id':f'c{len(components)+1}', **row,'pins':[],
                'value':None,'unit':None,'evidence':[{'source':'our_yolo26n','confidence':row['confidence']}]})
    if not components:
        raise ValueError('YOLO found no electrical components')
    if len(components)>32:
        raise ValueError('First build accepts at most 32 components in one cropped circuit')
    save(output/'detections.json',{'components':components,'junctions':junctions,
        'checkpoint_sha256':detector['sha256'],'inference':detector['predict_kwargs']})


def segment(config, root, source, output):
    import psutil
    from ._sam_backend import CircuitVisionBackend
    lock=root/config['sam']['lock']
    resource_guards=[]
    def reserve(stage):
        before=psutil.virtual_memory().available
        if stage in ('after_load','after_inference'):
            import gc,torch
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
        available=psutil.virtual_memory().available
        resource_guards.append({'stage':stage,'available_before_cleanup_bytes':before,'available_bytes':available})
        if available < 4*1024**3:
            raise RuntimeError(f'Insufficient free memory for the SAM stage at {stage}: {available/1024**3:.2f} GiB available')
    with CircuitVisionBackend.from_lock(lock, expected_sha256=config['sam']['sha256'], guard=reserve) as backend:
        backend.load(allow_model_load=True)
        result=backend.predict(source,output/'sam',expected_image_sha256=sha256(source))
    result['resource_guards']=resource_guards
    save(output/'segmentation.json',result)


def graph(config, root, source, output):
    from PIL import Image
    from .graph import extract
    perception=load(output/'detections.json')
    segmentation=load(output/'segmentation.json')
    result=extract(source,perception,segmentation['mask_path'],output/'geometry',
        junction_convention=config['graph']['junction_convention'])
    with Image.open(source) as image:
        result['image']['size']=list(image.size)
    save(output/'candidate.json',result)


if __name__=='__main__':
    stage,config_path,root,source,output=sys.argv[1:]
    {'detect':detect,'segment':segment,'graph':graph}[stage](load(config_path),Path(root),Path(source),Path(output))
