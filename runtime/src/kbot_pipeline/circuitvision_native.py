"""Local CPU replay of pinned CircuitVision topology, with raw contact evidence.

Only selected, hash-checked AST method bodies are executed. The app constructor,
YOLO/SAM models, cloud clients, VLM enrichment and netlist guessing are not loaded.
Native node zero remains an upstream heuristic claim, not verified ground.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace
import cv2
import numpy as np
from PIL import Image

COMMIT = 'fc5b775043b577668906b6d0bbb238a690a82965'
SOURCE_HASHES = {
    'src/circuit_analyzer.py':'92da32b7010b02ebf3b48551efcc63b6ffd798d332c0645b38eb706ae637900d',
    'src/analysis_pipeline.py':'7f74322170a4c6b9dabe29556dee8aa965f37dcbadd9182a31ef2f1bc7bfb855',
    'app.py':'18ff0266f24b7f6384208acddecf5be3213ab12c02f39c7296afdd9cd00c48a8',
}
METHODS = ('enhance_lines','segment_circuit','get_contours','resize_bboxes',
    'resize_image_keep_aspect','is_point_near_bbox',
    '_are_bboxes_proximal_for_clustering','_component_has_nearby_text',
    'crop_image_and_adjust_bboxes','get_node_connections',
    'reclassify_terminals_based_on_connectivity','generate_netlist_from_nodes',
    '_get_terminal_nodes_relative_to_bbox')
CLASS_MAP = {'op_amp':'operational_amplifier','ground':'gnd',
    'voltage_source':'voltage.dc','current_source':'current.dc',
    'dependent_voltage_source':'voltage.dependent',
    'dependent_current_source':'current.dependent','crossing':'crossover'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    return value


def _native(vendor_root):
    """Compile original method ASTs only; no original module top-level execution."""
    root=Path(vendor_root).resolve()
    for relative,digest in SOURCE_HASHES.items():
        if sha256(root/relative)!=digest:raise ValueError('Pinned upstream source mismatch: '+relative)
    actual=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if actual!=COMMIT:raise ValueError('Unexpected upstream checkout commit')
    source=(root/'src/circuit_analyzer.py').read_text()
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CircuitAnalyzer')
    methods={n.name:n for n in original.body if isinstance(n,ast.FunctionDef)}
    selected=[methods[name] for name in METHODS]
    if any(isinstance(n,(ast.Import,ast.ImportFrom)) for m in selected for n in ast.walk(m)):
        raise ValueError('Unexpected imports inside native method boundary')
    isolated=ast.ClassDef(name='_IsolatedNativeTopology',bases=[],keywords=[],body=selected,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[isolated],type_ignores=[]))
    namespace={'cv2':cv2,'np':np,'deepcopy':deepcopy}
    exec(compile(module,str(root/'src/circuit_analyzer.py'),'exec'),namespace)
    native=namespace['_IsolatedNativeTopology']()
    literal_attributes={'non_components','source_components','netlist_map',
        'voltage_classes_names','current_source_classes_names','diode_classes_names'}
    for node in methods['__init__'].body:
        if not isinstance(node,ast.Assign):continue
        for target in node.targets:
            if isinstance(target,ast.Attribute) and target.attr in literal_attributes:
                value=set(ast.literal_eval(node.value.args[0])) if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='set' else ast.literal_eval(node.value)
                setattr(native,target.attr,value)
    if any(not hasattr(native,name) for name in literal_attributes):raise ValueError('Required native constructor literals missing')
    # Exact upstream debug=False would reference an uninitialized local in
    # get_node_connections. Debug=True changes only diagnostics/visual drawings.
    native.debug=True
    native.show_image=lambda *args,**kwargs:None
    native.yolo=SimpleNamespace(model=SimpleNamespace(names={}))
    identities={name:{'line':methods[name].lineno,'sha256':hashlib.sha256(ast.get_source_segment(source,methods[name]).encode()).hexdigest()} for name in METHODS}
    return native,identities


def adapt_components(components,image_size):
    """Accept vendor dictionaries or Ckt-Bot id/type/bbox; retain stable identity."""
    width,height=image_size;boxes=[];mapping=[];seen=set()
    for component in components:
        raw='class' in component and all(k in component for k in ('xmin','ymin','xmax','ymax'))
        uid=component.get('persistent_uid',component.get('id'))
        if not isinstance(uid,str) or not uid or uid in seen:raise ValueError('Unique persistent component IDs required')
        seen.add(uid)
        kind=component['class'] if raw else component['type']
        cls=kind if raw else CLASS_MAP.get(kind,kind)
        coords=[component[k] for k in ('xmin','ymin','xmax','ymax')] if raw else list(component['bbox'])
        if len(coords)!=4 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in coords):raise ValueError('Finite native bbox required')
        x0,y0,x1,y1=coords
        if not(0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('BBox outside original image frame')
        box=dict(zip(('xmin','ymin','xmax','ymax'),coords));box.update({'class':cls,'persistent_uid':uid})
        for key in ('confidence','_yolo_class_id_temp','semantic_direction'):
            if key in component:box[key]=deepcopy(component[key])
        boxes.append(box);mapping.append({'component_id':uid,'input_class':kind,'upstream_class':cls,'mapping':'vendor_identity' if raw else 'explicit_alias' if cls!=kind else 'identity'})
    return boxes,mapping


def _frame(native,image,boxes,crop_policy):
    if crop_policy=='upstream':
        cropped,adjusted,debug=native.crop_image_and_adjust_bboxes(image,deepcopy(boxes),padding=80)
        roi=debug['final_crop_window_abs'] if debug.get('crop_applied') else [0,0,image.shape[1],image.shape[0]]
    elif crop_policy=='full_frame':
        cropped=image.copy();adjusted=deepcopy(boxes);roi=[0,0,image.shape[1],image.shape[0]]
        debug={'crop_applied':False,'reason_for_no_crop':'explicit_full_frame_control'}
    else:raise ValueError('crop_policy must be upstream or full_frame')
    return cropped,adjusted,list(roi),debug


def prepare_native_frame(image_path,components,*,vendor_root,box_source,crop_policy='upstream'):
    """Plan native crop before SAM inference; no segmentation/model is run."""
    native,methods=_native(vendor_root)
    with Image.open(image_path) as source:image=np.array(source.convert('RGB'))
    boxes,mapping=adapt_components(components,(image.shape[1],image.shape[0]));log=io.StringIO()
    with redirect_stdout(log):crop,adjusted,roi,debug=_frame(native,image,boxes,crop_policy)
    return _plain({'schema_version':'circuitvision.native-frame/1','image_sha256':sha256(image_path),
        'box_source':box_source,'crop_policy':crop_policy,'roi_xyxy':roi,'size_wh':[crop.shape[1],crop.shape[0]],
        'bboxes':adjusted,'input_bboxes':boxes,'class_mapping':mapping,
        'dropped_component_ids':[b['persistent_uid'] for b in boxes if b['persistent_uid'] not in {a['persistent_uid'] for a in adjusted}],
        'crop_debug':debug,'upstream_commit':COMMIT,'source_hashes':SOURCE_HASHES,'method_hashes':methods,'log':log.getvalue()})


def _provisional_netlist(native,nodes):
    copied=deepcopy(nodes);incidences={}
    for node in copied:
        if node.get('contour') is not None:node['contour']=np.asarray(node['contour'],dtype=np.int32)
        for component in node['components']:
            # This is the explicit no-VLM export condition only. Raw nodes remain
            # unchanged; unknown native ordering and loss of extra pins are kept.
            component['semantic_direction']='UNKNOWN';component['semantic_reason']='UNKNOWN'
            uid=component.get('persistent_uid')
            if uid is not None:incidences.setdefault(uid,[]).append(node['id'])
    lines=native.generate_netlist_from_nodes(copied)
    incidence_report=[]
    for uid,observed in incidences.items():
        emitted=[line for line in lines if line.get('persistent_uid')==uid]
        assigned=[line[key] for line in emitted for key in ('node_1','node_2')]
        observed=list(dict.fromkeys(observed))
        incidence_report.append({'component_id':uid,'observed_node_ids':observed,'emitted_node_ids':assigned,
            'omitted_observed_node_ids':[n for n in observed if str(n) not in {str(v) for v in assigned}],
            'emitted_without_observed_incidence':[n for n in assigned if str(n) not in {str(v) for v in observed}]})
    return {'schema_version':'circuitvision.native-provisional-netlist/1','status':'provisional_not_product_netlist',
        'semantic_condition':'UNKNOWN direction and reason; no VLM',
        'lines':_plain(lines),'incidence_report':_plain(incidence_report),'verified':False,
        'limitations':['Native generator emits only node_1/node_2 even for components with three or more incidences.',
            'Native UNKNOWN direction uses upstream arbitrary/reversed node ordering.',
            'Native terminal records can add ground zero without observed incidence; this claim is preserved.',
            'Values remain native string None placeholders; no semantic pins or valid simulation model are asserted.']}


def generate_native_netlist(upstream_nodes,*,vendor_root):
    """Export saved raw nodes without rerunning topology or calling any model."""
    native,methods=_native(vendor_root);log=io.StringIO()
    with redirect_stdout(log):result=_provisional_netlist(native,upstream_nodes)
    result.update(upstream_commit=COMMIT,source_hashes=SOURCE_HASHES,
        method_hashes={name:methods[name] for name in ('generate_netlist_from_nodes','_get_terminal_nodes_relative_to_bbox')},log=log.getvalue())
    return result


def point_bbox_distance(point,bbox):
    """Euclidean distance to the closed processing-space rectangle."""
    x,y=map(float,point)
    dx=max(float(bbox['xmin'])-x,0.,x-float(bbox['xmax']))
    dy=max(float(bbox['ymin'])-y,0.,y-float(bbox['ymax']))
    return math.hypot(dx,dy)


def run_native_topology(image_path,mask_path,components,output_dir,*,vendor_root,box_source,
        expected_image_sha256,expected_mask_sha256,crop_policy='full_frame',mask_frame='full_frame',
        mask_origin='full_frame_inference',mask_roi_xyxy=None,terminal_reclassification='off',detector_names=None,
        contact_policy='upstream',export_netlist=False):
    """Replay saved native binary masks; return raw nodes and contour contacts.

    mask_origin is explicit provenance: full_frame_inference, native_crop_inference,
    or authored_control. Cropping a full-frame prediction is never labelled native
    crop inference. Native crop inference needs the exact matching mask_roi_xyxy.
    mask_frame describes raster storage: full_frame or analysis_frame.
    contact_policy='euclidean_bbox_v1' changes only node-stage proximity, with
    unchanged upstream thresholds. Native terminal reclassification stays exact.
    """
    if not isinstance(export_netlist,bool):raise ValueError('export_netlist must be explicit boolean')
    if contact_policy not in ('upstream','euclidean_bbox_v1'):raise ValueError('Unknown native contact policy')
    if sha256(image_path)!=expected_image_sha256 or sha256(mask_path)!=expected_mask_sha256:raise ValueError('Input source/mask SHA256 mismatch')
    native,methods=_native(vendor_root)
    with Image.open(image_path) as source:image=np.array(source.convert('RGB'))
    height,width=image.shape[:2]
    if height*width>20_000_000:raise ValueError('Native CPU replay image budget exceeded')
    boxes,mapping=adapt_components(components,(width,height))
    if len(boxes)>1000:raise ValueError('Native CPU replay component budget exceeded')
    log=io.StringIO()
    with redirect_stdout(log):crop,adjusted,roi,crop_debug=_frame(native,image,boxes,crop_policy)
    x0,y0,x1,y1=roi;ch,cw=crop.shape[:2]
    if int(600*cw/ch)*600>12_000_000:raise ValueError('Native resized processing budget exceeded')
    if mask_origin not in ('full_frame_inference','native_crop_inference','authored_control'):raise ValueError('Unknown saved mask origin')
    if mask_origin=='native_crop_inference' and list(mask_roi_xyxy or [])!=roi:raise ValueError('Crop inference ROI differs from native crop plan')
    with Image.open(mask_path) as saved:
        if saved.mode!='L':raise ValueError('Binary mask must be uint8 single channel')
        raster=np.array(saved)
    if not np.isin(raster,[0,255]).all():raise ValueError('Native binary mask must use exactly 0/255')
    if mask_frame=='full_frame':
        if raster.shape!=(height,width):raise ValueError('Full-frame mask shape mismatch')
        processing=raster[y0:y1,x0:x1].copy()
    elif mask_frame=='analysis_frame':
        if raster.shape!=(ch,cw):raise ValueError('Analysis-frame mask shape mismatch')
        if mask_origin=='full_frame_inference' and roi!=[0,0,width,height]:raise ValueError('Cropped raster cannot masquerade as full-frame prediction')
        processing=raster.copy()
    else:raise ValueError('Unknown mask coordinate frame')
    if terminal_reclassification not in ('off','upstream_after_crop'):raise ValueError('Unknown terminal reclassification condition')
    before_reclassification=deepcopy(adjusted);prelim_artifacts={}
    if terminal_reclassification=='upstream_after_crop':
        if detector_names is None:raise ValueError('Explicit detector class-name lookup required for native reclassification')
        native.yolo.model.names={int(k):v for k,v in detector_names.items()}
        native.show_image=lambda value,title:prelim_artifacts.setdefault(title,value.copy())
        # Match analysis_pipeline wrapper BGR->RGB, then the unmodified method.
        with redirect_stdout(log):native.reclassify_terminals_based_on_connectivity(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),adjusted)
        native.show_image=lambda *args,**kwargs:None
    captured={};accepted=[]
    predicate_audit={'tested':0,'upstream_accepted':0,'selected_accepted':0,'upstream_accepted_outside_euclidean':0}
    original_contours=native.get_contours;original_resize=native.resize_image_keep_aspect;original_near=native.is_point_near_bbox
    def capture_contours(self,img,area_threshold=.00040):
        captured['enhanced_before_contour_mutation']=img.copy()
        captured['contour_foreground']=(255-img if cv2.mean(img)[0]>127 else img)>0
        value=original_contours(img,area_threshold);captured['contours']=deepcopy(value[0]);return value
    def capture_resize(self,img,bboxes,new_height=600):
        value=original_resize(img,bboxes,new_height);captured['processing_mask']=value[0].copy();captured['resized_bboxes']=deepcopy(value[1]);return value
    def capture_near(self,point,bbox,pixel_threshold):
        upstream=original_near(point,bbox,pixel_threshold)
        distance=point_bbox_distance(point,bbox);euclidean=distance<=pixel_threshold
        result=upstream if contact_policy=='upstream' else euclidean
        predicate_audit['tested']+=1;predicate_audit['upstream_accepted']+=int(upstream)
        predicate_audit['selected_accepted']+=int(result)
        predicate_audit['upstream_accepted_outside_euclidean']+=int(upstream and not euclidean)
        if result:accepted.append({'component_id':bbox['persistent_uid'],'point_processing_px':list(point),
            'threshold_processing_px':pixel_threshold,'distance_to_bbox_processing_px':distance,'contact_policy':contact_policy})
        return result
    native.get_contours=MethodType(capture_contours,native);native.resize_image_keep_aspect=MethodType(capture_resize,native);native.is_point_near_bbox=MethodType(capture_near,native)
    previous_threads=cv2.getNumThreads();cv2.setNumThreads(1)
    try:
        with redirect_stdout(log):raw=native.get_node_connections(crop,processing.copy(),deepcopy(adjusted))
    finally:cv2.setNumThreads(previous_threads)
    nodes,emptied,enhanced,contour_viz,node_viz,contact_viz=raw
    ph,pw=enhanced.shape[:2];sx=pw/cw;sy=ph/ch
    def point_native(point):return [(float(point[0])+.5)/sx+x0,(float(point[1])+.5)/sy+y0]
    def contour_key(contour):return hashlib.sha256(np.asarray(contour,dtype=np.int32).tobytes()).hexdigest()
    final_by_contour={contour_key(n['contour']):n for n in nodes}
    contours=[]
    for item in captured.get('contours',[]):
        key=contour_key(item['contour']);final=final_by_contour.get(key)
        contours.append(dict(_plain(item),contour_sha256=key,native_node_id=final['id'] if final else None,
            contour_native_px=[point_native(p[0]) for p in item['contour']]))
    node_regions=np.zeros((ph,pw),dtype=np.uint16)
    for node in nodes:
        if not 0<=node['id']<65535:raise ValueError('Native node label budget exceeded')
        filled=np.zeros((ph,pw),np.uint8);cv2.drawContours(filled,[node['contour']],-1,1,-1)
        support=(filled>0)&captured['contour_foreground']
        if np.any(support&(node_regions!=0)):raise ValueError('Overlapping native external-contour regions')
        node_regions[support]=int(node['id'])+1
    contacts=[]
    for index,event in enumerate(accepted):
        matching=[c for c in contours if any(p[0]==_plain(event['point_processing_px']) for p in c['contour'])]
        node_ids=[c['native_node_id'] for c in matching if c['native_node_id'] is not None and any(b['persistent_uid']==event['component_id'] for b in final_by_contour[c['contour_sha256']]['components'])]
        contacts.append(dict(_plain(event),id=f'contact-{index}',point_native_px=point_native(event['point_processing_px']),
            contour_ids=[c['id'] for c in matching],node_ids=node_ids,
            kind='upstream_first_accepted_contour_vertex_bbox_proximity',physical_terminal_px=None,semantic_role=None))
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    artifact_arrays={'analysis-source.png':cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),'input-mask.png':processing,
        'emptied-mask.png':emptied,'resized-mask.png':captured['processing_mask'],
        'enhancedprecontours.png':captured['enhanced_before_contour_mutation'],
        'native-node-regions.png':node_regions,
        'enhanced-returned.png':enhanced,'contours.png':contour_viz,'nodes.png':node_viz,'contact-points.png':contact_viz}
    for i,(title,arr) in enumerate(prelim_artifacts.items()):artifact_arrays[f'reclassification-{i}.png']=arr
    artifacts={}
    for name,value in artifact_arrays.items():
        path=out/name
        if not cv2.imwrite(str(path),value):raise RuntimeError('Failed writing native diagnostic image')
        artifacts[name]={'path':str(path),'sha256':sha256(path),'shape':list(value.shape)}
    region_path=out/'native-node-regions.npy';np.save(region_path,node_regions,allow_pickle=False)
    artifacts['native-node-regions.npy']={'path':str(region_path),'sha256':sha256(region_path),'shape':[ph,pw],'dtype':'uint16','encoding':'0 background; native node id + 1; exact nonzero contour input intersect filled surviving contour'}
    provisional=None
    if export_netlist:
        with redirect_stdout(log):provisional=_provisional_netlist(native,nodes)
        netlist_path=out/'raw-native-netlist.json'
        netlist_path.write_text(json.dumps(provisional['lines'],indent=2,allow_nan=False)+'\n')
        provisional['artifact']={'path':str(netlist_path),'sha256':sha256(netlist_path)}
        provisional['method_hashes']={name:methods[name] for name in ('generate_netlist_from_nodes','_get_terminal_nodes_relative_to_bbox')}
    (out/'upstream.log').write_text(log.getvalue())
    raw_path=out/'raw-upstream-nodes.json';raw_path.write_text(json.dumps(_plain(nodes),indent=2)+'\n')
    result={'schema_version':'circuitvision.native-topology/1','status':'native_candidate','upstream_commit':COMMIT,
        'provenance':{'source_hashes':SOURCE_HASHES,'method_hashes':methods,'image_sha256':expected_image_sha256,
            'mask_sha256':expected_mask_sha256,'box_source':box_source,'mask_origin':mask_origin,'mask_frame':mask_frame,
            'mask_condition':'full_frame_prediction_then_crop' if mask_origin=='full_frame_inference' and roi!=[0,0,width,height] else mask_origin,
            'terminal_reclassification':terminal_reclassification,'detector_names':detector_names,
            'contact_policy':contact_policy,'contact_predicate_audit':predicate_audit,
            'adaptations':['Hash-checked AST method extraction; no app constructor/imports',
                'debug=True avoids upstream uninitialized debug local; show_image captured/no-op; logs retained',
                'Caller supplies saved binary SAM mask and boxes; no detector/segmentation model loaded',
                'Cloud semantic enrichment omitted; optional local netlist export is separately marked provisional, with UNKNOWN semantics',
                'Method wrappers capture returned intermediate arrays and successful proximity calls',
                'Node proximity uses exact upstream predicate' if contact_policy=='upstream' else 'Node proximity replaced by Euclidean closed-bbox distance at unchanged native thresholds; native method body and reclassification unchanged']},
        'frames':{'original_size_wh':[width,height],'analysis_size_wh':[cw,ch],'processing_size_wh':[pw,ph],
            'crop_roi_xyxy':roi,'crop_policy':crop_policy,'crop_debug':crop_debug,'processing_scale_xy':[sx,sy],
            'processing_vertex_to_native_center_matrix':[[1/sx,0,x0+.5/sx],[0,1/sy,y0+.5/sy],[0,0,1]],
            'bbox_boundary_inverse_matrix':[[1/sx,0,x0],[0,1/sy,y0],[0,0,1]],
            'coordinate_note':'OpenCV contour integer indices map to original image pixel-center coordinates; bbox integer truncation is retained upstream.'},
        'input_bboxes':boxes,'analysis_bboxes_before_reclassification':before_reclassification,'analysis_bboxes':adjusted,
        'processing_bboxes':captured['resized_bboxes'],'class_mapping':mapping,
        'dropped_component_ids':[b['persistent_uid'] for b in boxes if b['persistent_uid'] not in {a['persistent_uid'] for a in adjusted}],
        'upstream_nodes':nodes,'upstream_contours':contours,'contact_evidence':contacts,
        **({'native_netlist_export':provisional} if provisional is not None else {}),
        'native_ground_claim':{'node_id':0 if nodes else None,'basis':'lowest source-connected node, else max-degree/lowest fallback','physical_ground_verified':False},
        'artifacts':artifacts,'raw_nodes_artifact':{'path':str(raw_path),'sha256':sha256(raw_path)},
        'limitations':['Node incidence is one component per contour; native pins/roles are not supplied.',
            'Native enhancement can bridge gaps; native contours do not resolve nonconnecting crossings.',
            'Native proximity checks contour vertices against bbox edges; these are not body-terminal observations.',
            'Native ground zero is a heuristic; singleton nodes may be discarded by upstream filtering.'],
        'verified':False}
    result=_plain(result);(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
