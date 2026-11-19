"""Local CircuitVision SAM2.1/LoRA inference, separate from its web/API app.

Upstream: https://github.com/JKc66/CircuitVision (Apache-2.0), commit below.
Construction follows circuit_analyzer.py155..250; only the hash-pinned
sam2_infer.py is imported. No YOLO, CircuitAnalyzer, API keys or circuit upload.
Importing this module does not import torch or load/download a model.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import numpy as np
from PIL import Image

UPSTREAM_COMMIT = 'fc5b775043b577668906b6d0bbb238a690a82965'
SAM2_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
UPSTREAM_URL = 'https://github.com/JKc66/CircuitVision'
SOURCE_HASHES = {
    'src/sam2_infer.py':'832d1866a994223e67a9e469a840397e8faeadbeab6e9b5f90b73f6b7699c046',
    'src/circuit_analyzer.py':'92da32b7010b02ebf3b48551efcc63b6ffd798d332c0645b38eb706ae637900d',
    'models/configs/sam2.1_hiera_l.yaml':'1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107',
}
PROVIDER_ID = 'CircuitVision/SAM2.1-HieraL-LoRA'
RESOLUTION = 1024
# Exact targets are extracted as literals from the pinned analyzer, without
# importing it. A source hash change fails before any executable import.


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _checked_file(record):
    path=Path(record['path']).expanduser().resolve()
    if not path.is_file() or sha256(path)!=record['sha256']:
        raise ValueError(f'Artifact missing or SHA256 mismatch: {path}')
    return path


def validate_lock(lock):
    """Read-only identities; model construction is always a separate action."""
    if lock.get('schema_version')!='circuitvision.model-lock/1':raise ValueError('Unsupported model lock')
    root=Path(lock['vendor_root']).resolve()
    commit=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if commit!=UPSTREAM_COMMIT or lock.get('vendor_commit')!=UPSTREAM_COMMIT:raise ValueError('Unexpected CircuitVision source commit')
    for relative,digest in SOURCE_HASHES.items():
        if sha256(root/relative)!=digest:raise ValueError(f'Pinned upstream source mismatch: {relative}')
    artifacts=lock['artifacts'];paths={name:_checked_file(artifacts[name]) for name in ('config','base_checkpoint','finetuned_checkpoint')}
    if sha256(paths['config'])!=SOURCE_HASHES['models/configs/sam2.1_hiera_l.yaml']:raise ValueError('Expected matching SAM2.1 Hiera Large config')
    sam_root=Path(lock['sam2_root']).resolve()
    sam_commit=subprocess.run(['git','-C',str(sam_root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if sam_commit!=SAM2_COMMIT or lock.get('sam2_commit')!=SAM2_COMMIT:raise ValueError('Unexpected SAM2 source commit')
    dirty=subprocess.run(['git','-C',str(sam_root),'status','--porcelain','--untracked-files=no','--','sam2'],capture_output=True,text=True,check=True).stdout.strip()
    if dirty:raise ValueError('Pinned SAM2 source has tracked changes')
    if paths['config']!=(sam_root/'sam2'/SAM2_CONFIG).resolve():raise ValueError('Config must be the pinned SAM2 package config')
    runtime=lock['runtime']
    if runtime['device'] not in ('cpu','mps','cuda','cuda:0'):raise ValueError('Unsupported explicit device')
    fraction=runtime.get('mps_fraction')
    if fraction is not None and (isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not 0<fraction<=1):raise ValueError('Invalid explicit MPS memory fraction')
    if runtime.get('torch_threads')!=1:raise ValueError('Pilot requires one explicit torch CPU thread')
    if not isinstance(runtime.get('package_versions'),dict) or not all(k in runtime['package_versions'] for k in ('torch','torchvision','peft','hydra-core')):raise ValueError('Runtime package versions must be pinned')
    return paths


def model_kwargs(lock,paths):
    import ast
    tree=ast.parse((Path(lock['vendor_root'])/'src/circuit_analyzer.py').read_text())
    groups={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('base_parts','added_parts'):groups[target.id]=ast.literal_eval(node.value)
    targets=groups['base_parts']+groups['added_parts']
    if len(targets)!=36 or len(set(targets))!=36:raise ValueError('Unexpected LoRA target inventory')
    return dict(model_cfg_path=SAM2_CONFIG,checkpoint_path=str(paths['base_checkpoint']),device=lock['runtime']['device'],
        use_high_res_features=True,use_peft=True,lora_rank=4,lora_alpha=16,lora_dropout=.3,lora_target_modules=targets,
        use_wrapper=True,trainable_embedding_r=4,use_refinement_layer=True,refinement_kernels=[3,5,7,11],kernel_channels=2,
        weight_dice=.5,weight_focal=.4,weight_iou=.3,weight_freq=.1,focal_alpha=.25)


@contextmanager
def _offline():
    """Defense in depth around hash-audited local loading/inference."""
    names=('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE');old={k:os.environ.get(k) for k in names}
    connect,connect_ex,create=socket.socket.connect,socket.socket.connect_ex,socket.create_connection
    def denied(*args,**kwargs):raise RuntimeError('CircuitVision backend is local-only; network connection blocked')
    try:
        for name in names:os.environ[name]='1'
        socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
        yield
    finally:
        socket.socket.connect=connect;socket.socket.connect_ex=connect_ex;socket.create_connection=create
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def native_prediction(image,predictor,transforms,*,device='cpu',roi=None):
    """Native inverse mapping; fake predictors can exercise this without models."""
    import torch
    if image.mode!='RGB':raise ValueError('Prediction requires canonical RGB image')
    width,height=image.size
    bounds=[0,0,width,height] if roi is None else list(roi)
    if len(bounds)!=4 or any(isinstance(v,bool) or not isinstance(v,int) for v in bounds):raise ValueError('ROI must use integer native pixel bounds')
    x0,y0,x1,y1=bounds
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('ROI outside native frame')
    crop=image.crop(tuple(bounds));tensor=transforms(crop)
    if tuple(tensor.shape)!=(3,RESOLUTION,RESOLUTION) or not torch.isfinite(tensor).all():raise ValueError('Unexpected preprocessed tensor')
    with torch.inference_mode():
        result=predictor(tensor.unsqueeze(0).to(device))
        if not isinstance(result,(tuple,list)) or len(result)!=3:raise ValueError('Expected upstream high/low/IoU tuple')
        high=result[0]
        if not isinstance(high,torch.Tensor) or tuple(high.shape)!=(1,1,RESOLUTION,RESOLUTION) or not high.is_floating_point() or not torch.isfinite(high).all():raise ValueError('Unexpected upstream high-resolution logits')
        native=transforms.postprocess_masks(high.float(),(y1-y0,x1-x0))
        if tuple(native.shape)!=(1,1,y1-y0,x1-x0) or not torch.isfinite(native).all():raise ValueError('Invalid native logit shape or values')
        local=native.detach().cpu().numpy()[0,0].astype(np.float32,copy=False)
    logits=np.full((height,width),np.nan,dtype=np.float32);logits[y0:y1,x0:x1]=local
    coverage=np.zeros((height,width),np.uint8);coverage[y0:y1,x0:x1]=1
    probability=np.full_like(logits,np.nan);positive=local>=0;values=np.empty_like(local)
    values[positive]=1/(1+np.exp(-local[positive]));exp=np.exp(local[~positive]);values[~positive]=exp/(1+exp)
    probability[y0:y1,x0:x1]=values
    # Authoritative decision is the provider's strict native-logit >0. Sigmoid
    # rounding can equal .5 even for tiny positive float32 logits.
    binary=np.zeros((height,width),np.uint8);binary[y0:y1,x0:x1]=(local>0).astype(np.uint8)*255
    return {'logits':logits,'probabilities':probability,'coverage':coverage,'binary':binary,
        'coverage_record':{'mode':'full_frame' if roi is None else 'explicit_roi','roi_xyxy':bounds,'observed_pixels':int(coverage.sum()),'total_pixels':width*height,'outside_roi':'not_observed'},
        'placement':{'crop_to_native_translation_xy':[x0,y0],'native_size_wh':[width,height],'model_size_wh':[RESOLUTION,RESOLUTION],'preprocess':'RGB ToTensor; square Resize; ImageNet Normalize','postprocess':'float32 bilinear align_corners=False before threshold'}}


def write_prediction(image_path,output_dir,prediction,*,expected_image_sha256,provenance):
    """Write new artifacts only; binary raster is authoritative, floats unquantized."""
    source=Path(image_path).resolve()
    if sha256(source)!=expected_image_sha256:raise ValueError('Source image SHA256 mismatch')
    with Image.open(source) as image:width,height=image.size
    shape=(height,width)
    for key in ('logits','probabilities','coverage','binary'):
        if prediction[key].shape!=shape:raise ValueError('Prediction/native frame shape mismatch')
    logits=prediction['logits'];probabilities=prediction['probabilities'];coverage=prediction['coverage'];binary=prediction['binary'];observed=coverage==1
    if logits.dtype!=np.float32 or probabilities.dtype!=np.float32 or coverage.dtype!=np.uint8 or binary.dtype!=np.uint8:raise ValueError('Native artifact dtype mismatch')
    if not np.isin(coverage,[0,1]).all() or not np.isin(binary,[0,255]).all() or not np.isfinite(logits[observed]).all() or not np.isfinite(probabilities[observed]).all():raise ValueError('Invalid native values')
    if not np.isnan(logits[~observed]).all() or not np.isnan(probabilities[~observed]).all() or np.any(binary[~observed]):raise ValueError('Unobserved ROI pixels must remain unknown')
    if not np.array_equal(binary[observed],(logits[observed]>0).astype(np.uint8)*255):raise ValueError('Binary/logit decision mismatch')
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    npz=out/'predictions.npz';mask=out/'mask.png';cover=out/'coverage.png'
    np.savez_compressed(npz,logits=logits,probabilities=probabilities,coverage=coverage)
    Image.fromarray(binary).save(mask);Image.fromarray(coverage*255).save(cover)
    probability_artifact={'path':str(npz),'sha256':sha256(npz),'format':'native_float32_npz','logits_key':'logits','probabilities_key':'probabilities','coverage_key':'coverage','dtype':'float32','shape_hw':[height,width]}
    metadata={'schema_version':'circuitvision.conductor-observation/1','image_sha256':expected_image_sha256,'mask_sha256':sha256(mask),'format':'binary_u8','width':width,'height':height,
        'model_lock':provenance.get('model_lock'),'threshold':.5,'threshold_source':'pretrained_provider_default','threshold_operator':'>','decision_space':'logits','logit_threshold':0.,'provider_id':PROVIDER_ID,
        'probability_artifact':probability_artifact,'coverage':dict(prediction['coverage_record'],path=str(cover),sha256=sha256(cover)),'placement':prediction['placement'],'provenance':provenance}
    sidecar=Path(str(mask)+'.json');sidecar.write_text(json.dumps(metadata,indent=2)+'\n')
    return {'mask_path':str(mask),'mask_sha256':metadata['mask_sha256'],'mask_sidecar':str(sidecar),'mask_sidecar_sha256':sha256(sidecar),'probability_artifact':probability_artifact,'coverage':metadata['coverage'],'width':width,'height':height,'image_sha256':expected_image_sha256}


class CircuitVisionBackend:
    """Caller holds one-heavy-job lock; guard(stage) must validate it/resources."""
    def __init__(self,lock,*,guard,lock_sha256=None,lock_path=None):
        if not callable(guard):raise ValueError('Runner resource/lock guard is required')
        self.lock=lock;self.guard=guard;self.lock_sha256=lock_sha256;self.lock_path=lock_path;self.paths=validate_lock(lock);self.model=None;self.transforms=None;self.profile={'resource_snapshots':[],'mps_peak_semantics':'maximum of sampled allocation counters, not a true kernel-time peak'}

    @classmethod
    def from_lock(cls,path,*,expected_sha256,guard):
        if sha256(path)!=expected_sha256:raise ValueError('Model lock SHA256 mismatch')
        return cls(json.loads(Path(path).read_text()),guard=guard,lock_sha256=expected_sha256,lock_path=str(Path(path).resolve()))

    def load(self,*,allow_model_load=False):
        if not allow_model_load:raise PermissionError('Model loading needs explicit runner release')
        if self.model is not None:raise ValueError('Model already loaded')
        self.guard('before_load');started=time.monotonic();self._snapshot('before_load');self.paths=validate_lock(self.lock)
        for package,version in self.lock['runtime']['package_versions'].items():
            if importlib.metadata.version(package)!=version:raise ValueError(f'Runtime version mismatch: {package}')
        with _offline():
            import torch
            from hydra import initialize_config_module
            from hydra.core.global_hydra import GlobalHydra
            torch.set_num_threads(1)
            if torch.get_default_dtype()!=torch.float32:raise ValueError('Float32 default dtype required')
            if self.lock['runtime']['device']=='mps':
                if not torch.backends.mps.is_available():raise RuntimeError('Requested MPS device is unavailable; no device fallback')
                if self.lock['runtime'].get('mps_fraction') is not None:torch.mps.set_per_process_memory_fraction(self.lock['runtime']['mps_fraction'])
            sam_spec=importlib.util.find_spec('sam2')
            if sam_spec is None or Path(sam_spec.origin).resolve()!=(Path(self.lock['sam2_root'])/'sam2/__init__.py').resolve():raise ValueError('Installed sam2 does not resolve to pinned source tree')
            sampler_path=Path(self.lock['vendor_root'])/'src/sam2_infer.py'
            spec=importlib.util.spec_from_file_location('_kbot_circuitvision_'+SOURCE_HASHES['src/sam2_infer.py'][:16],sampler_path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            # SAM2 may initialize Hydra's package search path on import. Use the
            # exact checked local YAML, restoring the caller's Hydra state.
            global_hydra=GlobalHydra.instance();previous=global_hydra.hydra;global_hydra.clear()
            try:
                with initialize_config_module(config_module='sam2',version_base='1.2'):
                    model=module.get_modified_sam2(**model_kwargs(self.lock,self.paths))
            finally:
                global_hydra.clear();global_hydra.hydra=previous
            checkpoint=torch.load(self.paths['finetuned_checkpoint'],map_location='cpu',weights_only=True)
            state=checkpoint['state_dict'] if isinstance(checkpoint,dict) and 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state,strict=True)
            model.float();model.eval();model.requires_grad_(False)
            if model.training:raise ValueError('Model did not enter evaluation mode')
            resolution=getattr(getattr(model,'sam2_model',None),'image_size',None)
            if resolution!=RESOLUTION:raise ValueError('Loaded architecture has unexpected resolution')
            transforms=module.SAM2Transforms(resolution=RESOLUTION,mask_threshold=0,max_hole_area=0,max_sprinkle_area=0)
            self.model=model;self.transforms=transforms
        self.profile['load_seconds']=time.monotonic()-started;self._snapshot('after_load')
        self.guard('after_load')
        return self

    def predict(self,image_path,output_dir,*,expected_image_sha256,roi=None):
        if self.model is None:raise RuntimeError('Backend is not loaded; no base-model fallback')
        if not self.lock_sha256 or not self.lock_path:raise ValueError('Prediction requires hash-bound model lock file')
        self.guard('before_inference');started=time.monotonic();self._snapshot('before_inference')
        if sha256(image_path)!=expected_image_sha256:raise ValueError('Source image SHA256 mismatch')
        with Image.open(image_path) as source:image=source.convert('RGB')
        with _offline():
            prediction=native_prediction(image,self.model,self.transforms,device=self.lock['runtime']['device'],roi=roi)
        self.profile['prediction_seconds']=time.monotonic()-started;self._snapshot('after_inference')
        self.guard('after_inference')
        provenance={'upstream_url':UPSTREAM_URL,'upstream_commit':UPSTREAM_COMMIT,'upstream_license':'Apache-2.0','source_hashes':SOURCE_HASHES,'model_lock':{'path':self.lock_path,'sha256':self.lock_sha256},'sam2_commit':SAM2_COMMIT,'artifacts':self.lock['artifacts'],'runtime':self.lock['runtime'],'construction':'rank4 alpha16 dropout0.3;36 exact targets;embedding_r4;refinement kernels3,5,7,11;actual4 channels/branch','model_loaded':True,'training':False,'precision':'float32_no_autocast','runtime_profile':json.loads(json.dumps(self.profile))}
        result=write_prediction(image_path,output_dir,prediction,expected_image_sha256=expected_image_sha256,provenance=provenance)
        result['runtime_profile']=json.loads(json.dumps(self.profile))
        return result

    def _snapshot(self,stage):
        import resource
        import sys
        snapshot={'stage':stage,'process_peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)}
        try:
            import psutil
            snapshot['process_current_rss_bytes']=int(psutil.Process().memory_info().rss)
        except ImportError:pass
        torch=sys.modules.get('torch')
        if torch is not None and self.lock['runtime']['device']=='mps' and torch.backends.mps.is_available():
            for name in ('current_allocated_memory','driver_allocated_memory','recommended_max_memory'):
                fn=getattr(torch.mps,name,None)
                if fn is not None:snapshot['mps_'+name+'_bytes']=int(fn())
        self.profile['resource_snapshots'].append(snapshot)
        for field in ('mps_current_allocated_memory_bytes','mps_driver_allocated_memory_bytes'):
            values=[r[field] for r in self.profile['resource_snapshots'] if field in r]
            if values:self.profile[field+'_observed_max']=max(values)

    def close(self):
        """Release backend references before the caller starts another model."""
        import gc
        import sys
        self.model=None;self.transforms=None;gc.collect()
        torch=sys.modules.get('torch')
        if torch is not None and self.lock['runtime']['device']=='mps' and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self._snapshot('after_close')

    def __enter__(self):return self

    def __exit__(self,exc_type,exc_value,traceback):
        self.close()
        return False
