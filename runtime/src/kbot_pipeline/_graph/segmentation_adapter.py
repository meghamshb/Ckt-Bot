"""Load an externally inferred conductor mask without changing image coordinates.

No inference or truth geometry is accepted here. The required adjacent .json
manifest binds image and mask bytes and records validation selection or the explicit
pretrained-provider decision. Neither policy establishes circuit correctness.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def load_pretrained_artifact(metadata, binary_mask, *, allow_partial=False):
    """Validate typed native predictions without treating sigmoid rounding as a decision."""
    if (metadata.get('provider_id')!='CircuitVision/SAM2.1-HieraL-LoRA' or
        metadata.get('format')!='binary_u8' or metadata.get('threshold')!=.5 or
        metadata.get('threshold_operator')!='>' or metadata.get('decision_space')!='logits' or
        metadata.get('logit_threshold')!=0):
        raise ValueError('Unsupported pretrained provider decision contract')
    lock=metadata.get('model_lock',{});artifact=metadata.get('probability_artifact',{})
    for record in (lock,artifact):
        if not record.get('path') or hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record.get('sha256'):
            raise ValueError('Pretrained model lock or native prediction artifact changed')
    if artifact.get('format')!='native_float32_npz':raise ValueError('Native float prediction artifact required')
    with np.load(artifact['path'],allow_pickle=False) as values:
        logits=values[artifact.get('logits_key','logits')].copy()
        probabilities=values[artifact.get('probabilities_key','probabilities')].copy()
        coverage=values[artifact.get('coverage_key','coverage')].copy()
    if any(a.shape!=binary_mask.shape for a in (logits,probabilities,coverage)):
        raise ValueError('Pretrained prediction frame differs from native mask')
    if logits.dtype!=np.float32 or probabilities.dtype!=np.float32 or not np.isin(coverage,[0,1]).all():
        raise ValueError('Pretrained logits/probabilities must retain float32 values and binary coverage')
    observed=coverage.astype(bool)
    if not allow_partial and not observed.all():raise ValueError('Full-circuit assembly requires complete wire-perception coverage')
    if (not np.isfinite(logits[observed]).all() or not np.isfinite(probabilities[observed]).all() or
        (probabilities[observed]<0).any() or (probabilities[observed]>1).any()):
        raise ValueError('Invalid observed pretrained probabilities/logits')
    if (not np.isnan(logits[~observed]).all() or not np.isnan(probabilities[~observed]).all()):
        raise ValueError('Unobserved pixels must not acquire invented confidence')
    native=logits[observed];expected_prob=np.empty_like(native);positive=native>=0
    expected_prob[positive]=1/(1+np.exp(-native[positive]))
    exp=np.exp(native[~positive]);expected_prob[~positive]=exp/(1+exp)
    if not np.allclose(probabilities[observed],expected_prob,rtol=1e-6,atol=2e-7):
        raise ValueError('Pretrained probabilities differ from native sigmoid logits')
    auxiliary=metadata.get('coverage',{})
    if auxiliary.get('path'):
        path=Path(auxiliary['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=auxiliary.get('sha256'):
            raise ValueError('Auxiliary coverage artifact changed')
        with Image.open(path) as image:
            if image.mode!='L' or not np.array_equal(np.array(image),coverage*255):
                raise ValueError('Auxiliary coverage differs from native observations')
    placement=metadata.get('placement',{})
    if placement and placement.get('native_size_wh')!=[binary_mask.shape[1],binary_mask.shape[0]]:
        raise ValueError('Recorded transform differs from native frame')
    expected=np.zeros_like(binary_mask);expected[observed]=(logits[observed]>0).astype(np.uint8)
    if not np.array_equal(binary_mask,expected):raise ValueError('Pretrained mask differs from authoritative native logits')
    return dict(logits=logits,probabilities=probabilities,coverage=coverage)


def load_conductor_mask(mask_path, image_path, image_size, *, source_policy="synthetic_validation", with_probability=False):
    """Return uint8 binary mask and checked provenance; never resize or threshold RGB."""
    path=Path(mask_path)
    manifest_path=Path(str(path)+'.json')
    metadata=json.loads(manifest_path.read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    image_digest=hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    if metadata.get('mask_sha256')!=digest or metadata.get('image_sha256')!=image_digest:
        raise ValueError('Conductor mask/image SHA256 identity mismatch')
    threshold=metadata.get('threshold')
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not 0<threshold<1:
        raise ValueError('Conductor mask requires fixed probability threshold in (0,1)')
    allowed={'synthetic_validation','development_validation','training_validation','manual_annotation_control','pretrained_provider_default'}
    if source_policy not in allowed:raise ValueError('Unknown conductor mask source policy')
    expected='manual_annotation' if source_policy=='manual_annotation_control' else source_policy
    if metadata.get('threshold_source')!=expected:raise ValueError('Conductor threshold source differs from explicit policy')
    if source_policy=='manual_annotation_control':
        if metadata.get('format')!='binary_u8' or not metadata.get('annotation_id'):
            raise ValueError('Manual control requires binary mask and annotation identity')
    elif source_policy!='pretrained_provider_default' and not metadata.get('validation_id'):raise ValueError('Conductor threshold must identify frozen validation selection')
    with Image.open(path) as image:
        if image.mode!='L' or image.size!=tuple(image_size):
            raise ValueError('Conductor mask must be native-size single-channel uint8; resizing forbidden')
        raster=np.array(image,dtype=np.uint8)
    encoding=metadata.get('format')
    if encoding=='binary_u8':
        if not np.isin(raster,[0,255]).all():raise ValueError('Binary conductor mask contains nonbinary pixels')
        mask=(raster==255).astype(np.uint8)
    elif encoding=='probability_u8':mask=(raster.astype(np.float32)/255.>=threshold).astype(np.uint8)
    else:raise ValueError('Unsupported conductor mask format')
    if source_policy=='pretrained_provider_default':
        load_pretrained_artifact(metadata,mask)
        if with_probability:
            raise ValueError('Existing probability-aware pin policies assume probability-threshold decisions; use frozen assembly or a future logit-aware policy for this pretrained backend')
    evidence={'mask_sha256':digest,'image_sha256':image_digest,'format':encoding,
                 'threshold':threshold,'threshold_source':expected,'source_policy':source_policy,
                 'validation_id':metadata.get('validation_id'),'annotation_id':metadata.get('annotation_id'),'mask_path':str(path.resolve()),
                 'manifest_path':str(manifest_path.resolve()),
                 'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 'coordinate_frame':'original_image_pixels','verified':False}
    if source_policy=='pretrained_provider_default':
        evidence.update(provider_id=metadata['provider_id'],model_lock=metadata['model_lock'],
            probability_artifact=metadata['probability_artifact'],decision_space='logits',threshold_operator='>',
            logit_threshold=0.,locally_validated_threshold=False,transforms=metadata.get('placement'),coverage=metadata.get('coverage'))
    if with_probability:
        if encoding!='probability_u8':raise ValueError('Probability-aware topology policy requires original probability_u8 observation')
        return mask,evidence,raster.astype(np.float32)/255.
    return mask,evidence
