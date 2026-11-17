"""Reconstruct a restricted Flash compressed-tensors FP8 thinker as BF16 on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def validate_config(config):
    q = config.get('quantization_config', {})
    if q.get('quant_method') != 'compressed-tensors' or q.get('format') != 'float-quantized':
        raise ValueError('only compressed-tensors float-quantized checkpoints are supported')
    groups = q.get('config_groups')
    if not isinstance(groups, dict) or not groups:
        raise ValueError('missing config_groups')
    for group in groups.values():
        w = group.get('weights', {})
        if not (w.get('num_bits') == 8 and w.get('type') == 'float'
                and w.get('strategy') == 'channel' and w.get('symmetric') is True
                and w.get('dynamic') is False):
            raise ValueError('requires static symmetric per-channel FP8 weights')
        if w.get('block_structure') or w.get('group_size'):
            raise ValueError('block/group quantization is unsupported')
        act = group.get('input_activations')
        if act is not None and not (act.get('num_bits') == 8 and act.get('type') == 'float'
                and act.get('strategy') == 'token' and act.get('dynamic') is True
                and act.get('symmetric') is True):
            raise ValueError('unsupported activation quantization')
        if group.get('output_activations') is not None:
            raise ValueError('output activation quantization is unsupported')
    return q


def inventory(source):
    thinker = source / 'thinker'
    config_path = thinker / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    entries = {}
    for shard in sorted(thinker.glob('*.safetensors')):
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in entries:
                    raise ValueError(f'duplicate tensor: {key}')
                t = f.get_slice(key)
                entries[key] = dict(shard=shard, dtype=t.get_dtype(), shape=t.get_shape())
    if not entries:
        raise ValueError('no thinker safetensors')
    index = thinker / 'model.safetensors.index.json'
    if index.exists():
        expected = json.loads(index.read_text())['weight_map']
        actual = {k: v['shard'].name for k, v in entries.items()}
        if expected != actual:
            raise ValueError('source index does not match thinker shards')
    pairs = {}
    output_bytes = 0
    for key, value in entries.items():
        if value['dtype'] == 'F8_E4M3':
            if not key.endswith('.weight') or len(value['shape']) != 2:
                raise ValueError(f'unsupported FP8 tensor: {key}')
            scale_key = key[:-len('.weight')] + '.weight_scale'
            scale = entries.get(scale_key)
            if not scale or scale['shape'] != [value['shape'][0], 1]:
                raise ValueError(f'missing/invalid channel scale: {key}')
            if scale['dtype'] not in ('BF16', 'F32'):
                raise ValueError(f'unsupported scale dtype: {scale_key}')
            pairs[key] = scale_key
        elif value['dtype'] != 'BF16' and not key.endswith('.weight_scale'):
            raise ValueError(f'unsupported unquantized tensor dtype: {key}')
    if not pairs:
        raise ValueError('no supported FP8 weights')
    scales = set(pairs.values())
    for key, value in entries.items():
        if key.endswith('.weight_scale') and key not in scales:
            raise ValueError(f'unmatched scale tensor: {key}')
        if key not in scales:
            output_bytes += 2 * __import__('math').prod(value['shape'])
    return config, config_path, entries, pairs, output_bytes


def dequantize(weight, scale, rows=256):
    if not torch.isfinite(scale.float()).all() or not (scale > 0).all():
        raise ValueError('scales must be finite and positive')
    result = torch.empty(weight.shape, dtype=torch.bfloat16, device='cpu')
    for start in range(0, weight.shape[0], rows):
        values = weight[start:start+rows].float() * scale[start:start+rows].float()
        result[start:start+rows] = values.to(torch.bfloat16)
        if not torch.isfinite(result[start:start+rows].float()).all():
            raise ValueError('non-finite reconstructed weight')
    return result


def convert(src, dst, check_only=False):
    source = Path(src).resolve()
    destination = Path(dst).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('source and destination must be separate, non-nested paths')
    if destination.exists():
        raise FileExistsError(destination)
    if not (source/'config.json').is_file() or not (source/'talker').is_dir():
        raise ValueError('expected complete Flash model root with config.json, thinker/, talker/')
    config, config_path, entries, pairs, output_bytes = inventory(source)
    scales = set(pairs.values())
    report = dict(source=str(source), fp8_weights=len(pairs),
                  output_tensors=len(entries)-len(scales), thinker_tensor_bytes=output_bytes,
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  method='BF16(float32(FP8 weight) * float32(per-output-channel scale))',
                  limitation='Approximate reconstruction, not original pre-quantization BF16 weights.')
    # Validate every scale before creating any output.
    for scale_key in scales:
        with safe_open(str(entries[scale_key]['shard']), framework='pt', device='cpu') as f:
            scale=f.get_tensor(scale_key)
            if not torch.isfinite(scale.float()).all() or not (scale > 0).all():
                raise ValueError(f'invalid scale values: {scale_key}')
    print(json.dumps(report, indent=2), flush=True)
    if check_only:
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied_bytes=sum(p.stat().st_size for p in source.rglob('*') if p.is_file() and not
                     (p.parent == source/'thinker' and p.suffix == '.safetensors'))
    required=output_bytes+copied_bytes+1024**3
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError(f'insufficient disk space; estimated minimum {required} bytes')
    stage=Path(tempfile.mkdtemp(prefix=destination.name+'.incomplete-',dir=destination.parent))
    # On failure preserve the clearly named incomplete directory for diagnosis.
    try:
        def ignore(directory, names):
            if Path(directory) == source/'thinker':
                return [n for n in names if n.endswith('.safetensors') or n.endswith('.safetensors.index.json')]
            return []
        shutil.copytree(source,stage,dirs_exist_ok=True,ignore=ignore)
        mapping={}
        for shard in sorted({v['shard'] for v in entries.values()}):
            output={}
            with safe_open(str(shard),framework='pt',device='cpu') as f:
                for key in f.keys():
                    if key in scales:
                        continue
                    weight=f.get_tensor(key)
                    if key in pairs:
                        scale_key=pairs[key]
                        with safe_open(str(entries[scale_key]['shard']),framework='pt',device='cpu') as other:
                            weight=dequantize(weight,other.get_tensor(scale_key))
                    output[key]=weight.contiguous()
                    mapping[key]=shard.name
            if output:
                save_file(output,str(stage/'thinker'/shard.name),metadata={'format':'pt'})
            del output
            print('converted',shard.name,flush=True)
        config.pop('quantization_config')
        config['dtype']='bfloat16'
        if 'torch_dtype' in config:
            config['torch_dtype']='bfloat16'
        (stage/'thinker/config.json').write_text(json.dumps(config,indent=2)+'\n')
        (stage/'thinker/model.safetensors.index.json').write_text(json.dumps(
            dict(metadata=dict(total_size=output_bytes),weight_map=mapping),indent=2)+'\n')
        report['status']='complete'
        (stage/'BF16_CONVERSION.json').write_text(json.dumps(report,indent=2)+'\n')
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(stage,destination)
        print('complete:',destination,flush=True)
        return report
    except BaseException:
        print('Conversion failed; incomplete output retained at:',stage,flush=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src',required=True)
    p.add_argument('--dst',required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    convert(args.src,args.dst,args.check_only)
