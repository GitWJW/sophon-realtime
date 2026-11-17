import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file

spec=importlib.util.spec_from_file_location('converter',Path(__file__).parents[1]/'convert_fp8_to_bf16.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def fixture(root):
    src=root/'source';(src/'thinker').mkdir(parents=True);(src/'talker').mkdir()
    (src/'config.json').write_text('{}');(src/'talker/config.json').write_text('{}')
    config={'quantization_config':{'quant_method':'compressed-tensors','format':'float-quantized',
        'config_groups':{'g':{'weights':{'num_bits':8,'type':'float','strategy':'channel','symmetric':True,'dynamic':False}}}}}
    (src/'thinker/config.json').write_text(json.dumps(config))
    w=torch.tensor([[1.,2.],[-1.,4.]]).to(torch.float8_e4m3fn)
    scale=torch.tensor([[.5],[2.]],dtype=torch.bfloat16)
    save_file({'layer.weight':w,'bias':torch.ones(2,dtype=torch.bfloat16)},str(src/'thinker/a.safetensors'))
    save_file({'layer.weight_scale':scale},str(src/'thinker/b.safetensors'))
    return src,w,scale


class ConversionTests(unittest.TestCase):
    def test_cross_shard_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src,w,scale=fixture(root);dst=root/'out'
            c.convert(src,dst,True)
            self.assertFalse(dst.exists())
            c.convert(src,dst)
            got=load_file(str(dst/'thinker/a.safetensors'))
            self.assertTrue(torch.equal(got['layer.weight'],(w.float()*scale.float()).bfloat16()))
            self.assertTrue((dst/'BF16_CONVERSION.json').exists())
            self.assertNotIn('quantization_config',json.loads((dst/'thinker/config.json').read_text()))
            index=json.loads((dst/'thinker/model.safetensors.index.json').read_text())
            self.assertEqual(set(index['weight_map']),{'layer.weight','bias'})
            self.assertEqual(load_file(str(src/'thinker/a.safetensors'))['layer.weight'].dtype,torch.float8_e4m3fn)
            with self.assertRaises(FileExistsError):c.convert(src,dst)

    def test_reject_unsupported_scheme(self):
        with tempfile.TemporaryDirectory() as d:
            src,_,_=fixture(Path(d));p=src/'thinker/config.json'
            config=json.loads(p.read_text());config['quantization_config']['config_groups']['g']['weights']['strategy']='group'
            p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):c.convert(src,Path(d)/'out',True)

    def test_invalid_scale_and_nested_paths(self):
        with tempfile.TemporaryDirectory() as d:
            src,_,_=fixture(Path(d));dst=Path(d)/'out'
            save_file({'layer.weight_scale':torch.tensor([[0.],[1.]],dtype=torch.bfloat16)},str(src/'thinker/b.safetensors'))
            with self.assertRaises(ValueError):c.convert(src,dst)
            self.assertFalse(dst.exists())
            with self.assertRaises(ValueError):c.convert(src,src/'nested',True)

    def test_nonfinite_output(self):
        w=torch.tensor([[float('nan')]]).to(torch.float8_e4m3fn)
        with self.assertRaises(ValueError):c.dequantize(w,torch.ones(1,1))


if __name__=='__main__':unittest.main()
