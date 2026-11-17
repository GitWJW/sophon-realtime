import asyncio
import argparse
import contextlib
import base64
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import aiohttp
from aiohttp import web
import numpy as np
import soundfile as sf

spec=importlib.util.spec_from_file_location('bench',Path(__file__).parents[1]/'benchmark_http.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


def audio_event():
    out=io.BytesIO();sf.write(out,np.ones(1600,dtype=np.float32)*.1,16000,format='WAV')
    return json.dumps({'modality':'audio','choices':[{'delta':{'content':base64.b64encode(out.getvalue()).decode()}}]})


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode='normal'
        self.active=0
        self.peak=0
        async def handler(req):
            if self.mode=='http': return web.Response(status=503)
            if self.mode=='timeout':
                await asyncio.sleep(.1);return web.Response()
            resp=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await resp.prepare(req)
            if self.mode=='error':payload='data: {"error":"test"}\n\n'
            elif self.mode=='bad':payload='data: {"modality":"audio","choices":[{"delta":{"content":"!"}}]}\n\n'
            elif self.mode=='text':payload='data: {"choices":[]}\n\ndata: [DONE]\n\n'
            else:
                event=audio_event()
                # Valid multiline JSON data plus fragmented transport writes.
                payload=': comment\n'+'data: '+event[:1]+'\n'+'data: '+event[1:]+'\n\n'
                if self.mode!='truncated':payload+='data: [DONE]\n\n'
            raw=payload.encode()
            for i in range(0,len(raw),17):await resp.write(raw[i:i+17])
            await resp.write_eof();return resp
        @web.middleware
        async def count_active(req, handler):
            if req.path != '/v1/chat/completions':
                return await handler(req)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(.02)
                return await handler(req)
            finally:
                self.active -= 1
        async def models(req):
            return web.json_response({'data': [{'id': 'mock-model'}]})
        app=web.Application(middlewares=[count_active])
        app.router.add_post('/v1/chat/completions',handler)
        app.router.add_get('/v1/models',models)
        self.runner=web.AppRunner(app);await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.url='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):await self.runner.cleanup()

    async def test_stream_cases(self):
        async with aiohttp.ClientSession() as session:
            for mode in ('normal','http','error','bad','text','truncated'):
                with self.subTest(mode=mode):
                    self.mode=mode
                    r=await b.request(session,self.url,{},mode)
                    self.assertEqual(r['success'],mode=='normal')
                    if mode=='normal':
                        self.assertEqual(r['audio_chunk_count'],1)
                        self.assertAlmostEqual(r['audio_duration_s'],.1)
                        self.assertGreaterEqual(r['first_audio_decoded_s'],r['first_audio_received_s'])
                    if mode=='truncated':
                        self.assertIsNotNone(r['first_audio_received_s'])

    async def test_timeout(self):
        self.mode='timeout'
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=.01)) as session:
            r=await b.request(session,self.url,{},'timeout')
            self.assertFalse(r['success'])
            self.assertIn('Timeout',r['error_type'])

    async def exercise_load(self, arrival=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sf.write(root/'x.wav',np.ones(1600)*.1,16000)
            items=[dict(id=str(i),wav='x.wav',language='zh') for i in range(3)]
            (root/'inputs.json').write_text(json.dumps(items))
            (root/'prompt.json').write_text(json.dumps({'zh':'test'}))
            a=argparse.Namespace(system_prompt=str(root/'prompt.json'),wav=None,
                manifest=str(root/'inputs.json'),language='zh',out=str(root/'out'),
                timeout=3,url=self.url,model=None,seed=0,voice='chelsie',
                max_tokens=10,warmup=0,requests=6,selection='round-robin',
                concurrency=2 if arrival is None else None,arrival_window=arrival,
                initial_buffer=0,audible_threshold=.02)
            with contextlib.redirect_stdout(io.StringIO()):
                await b.run(a)
            result=json.loads((root/'out/summary.json').read_text())
            rows=[json.loads(x) for x in (root/'out/requests.jsonl').read_text().splitlines()]
            self.assertEqual(result['success'],6)
            self.assertEqual(sorted(r['input_id'] for r in rows),['0','0','1','1','2','2'])
            self.assertEqual(result['total'],len(rows))
            if arrival is None:
                self.assertLessEqual(self.peak,2)
                self.assertLessEqual(result['peak_inflight'],2)
                self.assertTrue(all(r['send_lag_s'] is None for r in rows))
            else:
                for r in rows:
                    expected=arrival*int(r['request_id'])/5
                    self.assertAlmostEqual(r['scheduled_start_s'],expected,places=5)
                    self.assertGreaterEqual(r['actual_start_s']+1e-6,r['scheduled_start_s'])
                    self.assertGreaterEqual(r['send_lag_s'],0)

    async def test_bounded_load_and_input_selection(self):
        await self.exercise_load()

    async def test_staggered_schedule(self):
        await self.exercise_load(.1)

    def test_all_failed_summary(self):
        r=dict(success=False,error_type='Test',audio_duration_s=0)
        s=b.summary([r],1,1,{},'model',[])
        self.assertEqual(s['success_rate'],0)
        self.assertIsNone(s['first_audio_received_s'])
        self.assertEqual(s['audio_seconds_per_s'],0)

    def test_stall(self):
        chunks=[dict(arrival_s=.5,audio_s=.2),dict(arrival_s=1.,audio_s=.4)]
        self.assertAlmostEqual(b.playback_stall(chunks),.3)
        self.assertAlmostEqual(b.playback_stall(chunks,.3),0.)

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sf.write(root/'x.wav',np.zeros(1600),16000)
            items=[dict(id='a',wav='x.wav',language='zh'),dict(id='b',wav='x.wav',language='en')]
            p=root/'inputs.json';p.write_text(json.dumps(items))
            rows=b.load_inputs(None,str(p),'zh',{'zh':'a','en':'b'})
            self.assertEqual([r['id'] for r in rows],['a','b'])
            self.assertAlmostEqual(rows[0]['duration_s'],.1)
            with self.assertRaises(ValueError):b.load_inputs(None,str(p),'zh',{'zh':'a'})

    def test_summary_excludes_failure(self):
        good=dict(success=True,audio_duration_s=2.,first_audio_received_s=.5,first_audio_decoded_s=.6,first_audible_s=.5,stall_s=0.,e2e_s=1.)
        bad=dict(good,success=False,error_type='Test',audio_duration_s=99.,first_audio_received_s=100.)
        s=b.summary([good,bad],2.,1,{},'model',[])
        self.assertEqual(s['success_rate'],.5)
        self.assertEqual(s['audio_seconds_per_s'],1.)
        self.assertEqual(s['first_audio_received_s']['p50'],.5)


if __name__=='__main__':unittest.main()
