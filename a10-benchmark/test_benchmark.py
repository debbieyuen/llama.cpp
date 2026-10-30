import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import httpx
import benchmark as b

class MetricsTests(unittest.TestCase):
    def test_percentile_and_missing(self):
        self.assertEqual(b.percentile([10,20,None],.5),15)
        self.assertIsNone(b.percentile([], .99))

    def test_failed_requests_not_in_tokens_or_latency(self):
        rows=[dict(status='ok',output_tokens=10,ttft_ms=20,correct=True),
              dict(status='error',output_tokens=99,ttft_ms=999,correct=None)]
        result=b.summarize(rows,2)
        self.assertEqual(result['output_tok_s'],5)
        self.assertEqual(result['ttft_ms_p50'],20)
        self.assertEqual(result['failed'],1)

class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, events):
        async def handler(request):
            return httpx.Response(200,text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
        args=SimpleNamespace(timeout=5,cache_prompt=False)
        with TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(handler)) as client:
                return await b.request(client,[1,2],16,'42',args,'test',Path(tmp))

    async def test_count_from_usage_not_chunks(self):
        row=await self.invoke([{'content':'42','tokens':[10,11]},
            {'stop':True,'timings':{'predicted_n':2,'prompt_n':2,'predicted_per_second':8}}])
        self.assertEqual(row['status'],'ok')
        self.assertEqual(row['output_tokens'],2)
        self.assertTrue(row['correct'])

    async def test_missing_final_is_failure(self):
        row=await self.invoke([{'content':'42'}])
        self.assertEqual(row['status'],'error')

    async def test_truncated_context_is_failure(self):
        row=await self.invoke([{'content':'42','tokens':[10]},
                              {'stop':True,'truncated':True,'tokens_predicted':1}])
        self.assertEqual(row['status'],'error')
        self.assertIsNone(row['tpot_ms'])

    async def test_wrong_answer_separate_from_transport(self):
        row=await self.invoke([{'content':'wrong','tokens':[10]},
                              {'stop':True,'tokens_predicted':1}])
        self.assertEqual(row['status'],'ok')
        self.assertFalse(row['correct'])

    async def test_exact_prompt_size(self):
        async def handler(request):
            content=json.loads(request.content)['content']
            return httpx.Response(200,json={'tokens':list(content.encode())})
        async with httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(handler)) as client:
            for pos in (.1,.5,.9):
                prompt,expected=await b.make_case(client,512,'retrieval',pos)
                self.assertEqual(len(prompt),512)
                self.assertIn(expected,bytes(prompt).decode())

if __name__=='__main__':
    unittest.main()

class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_matrix_writes_summary(self):
        from unittest.mock import patch
        real_client = httpx.AsyncClient
        async def handler(request):
            if request.url.path == '/tokenize':
                return httpx.Response(200,json={'tokens':list(json.loads(request.content)['content'].encode())})
            if request.url.path == '/completion':
                payload=json.loads(request.content)
                events=[{'content':'42','tokens':[1,2]}, {'stop':True,'timings':{'predicted_n':2,'prompt_n':len(payload['prompt'])}}]
                return httpx.Response(200,text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
            return httpx.Response(200,json={})
        with TemporaryDirectory() as tmp:
            args=SimpleNamespace(results=tmp,api_key_env='BENCH_TEST_UNUSED_KEY',url='http://test',
                timeout=5,concurrency=[1,2],suite='performance',input=[128],output=[8],positions=[.5],
                context_per_slot=256,warmups=1,requests=3,label='mock transport test',cache_prompt=False,duration=1)
            def factory(**kwargs):
                return real_client(**kwargs,transport=httpx.MockTransport(handler))
            with patch.object(b.httpx,'AsyncClient',side_effect=factory):
                self.assertEqual(await b.run(args),0)
            summary=json.loads(next(Path(tmp).glob('*/summary.json')).read_text())
            self.assertEqual(len(summary),2)
            self.assertTrue(all(r['successful']==3 for r in summary))
            self.assertEqual(len(list(Path(tmp).glob('*/case-*/warmup-*.json'))),2)
            self.assertTrue(next(Path(tmp).glob('*/summary.csv')).is_file())
