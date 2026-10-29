"""Synthetic controls; scores from these fixtures are never scientific results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import train as run
from pipeline import A,B,Y,NAMES,audit,digest,normalized,file_hash

class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def test_head_reload_and_symmetry(self):
        rng=np.random.default_rng(44)
        cache={'AA':rng.normal(size=8).astype('float32'),'CC':rng.normal(size=8).astype('float32')}
        frame=pd.DataFrame({A:['AA','CC']*8,B:['CC','AA']*8,Y:[0,1]*8})
        x=run.build_pair_features(frame,cache);y=frame[Y].to_numpy()
        for ops in [run.OPS]+[[x for x in run.OPS if x!=omit] for omit in run.OPS]:
            features=run.build_pair_features(frame,cache,ops)
            swapped=frame.rename(columns={A:B,B:A})
            np.testing.assert_array_equal(features,run.build_pair_features(swapped,cache,ops))
        for head in ['MLP','RF']:
            plan={'head':head}
            model,history=run.fit_model(plan,x,y,x,y,epochs=2,trees=5)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model'
                run.serialize(model,path,x.shape[1])
                np.testing.assert_allclose(run.probability(model,x),run.probability(run.deserialize(plan,path),x),atol=1e-7)
    def test_dominant_component_declines_inference(self):
        frame=pd.DataFrame({A:['AAA']*20,B:['BBB']*20})
        result=run.block_uncertainty(frame,np.linspace(.1,.9,20),np.full(20,.5),np.array([0,1]*10))
        self.assertEqual(result['status'],'INCONCLUSIVE')
    def test_missing_embedding_refuses_drop(self):
        frame=pd.DataFrame({A:['AA'],B:['CC'],Y:[1]})
        with self.assertRaises(KeyError):run.build_pair_features(frame,{'AA':np.ones(8,dtype='float32')})
    def test_paired_component_interval(self):
        frame=pd.DataFrame({A:[f'group{i}' for i in range(12) for _ in range(2)],
                            B:[f'partner{i}' for i in range(24)]})
        labels=np.array([0,1]*12)
        result=run.block_uncertainty(frame,labels.astype(float),np.full(24,.5),labels)
        self.assertEqual(result['components'],12)
        np.testing.assert_allclose(result['paired_macro_ap_difference_95_interval'],[.5,.5])
    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'x').write_text('original')
            run.save(path/'receipt.json',{'artifacts':{'x':file_hash(path/'x')}})
            run.check_receipt(path)
            (path/'x').write_text('changed')
            with self.assertRaises(ValueError):run.check_receipt(path)
    def test_complete_lifecycle(self):
        rng=np.random.default_rng(44);alphabet=np.array(list('ACDEFGHIKLMNPQRSTVWY'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);split=root/'split';split.mkdir();caches=root/'caches';caches.mkdir();out=root/'run'
            allseq=[];hashes={};mapping={}
            for n in NAMES:
                rows=[]
                for i in range(100):
                    a=''.join(rng.choice(alphabet,24));b=''.join(rng.choice(alphabet,24))
                    rows.append({A:a,B:b,Y:i%2})
                frame=audit(pd.DataFrame(rows))[0];frame.to_csv(split/(n+'.csv'),index=False)
                hashes[n]=file_hash(split/(n+'.csv'))
                allseq.extend(frame[A]);allseq.extend(frame[B])
            for s in allseq:mapping[digest(normalized(s,'full'))]=digest(normalized(s,'full'))
            pd.DataFrame({'sequence_sha256':list(mapping),'cluster_id':list(mapping.values())}).to_csv(split/'mmseqs2_clusters.csv',index=False)
            run.save(split/'frozen_splits.json',hashes)
            run.save(split/'source_manifest.json',{n+'.csv':{'sha256':sha} for n,sha in hashes.items()})
            run.save(split/'clustering.json',{'synthetic_fixture_only':True})
            run.save(split/'protocol.json',{'synthetic_fixture_only':True})
            run.save(split/'split_provenance.json',{
                'mapping_sha256':file_hash(split/'mmseqs2_clusters.csv'),
                'clustering_sha256':file_hash(split/'clustering.json'),
                'source_manifest_sha256':file_hash(split/'source_manifest.json'),
                'protocol_sha256':file_hash(split/'protocol.json'),
                'pipeline_sha256':file_hash(Path(run.__file__).with_name('pipeline.py'))})
            from snooppi_utils import ENCODER_SPECS
            for name,filename in run.ENCODERS.items():
                spec=ENCODER_SPECS[name];dim=spec['embedding_dim']
                payload={'embeddings':{s:torch.from_numpy(rng.normal(size=dim).astype('float32')) for s in allseq},
                    'model_id':spec['model_id'],'embedding_dim':dim,'max_residues':1024,'pooling':'mean over non-special residue tokens'}
                if name=='ESM-C 600M':payload.update(normalization='canonical_X_spaces_removed_prefix_v1',esm_version='3.2.1')
                torch.save(payload,caches/filename)
            args=['run','fit','--splits',str(split),'--caches',str(caches),'--output',str(out)]
            original=run.fit_model
            def bounded(*a,**kw):return original(*a,**kw,epochs=2,trees=5)
            features=run.build_pair_features
            testids=set(pd.read_csv(split/'test.csv').full_pair_id)
            def no_test_features(frame,*a,**kw):
                self.assertFalse(set(frame.full_pair_id)&testids)
                return features(frame,*a,**kw)
            with patch('sys.argv',args),patch.object(run,'fit_model',side_effect=bounded),patch.object(run,'build_pair_features',side_effect=no_test_features):run.main()
            self.assertEqual(len(list(out.glob('*/receipt.json'))),9)
            with patch('sys.argv',args),patch.object(run,'fit_model',side_effect=AssertionError('Duplicate training')):run.main()
            args[1]='evaluate'
            with patch('sys.argv',args),patch.object(run,'block_uncertainty',return_value={'status':'synthetic_smoke_only'}):run.main()
            self.assertEqual(len(pd.read_csv(out/'final_metrics.csv')),10)
            with patch('sys.argv',args),patch.object(run,'probability',side_effect=AssertionError('Duplicate inference')),patch.object(run,'block_uncertainty',return_value={'status':'synthetic_smoke_only'}):run.main()
            args[1]='fit'
            with patch('sys.argv',args),self.assertRaisesRegex(ValueError,'Final test consumed'):run.main()
            (split/'test.csv').write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'tampering'):run.load_splits(split)

if __name__=='__main__':unittest.main()
