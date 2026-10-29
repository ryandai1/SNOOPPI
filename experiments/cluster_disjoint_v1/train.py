"""Bounded, resumable frozen-cache comparisons; fit and final evaluation are separate."""
import argparse
import copy
import json
import random
import signal
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from pipeline import A, B, Y, NAMES, Groups, audit, digest, file_hash, normalized, validate
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'drafting_code'))
from snooppi_utils import (load_embedding_cache, validate_cache_coverage,
                          build_pair_features, select_threshold, summarize_predictions)

ENCODERS = {
    'ProtT5': 'prot_t5_embedding_cache.pt',
    'ESM-2 650M': 'esm2_t33_650M_snooppi_mean_pool_embeddings.pt',
    'ESM-C 600M': 'esmc_600m_snooppi_mean_pool_embeddings.pt',
}
OPS = ['sum', 'absdiff', 'product']


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def event(root, kind, **kwargs):
    with (root/'events.jsonl').open('a') as f:
        f.write(json.dumps({'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                            'event': kind, **kwargs})+'\n')


def plans():
    result = []
    for encoder in ENCODERS:
        for head in ['RF', 'MLP']:
            result.append({'id': encoder.replace(' ','_')+'_'+head, 'encoder':encoder, 'head':head, 'operations':OPS})
    for omit in OPS:
        result.append({'id':'ProtT5_MLP_without_'+omit,'encoder':'ProtT5','head':'MLP','operations':[x for x in OPS if x != omit]})
    return result


def load_splits(root):
    provenance=json.loads((root/'split_provenance.json').read_text())
    for name,key in [('mmseqs2_clusters.csv','mapping_sha256'),('clustering.json','clustering_sha256'),
                     ('source_manifest.json','source_manifest_sha256'),('protocol.json','protocol_sha256')]:
        if file_hash(root/name)!=provenance[key]: raise ValueError('Preparation artifact tampering: '+name)
    sources=json.loads((root/'source_manifest.json').read_text())
    if set(sources)!={n+'.csv' for n in NAMES}: raise ValueError('Complete source provenance required')
    if provenance['pipeline_sha256']!=file_hash(Path(__file__).with_name('pipeline.py')):
        raise ValueError('Preparation code changed; revalidate under a versioned protocol')
    expected = json.loads((root/'frozen_splits.json').read_text())
    if set(expected) != set(NAMES):
        raise ValueError('Incomplete frozen split manifest')
    parts = {}
    for n in NAMES:
        if file_hash(root/(n+'.csv')) != expected[n]:
            raise ValueError('Split tampering: '+n)
        parts[n] = pd.read_csv(root/(n+'.csv')).assign(split=n)
    combined = pd.concat(parts.values(),ignore_index=True)
    clean, conflicts, collisions, _ = audit(combined)
    if len(clean) != len(combined) or len(conflicts) or len(collisions):
        raise ValueError('Frozen split contains duplicates, conflicts or input collisions')
    for mode in ['full','t5','esm']:
        if not combined[mode+'_pair_id'].reset_index(drop=True).equals(clean[mode+'_pair_id'].reset_index(drop=True)):
            raise ValueError('Stored pair IDs disagree with sequences')
    mapping = pd.read_csv(root/'mmseqs2_clusters.csv')
    if mapping.sequence_sha256.duplicated().any():
        raise ValueError('Ambiguous mapping')
    clusters = dict(zip(mapping.sequence_sha256,mapping.cluster_id))
    validate(combined, clusters)
    return parts, expected, clusters


class Head(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(width,512),torch.nn.ReLU(),torch.nn.Dropout(.1),
            torch.nn.Linear(512,512),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(512,256),
            torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(256,1))
    def forward(self, x):
        return self.network(x).squeeze(-1)


def probability(model, x):
    if isinstance(model,torch.nn.Module):
        model.eval()
        with torch.inference_mode():
            p = np.concatenate([torch.sigmoid(model(torch.from_numpy(x[i:i+1024]))).numpy() for i in range(0,len(x),1024)])
    else:
        p = model.predict_proba(x)[:,list(model.classes_).index(1)]
    if not np.isfinite(p).all(): raise ValueError('Non-finite predictions')
    return p


def fit_model(plan, x, y, xv, yv, epochs=50, trees=400):
    random.seed(44); np.random.seed(44); torch.manual_seed(44)
    weights={0:1.,1:float((y==0).sum()/(y==1).sum())}
    if plan['head']=='RF':
        model=RandomForestClassifier(n_estimators=trees,criterion='gini',max_features='sqrt',
            min_samples_leaf=2,bootstrap=True,class_weight=weights,random_state=44,n_jobs=3)
        model.fit(x,y)
        return model,[]
    model=Head(x.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weights[1]]))
    history=[]; best=-float('inf'); stale=0; best_state=None
    tx=torch.from_numpy(x); ty=torch.tensor(y,dtype=torch.float32)
    for epoch in range(1,epochs+1):
        model.train(); loss_total=0.
        for idx in torch.randperm(len(y)).split(1024):
            opt.zero_grad(); loss=loss_fn(model(tx[idx]),ty[idx])
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward();opt.step();loss_total+=loss.item()*len(idx)
        p=probability(model,xv); score=summarize_predictions(yv,p,.5)['macro_ap']
        history.append({'epoch':epoch,'train_weighted_bce':loss_total/len(y),'validation_macro_ap':score})
        if score>best:
            best=score;best_state=copy.deepcopy(model.state_dict());stale=0
        else: stale+=1
        if stale>=10: break
    model.load_state_dict(best_state)
    return model,history


def serialize(model, path, width):
    if isinstance(model,torch.nn.Module):
        torch.save({'width':width,'state_dict':model.state_dict()},path)
    else: joblib.dump(model,path)


def deserialize(plan,path):
    if plan['head']=='MLP':
        obj=torch.load(path,map_location='cpu',weights_only=True)
        model=Head(obj['width']);model.load_state_dict(obj['state_dict']);return model
    return joblib.load(path)


def check_receipt(directory):
    record=json.loads((directory/'receipt.json').read_text())
    for name, sha in record['artifacts'].items():
        if file_hash(directory/name)!=sha: raise ValueError('Run artifact changed: '+name)
    return record


def check_caches(directory,parts,allow_legacy):
    info={}
    sequences={s for d in parts.values() for c in [A,B] for s in d[c]}
    if any(normalized(s,'t5')!=normalized(s,'esm') for s in sequences):
        raise ValueError('Encoder inputs differ; no silent change to comparison population')
    for name,filename in ENCODERS.items():
        path=directory/filename
        cache,metadata=load_embedding_cache(path,name)
        coverage=validate_cache_coverage(cache,parts)
        legacy=metadata.get('format')=='legacy_direct_dictionary'
        if legacy and not allow_legacy:
            raise ValueError('Legacy cache provenance unverified; document evidence before --allow-legacy-provenance')
        info[name]={'sha256':file_hash(path),'metadata':metadata,'coverage':coverage,
                    'legacy_provenance_unverified':legacy}
        del cache
    return info


def block_uncertainty(frame, scores, reference, labels):
    """Paired interaction-component bootstrap; decline inference with too few blocks."""
    g=Groups()
    for a,b in zip(frame[A],frame[B]): g.union(digest(normalized(a,'full')),digest(normalized(b,'full')))
    # Caller supplies cluster IDs as A/B, so pairs sharing homolog groups stay linked.
    blocks=pd.Series([g.find(digest(normalized(a,'full'))) for a in frame[A]])
    indices=[np.flatnonzero(blocks.to_numpy()==b) for b in sorted(set(blocks))]
    sizes=[len(x) for x in indices]
    details={'components':len(indices),'largest_component_fraction':max(sizes)/len(frame)}
    if len(indices)<10 or max(sizes)>.5*len(frame):
        return {**details,'status':'INCONCLUSIVE','reason':'Too few independent components or one dominant component'}
    rng=np.random.default_rng(44);delta=[]
    for _ in range(500):
        idx=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        if len(np.unique(labels[idx]))<2: continue
        delta.append(summarize_predictions(labels[idx],scores[idx],.5)['macro_ap']-
                     summarize_predictions(labels[idx],reference[idx],.5)['macro_ap'])
    if len(delta)<450: return {**details,'status':'INCONCLUSIVE','reason':'Too many single-class bootstrap samples'}
    return {**details,'status':'EXPLORATORY','paired_macro_ap_difference_95_interval':np.quantile(delta,[.025,.975]).tolist(),
            'replicates':len(delta),'reference':'ProtT5_MLP'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['fit','evaluate'])
    parser.add_argument('--splits',type=Path,required=True)
    parser.add_argument('--caches',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-legacy-provenance',action='store_true')
    args=parser.parse_args();torch.set_num_threads(3)
    parts,hashes,clusters=load_splits(args.splits)
    cache_info=check_caches(args.caches,parts,args.allow_legacy_provenance)
    config={'plans':plans(),'seed':44,'epochs':50,'patience':10,'batch_size':1024,'rf_trees':400,
        'learning_rate':1e-4,'weight_decay':1e-4,'split_hashes':hashes,
        'mapping_sha256':file_hash(args.splits/'mmseqs2_clusters.csv'),
        'preparation_provenance_sha256':file_hash(args.splits/'split_provenance.json'),'caches':cache_info,
        'code_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in ['train.py','pipeline.py']},
        'helper_sha256':file_hash(Path(__file__).resolve().parents[2]/'drafting_code/snooppi_utils.py'),
        'torch':torch.__version__,'numpy':np.__version__,'max_fit_wall_seconds':3600}
    config_id=digest(json.dumps(config,sort_keys=True))
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    with (root/'active.lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'config.json').exists():
            if json.loads((root/'config.json').read_text())!=config: raise ValueError('Config/cache/split/code identity changed; use a new output directory')
        else:
            if args.stage!='fit': raise ValueError('Fit all comparisons before evaluation')
            save(root/'config.json',config)
        if args.stage=='fit':
            if (root/'final_evaluation_started.json').exists(): raise ValueError('Final test consumed; fitting is closed')
            budget_path=root/'budget.json'
            budget=json.loads(budget_path.read_text()) if budget_path.exists() else {'used_seconds':0}
            remaining=3600-budget['used_seconds']
            if remaining<=0: raise RuntimeError('Compute budget exhausted')
            start=time.monotonic()
            def timeout(*unused): raise TimeoutError('Compute budget exhausted')
            signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(remaining)))
            event(root,'fit_started',config_id=config_id)
            prevalence=float(parts['train'][Y].mean())
            save(root/'baseline_validation.json',{'training_prevalence':prevalence,
                'metrics':summarize_predictions(parts['validation'][Y].to_numpy(),
                    np.full(len(parts['validation']),prevalence),.5)})
            try:
                for plan in config['plans']:
                    directory=root/plan['id']
                    if (directory/'receipt.json').exists(): check_receipt(directory);continue
                    if directory.exists(): raise RuntimeError('Interrupted attempt preserved; review before creating a new run directory: '+plan['id'])
                    directory.mkdir();event(root,'model_started',model=plan['id'])
                    cache,_=load_embedding_cache(args.caches/ENCODERS[plan['encoder']],plan['encoder'])
                    x=build_pair_features(parts['train'],cache,plan['operations'])
                    xv=build_pair_features(parts['validation'],cache,plan['operations']);del cache
                    y=parts['train'][Y].to_numpy();yv=parts['validation'][Y].to_numpy()
                    t=time.monotonic();model,history=fit_model(plan,x,y,xv,yv)
                    p=probability(model,xv);threshold=float(select_threshold(yv,p))
                    name='model.pt' if plan['head']=='MLP' else 'model.joblib'
                    serialize(model,directory/name,x.shape[1]);loaded=deserialize(plan,directory/name)
                    if not np.allclose(p,probability(loaded,xv),rtol=1e-6,atol=1e-7): raise ValueError('Reloaded predictions differ')
                    pred=parts['validation'][['full_pair_id',Y]].copy();pred['probability']=p;pred.to_csv(directory/'validation.csv',index=False)
                    save(directory/'history.json',history)
                    save(directory/'receipt.json',{'config_id':config_id,'threshold':threshold,'seconds':time.monotonic()-t,
                        'metrics':summarize_predictions(yv,p,threshold),'artifacts':{f:file_hash(directory/f) for f in [name,'validation.csv','history.json']}})
                    event(root,'model_complete',model=plan['id']);del x,xv,model,loaded
                save(root/'FROZEN_BATCH.json',{'config_id':config_id,'receipts':{p['id']:file_hash(root/p['id']/'receipt.json') for p in config['plans']}})
                save(root/'RUN_STATE.json',{'status':'READY_FOR_FINAL_EVALUATION','config_id':config_id})
            finally:
                signal.alarm(0);budget['used_seconds']+=time.monotonic()-start;save(budget_path,budget)
            return
        frozen=json.loads((root/'FROZEN_BATCH.json').read_text())
        if frozen['config_id']!=config_id: raise ValueError('Batch configuration mismatch')
        for plan in config['plans']:
            directory=root/plan['id'];check_receipt(directory)
            if file_hash(directory/'receipt.json')!=frozen['receipts'][plan['id']]: raise ValueError('Frozen receipt changed')
        marker=root/'final_evaluation_started.json'
        if not marker.exists(): save(marker,{'config_id':config_id,'policy':'Single frozen batch, resume completed models by artifact identity'})
        elif json.loads(marker.read_text())['config_id']!=config_id: raise ValueError('Test receipt belongs to different configuration')
        result=[]; predictions={};test=parts['test'];labels=test[Y].to_numpy()
        for plan in config['plans']:
            directory=root/plan['id'];receipt=check_receipt(directory);out=directory/'test.csv';done=directory/'test_receipt.json'
            if done.exists():
                if file_hash(out)!=json.loads(done.read_text())['sha256']: raise ValueError('Test predictions changed')
                df=pd.read_csv(out)
                if df.full_pair_id.tolist()!=test.full_pair_id.tolist() or not np.array_equal(df[Y],labels): raise ValueError('Test identity mismatch')
                p=df.probability.to_numpy()
            else:
                cache,_=load_embedding_cache(args.caches/ENCODERS[plan['encoder']],plan['encoder'])
                x=build_pair_features(test,cache,plan['operations']);del cache
                name='model.pt' if plan['head']=='MLP' else 'model.joblib'
                p=probability(deserialize(plan,directory/name),x);del x
                df=test[['full_pair_id',Y]].copy();df['probability']=p;df.to_csv(out,index=False)
                save(done,{'sha256':file_hash(out),'config_id':config_id})
            predictions[plan['id']]=p
            result.append({'model':plan['id'],**summarize_predictions(labels,p,receipt['threshold'])})
        # Fixed prevalence control, fitted only on training labels.
        prevalence=float(parts['train'][Y].mean())
        result.append({'model':'training_prevalence',**summarize_predictions(labels,np.full(len(labels),prevalence),.5)})
        pd.DataFrame(result).to_csv(root/'final_metrics.csv',index=False)
        group_frame=pd.DataFrame({c:[clusters[digest(normalized(s,'full'))] for s in test[c]] for c in [A,B]})
        uncertainty={n:block_uncertainty(group_frame,p,predictions['ProtT5_MLP'],labels) for n,p in predictions.items() if n!='ProtT5_MLP'}
        save(root/'uncertainty.json',uncertainty)
        save(root/'RUN_STATE.json',{'status':'COMPLETE','config_id':config_id,'limitations':['single training seed','historical source test previously exposed','pretraining overlap unknown','cluster heuristic similarity; not a universal homology exclusion']})
        event(root,'final_batch_complete',config_id=config_id)

if __name__=='__main__': main()
