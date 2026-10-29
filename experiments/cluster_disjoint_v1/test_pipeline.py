import unittest
import pandas as pd
from pipeline import A,B,Y,audit,partition,validate,digest,normalized,pair_id

class Gates(unittest.TestCase):
    def frame(self):
        rows=[]
        aa='ACDEFGHIKLMNPQRSTVWY'
        for i in range(2000):
            s=''.join(aa[(i//20**j)%20] for j in range(4))+'ACDEFG'
            rows.append({A:s, B:s+'H',Y:i%2})
        return pd.DataFrame(rows)
    def mapping(self,d):
        return {digest(normalized(s,'full')):digest(normalized(s,'full')) for c in [A,B] for s in d[c]}
    def test_reversed_duplicate(self):
        d=pd.DataFrame([{A:'AAA',B:'CCC',Y:1},{A:'CCC',B:'AAA',Y:1}])
        self.assertEqual(len(audit(d)[0]),1)
    def test_conflict_quarantined(self):
        d=pd.DataFrame([{A:'AAA',B:'CCC',Y:1},{A:'CCC',B:'AAA',Y:0}])
        self.assertEqual(len(audit(d)[1]),2)
        self.assertTrue(audit(d)[0].empty)
    def test_truncation_collision(self):
        d=pd.DataFrame([{A:'A'*1024+'C',B:'CCC',Y:1},{A:'A'*1024+'D',B:'CCC',Y:1}])
        self.assertEqual(len(audit(d)[2]),2)
    def test_invalid_labels(self):
        with self.assertRaises(AssertionError): audit(pd.DataFrame([{A:'AAA',B:'CCC',Y:.5}]))
    def test_invalid_sequence(self):
        with self.assertRaises(AssertionError): audit(pd.DataFrame([{A:'<repr 123>',B:'CCC',Y:1}]))
    def test_partition_and_repeatability(self):
        d=self.frame(); m=self.mapping(d)
        x,_=partition(d,m); z,_=partition(d,m)
        self.assertTrue(x.equals(z)); self.assertTrue(validate(x,m,min_per_class=1))
    def test_injected_reversed_cross_partition_pair(self):
        d=self.frame(); m=self.mapping(d); x,_=partition(d,m)
        row=x[x.split=='train'].iloc[0].copy(); row[A],row[B]=row[B],row[A]; row['split']='test'
        with self.assertRaises(AssertionError): validate(pd.concat([x,row.to_frame().T]),m,1)
    def test_missing_cluster(self):
        d=self.frame();m=self.mapping(d);m.pop(next(iter(m)))
        with self.assertRaises(AssertionError): partition(d,m)
    def test_injected_homolog(self):
        d=self.frame();m=self.mapping(d);x,_=partition(d,m)
        a=x[x.split=='train'].iloc[0][A];b=x[x.split=='test'].iloc[0][B]
        m[digest(normalized(b,'full'))]=m[digest(normalized(a,'full'))]
        with self.assertRaises(AssertionError): validate(x,m,1)

if __name__=='__main__': unittest.main()
