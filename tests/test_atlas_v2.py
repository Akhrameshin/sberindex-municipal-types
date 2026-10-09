import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,pytest
from graph_support import weighted_knn
from atlas_v2 import aggregate_panel,build_arrays,scale_arrays,viterbi_costs,assigned_margin,centre_windows,CFG

def test_identical_profiles_keep_zero_distance_edges():
    x=np.array([[1.,2.],[1.,2.],[2.,1.],[3.,1.]])
    a=weighted_knn(x,'euclidean',k=2,minimum=1)
    assert a[0,1]>0 and np.allclose(a.toarray(),a.T.toarray())
    assert not a.diagonal().any()

def test_precomputed_missing_routes_do_not_create_edges():
    d=np.array([[0.,0.,np.inf],[0.,0.,np.inf],[np.inf,np.inf,0.]])
    a=weighted_knn(d,'precomputed',k=2,minimum=1)
    assert a[0,1]>0 and a.getrow(2).nnz==0

def test_lag_correlation_normalises_each_overlap():
    from edge_rules import lag_corr_dist
    rng=np.random.default_rng(2);series=rng.normal(size=(12,8,7));series[-3:]*=100
    d,_=lag_corr_dist(list(series),3)
    assert (d>=0).all() and (d<=2).all() and np.allclose(d,d.T)

def test_distance_cache_uses_input_content():
    from edge_rules import _dist,_DCACHE
    _DCACHE.clear();r=np.random.default_rng(12);x=r.normal(size=(10,4,7));y=x.copy();y[:,0,0]=np.arange(10)*20
    a=_dist('lag_corr',list(x));b=_dist('lag_corr',list(y))
    assert len(_DCACHE)==2 and not np.allclose(a,b)

def test_viterbi_exact_conditional_cost():
    r=np.random.default_rng(5);d=r.uniform(size=(5,4,3));labels,total=viterbi_costs(d,.2)
    best=0
    for i in range(4):
        costs=[sum(d[t,i,p[t]] for t in range(5))+.2*sum(p[t]!=p[t-1] for t in range(1,5)) for p in itertools.product(range(3),repeat=5)]
        best+=min(costs)
    assert total==pytest.approx(best)
    assert total==pytest.approx(sum(d[t,i,labels[t,i]] for t in range(5) for i in range(4))+.2*np.sum(labels[1:]!=labels[:-1]))

def test_signed_margin_follows_assigned_trajectory():
    d=np.array([[[1.,4.,9.]]]);lab=np.array([[1]])
    margin,alt=assigned_margin(d,lab)
    assert margin[0,0]==pytest.approx(-.6) and alt[0,0]==0

def city_panel():
    rows=[]
    for tid,pop,emp,wage,total in [('a',100.,10.,1000.,20.),('b',300.,30.,2000.,100.)]:
        for month in ['2023-01','2023-02']:
            rows.append(dict(tid=tid,territory_id=1,month=month,municipal_district_name='Внутригородское муниципальное образование '+tid,
                             region_name='Москва',pop=pop,emp_total=emp,wage=wage,total=total,health=total*.1,catering=total*.1,
                             food=total*.3,market=total*.1,transport=total*.1,emp_manuf=.2 if tid=='a' else .8,
                             municipal_district_center_lat=55.,municipal_district_center_lon=37.,market_access=1.,oktmo8='1'))
    return pd.DataFrame(rows)

def test_city_aggregation_uses_original_levels_and_employment():
    p,m=aggregate_panel(city_panel());row=p.iloc[0]
    assert row.total==pytest.approx(80.) and row['pop']==400.
    assert row.wage==pytest.approx(1750.) and row.emp_manuf==pytest.approx(.65)
    assert len(m)==2 and row.tid=='city-moscow'

def test_city_missing_population_coverage_excludes_month():
    panel=city_panel();panel=panel[~((panel.tid=='b')&(panel.month=='2023-02'))]
    p,_=aggregate_panel(panel,coverage=.99)
    assert list(p.month)==['2023-01']

def test_training_scalers_ignore_heldout_extreme():
    rng=np.random.default_rng(7);a={'spending':rng.normal(size=(2,8,7)),'context':rng.normal(size=(8,9))}
    train=np.arange(7);x,t=scale_arrays(a,training_indices=train);b={k:v.copy() for k,v in a.items()};b['spending'][:,7]=1e8;b['context'][7]=1e8
    y,u=scale_arrays(b,training_indices=train)
    assert np.allclose(x[:,train],y[:,train])
    assert np.allclose(t['spend_scaler'].mean_,u['spend_scaler'].mean_)

def test_unknown_sector_mass_is_explicit():
    from atlas_assignment import context_profile
    from atlas_v2 import GROUPS
    names=['emp_'+x for sectors in GROUPS.values() for x in sectors]
    m=pd.Series({c:np.nan for c in names});m['emp_gov']=.3;m['emp_edu']=.2;m['pop']=100;m['wage']=1000;m['emp_total']=10
    c=context_profile(m)
    assert c[-1]==pytest.approx(.5) and np.sum(c[2:])==pytest.approx(1)

def test_centroid_projection_does_not_mutate_train_graph():
    from atlas_v2 import fit_model
    from atlas_experiments import inductive_labels
    rng=np.random.default_rng(12);x=rng.normal(size=(1,30,16));f=fit_model(x,3)
    before=f['graphs'][0].copy();c=f['centers'].copy();pred=inductive_labels(x[0],rng.normal(size=(4,16)),f)
    assert pred.shape==(4,) and (f['graphs'][0]!=before).nnz==0 and np.array_equal(c,f['centers'])

def test_late_transitions_are_censored():
    from atlas_experiments import transitions
    a={'ids':['x'],'months':['2024-'+str(i+1).zfill(2) for i in range(5)],'meta':pd.DataFrame([{'municipal_district_name':'X','region_name':'R'}])}
    labels=np.array([[0],[0],[0],[1],[1]]);fit={'labels':labels,'distances':np.array([[[0.,1.]]]*3+[[[1.,0.]]]*2)}
    t=transitions(a,fit,{'confidence':np.ones((5,1))})
    assert t.iloc[0].censored and not t.iloc[0].robust_descriptive

def test_degenerate_partition_has_no_fictitious_feature_scores():
    from icvi import all_indices
    from scipy import sparse
    r=all_indices(np.arange(12).reshape(6,2),sparse.csr_matrix(np.ones((6,6))),np.zeros(6))
    assert np.isnan(r['SW']) and np.isnan(r['CH']) and np.isnan(r['S_Dbw'])

def test_validation_units_are_preserved():
    f=Path(__file__).resolve().parents[1]/'data/external/validation.csv'
    if not f.exists():pytest.skip('External prepared tables not supplied')
    d=pd.read_csv(f)
    assert set(d[d.indicator.eq('housing')].unit_multiplier)=={1}
    assert set(d[d.indicator.eq('retail')].unit_multiplier)=={1000}

def test_controls_remove_spurious_type_effect_from_population():
    from atlas_experiments import _within_effect,_controlled_effect
    rng=np.random.default_rng(11);c=rng.normal(size=(500,2));reg=np.repeat(np.arange(10),50);lab=(c[:,0]>0).astype(int)
    y=c[:,0]+rng.normal(0,.02,500)
    assert _within_effect(y,lab,reg)>.4
    assert abs(_controlled_effect(y,lab,reg,c))<.001


def test_window_centering_removes_common_drift_without_touching_input():
    rng=np.random.default_rng(5);s=rng.normal(size=(13,40,7))+np.linspace(0,2,13)[:,None,None]
    a={'spending':s.copy()};b=centre_windows(a)
    assert np.allclose(np.median(b['spending'],axis=1),0) and np.array_equal(a['spending'],s)
    c=centre_windows(a,[0])
    assert np.allclose(np.median(c['spending'][:,:,0],axis=1),0) and np.array_equal(c['spending'][:,:,1:],s[:,:,1:])


def test_null_rewire_preserves_degrees_and_weights():
    import random, igraph as ig, scipy.sparse as sp
    from atlas_null import rewire
    ig.set_random_number_generator(random.Random(0))
    rng = np.random.default_rng(0)
    g = ig.Graph.Erdos_Renyi(n=60, m=180, directed=False)
    edges = np.array(g.get_edgelist())
    w = rng.random(len(edges)) + .5
    a = sp.coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(60, 60)); a = (a + a.T).tocsr()
    b = rewire(a, rng)
    assert (b != b.T).nnz == 0
    assert np.array_equal(np.asarray((a > 0).sum(1)).ravel(), np.asarray((b > 0).sum(1)).ravel())
    assert np.isclose(np.sort(a.data).sum(), np.sort(b.data).sum())


def test_null_z_is_positive_for_planted_partition():
    from icvi import all_indices
    rng = np.random.default_rng(1)
    lab = np.repeat(np.arange(3), 40)
    x = rng.normal(size=(120, 4)) + 3 * np.eye(3, 4)[lab]
    a = (lab[:, None] == lab[None]).astype(float) * 0.2; np.fill_diagonal(a, 0)
    real = all_indices(x, a, lab)
    null = all_indices(x, a, rng.permutation(lab))
    assert real['SW'] > null['SW'] and real['AVI'] > null['AVI'] and real['AVU'] < null['AVU']
