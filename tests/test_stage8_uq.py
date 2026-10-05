"""Raw-spread mathematics, independent members, and frozen training-role guards."""
from pathlib import Path
import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch.utils.data import TensorDataset
from neural_operator.normalization import Normalization
from neural_operator.data import trajectory_split
from stage5_training.training import fresh_model, state_hash, loader
from stage8_uq.config import MEMBER_SEEDS, ARCHITECTURE, OPTIMIZER, member_config, D_HASH
from stage8_uq.metrics import ensemble_statistics, spread_scores
from stage8_uq.ensemble import predict_members, rollout_members
from stage8_uq.training import training_inputs

SMALL = dict(modes=2, width=4, blocks=1, projection_width=8)
NORM = Normalization(2., 3., [0, 0, 0], [1, 1, 1])


def test_declared_seeds_and_only_seed_changes():
    assert MEMBER_SEEDS == ((2028,5029),(8101,8201),(8102,8202),(8103,8203),(8104,8204))
    assert len({v for pair in MEMBER_SEEDS for v in pair}) == 10
    expected = member_config(0).to_dict()
    for i in range(5):
        cfg = member_config(i).to_dict()
        for key in expected:
            if key not in ('seed', 'loader_seed'):
                assert cfg[key] == expected[key]
    with pytest.raises(ValueError): member_config(5)


def test_architecture_initialization_storage_optimizer_independence():
    a, b = [fresh_model(member_config(i), SMALL) for i in (1, 2)]
    assert a.config == b.config
    assert [(k, v.shape) for k, v in a.state_dict().items()] == [(k, v.shape) for k, v in b.state_dict().items()]
    assert state_hash(a) != state_hash(b)
    assert state_hash(a) == state_hash(fresh_model(member_config(1), SMALL))
    assert {p.data_ptr() for p in a.parameters()}.isdisjoint({p.data_ptr() for p in b.parameters()})
    oa, ob = [torch.optim.Adam(m.parameters(), **OPTIMIZER) for m in (a, b)]
    before = state_hash(b)
    oa.zero_grad(); a(torch.ones(2,4,8,8)).square().mean().backward(); oa.step()
    assert state_hash(b) == before and not ob.state and oa.state
    assert all(p.grad is None for p in b.parameters())
    ob.zero_grad(); b(torch.ones(2,4,8,8)).square().mean().backward(); ob.step()
    assert oa.state is not ob.state
    first_a, first_b = next(iter(oa.state.values())), next(iter(ob.state.values()))
    assert first_a['exp_avg'].data_ptr() != first_b['exp_avg'].data_ptr()


def test_loader_generators_are_independent_and_reproducible():
    data = TensorDataset(torch.arange(48))
    a, b = [loader(data, member_config(i), shuffle=True) for i in (1, 2)]
    state = b.generator.get_state().clone()
    order_a = torch.cat([batch[0] for batch in a])
    assert torch.equal(state, b.generator.get_state()) and a.generator is not b.generator
    order_b = torch.cat([batch[0] for batch in b])
    assert not torch.equal(order_a, order_b)
    repeat = loader(data, member_config(1), shuffle=True)
    assert torch.equal(order_a, torch.cat([batch[0] for batch in repeat]))


def test_controlled_sample_statistics():
    values = np.arange(5, dtype=float)[:,None,None,None,None] * np.ones((5,2,1,4,4))
    mean, spread = ensemble_statistics(values)
    assert mean.shape == spread.shape == (2,1,4,4)
    np.testing.assert_array_equal(mean, 2)
    np.testing.assert_allclose(spread, np.sqrt(2.5))
    assert np.all(spread > 0)
    _, identical = ensemble_statistics(np.full_like(values, 0.1))
    np.testing.assert_array_equal(identical, 0)


@pytest.mark.parametrize('shape', [(1,2,1,4,4), (5,2,2,4,4), (5,2,4,4)])
def test_invalid_ensemble_shapes(shape):
    with pytest.raises(ValueError): ensemble_statistics(np.ones(shape))


def test_nonfinite_members_are_not_dropped():
    values = np.ones((5,2,1,4,4)); values[0,0,0,0,0] = np.nan
    with pytest.raises(FloatingPointError): ensemble_statistics(values)


def test_rms_relative_and_floor():
    mean, spread = np.full((2,3,1,4,4), 4.), np.full((2,3,1,4,4), 2.)
    scores = spread_scores(mean, spread, 3.)
    np.testing.assert_array_equal(scores['U_rms'], 2)
    np.testing.assert_array_equal(scores['U_rel'], .5)
    assert not scores['floor_active'].any() and scores['epsilon_rms'] == 3e-12
    mean[:] = 0
    scores = spread_scores(mean, spread, 3.)
    assert scores['floor_active'].all()
    np.testing.assert_allclose(scores['U_rel'], 2 / 3e-12)


class Advance(torch.nn.Module):
    def __init__(self, factor):
        super().__init__(); self.factor = factor; self.conditions = []
    def forward(self, x):
        self.conditions.append(x[:,1:].clone())
        return x[:,:1] * self.factor


def test_physical_units_and_own_feedback_fixed_coefficients():
    fields = np.full((2,4,8,8), 5.)
    coefficients = np.array([[.2,.3,.01],[-.4,.6,.02]])
    models = [Advance(1), Advance(2)]
    one = predict_members(models, NORM, fields[:,0], coefficients)
    assert one['members'].shape == (2,2,1,8,8)
    np.testing.assert_array_equal(one['mean'], 6.5)
    np.testing.assert_allclose(one['spread'], 3 / np.sqrt(2))
    ar = rollout_members(models, NORM, fields, coefficients, np.arange(4)*.1)
    assert ar['members'].shape == (2,2,4,1,8,8)
    np.testing.assert_array_equal(ar['members'][0], 5)
    for k in range(4):
        np.testing.assert_array_equal(ar['members'][1,:,k], 2+3*2**k)
    np.testing.assert_array_equal(ar['spread'][:,0], 0)
    for model in models:
        for condition in model.conditions:
            np.testing.assert_allclose(condition[:,:,0,0], coefficients)


def test_roles_and_no_refit(monkeypatch):
    monkeypatch.setattr(Normalization, 'fit', lambda *args: pytest.fail('Normalization refit'))
    splits = trajectory_split()
    data = {'fields':np.ones((64,11,64,64)), 'coefficients':np.ones((64,3)), 'times':np.linspace(0,1,11)}
    dataset, validation_ids, norm = training_inputs(data, {'split_ids':splits, 'normalization':NORM.to_dict()})
    assert dataset.pairs == [(i,k) for i in splits['train'] for k in range(10)]
    assert validation_ids == splits['validation']
    assert set(validation_ids).isdisjoint(splits['test'])
    assert norm.to_dict() == NORM.to_dict()


def test_real_member_zero_and_frozen_context(monkeypatch):
    if not Path('runs/stage6/best.pt').exists():
        pytest.skip('Historical artifacts omitted from public Git; sanity command requires them')
    from stage8_uq.checkpoint import frozen_protocol, load_d
    monkeypatch.setattr(Normalization, 'fit', lambda *args: pytest.fail('Normalization refit'))
    protocol = frozen_protocol()
    model, norm, selected = load_d('runs/stage6')
    assert selected['sha256'] == D_HASH and model.config == ARCHITECTURE
    assert norm.to_dict() == protocol['normalization']
    new = fresh_model(member_config(1), ARCHITECTURE)
    original_initial = fresh_model(member_config(0), ARCHITECTURE)
    assert state_hash(original_initial) == protocol['initial_state_sha256']
    assert state_hash(new) != state_hash(original_initial)
    assert [(k,v.shape) for k,v in model.state_dict().items()] == [(k,v.shape) for k,v in new.state_dict().items()]


def test_failed_gate_stops_member(tmp_path, monkeypatch):
    import stage8_uq.training as training
    frozen = {'architecture':SMALL}
    monkeypatch.setattr(training, 'training_inputs', lambda *args:([], [], NORM))
    monkeypatch.setattr(training, 'make_protocol', lambda i,h,f:{'architecture':SMALL})
    monkeypatch.setattr(training, 'tiny_gate', lambda *args:{'passed':False})
    monkeypatch.setattr(training, 'full_training', lambda *args:pytest.fail('Training after failed gate'))
    with pytest.raises(RuntimeError, match='gate failed'):
        training.train_member(1, tmp_path, frozen, {})
    assert (tmp_path/'member_1/failure.json').exists()


def test_manifest_round_trip_and_tamper(tmp_path, monkeypatch):
    from stage8_uq import checkpoint as cp
    from stage6_control.config import budget
    from stage4_evaluation.checkpoint import sha256, write_json
    folder = tmp_path/'member_1'; folder.mkdir()
    model = fresh_model(member_config(1), SMALL)
    frozen = {'architecture':SMALL, 'normalization':NORM.to_dict(), 'split_ids':trajectory_split(),
              'optimizer':{}, 'objective':'one-step normalized MSE only', 'selection_rule':'validation',
              'training_archive_sha256':'archive', 'trainable_parameter_elements':1,
              'trainable_real_scalar_parameters':1, 'completed_budget':budget(60)}
    protocol = {**frozen, 'member_id':1, 'config':member_config(1).to_dict(),
                'initial_state_sha256':state_hash(model), 'normalization_sha256':cp.object_hash(NORM.to_dict()),
                'frozen_d_protocol_sha256':'protocol', 'source_sha256':{}}
    state = {k:protocol[k] for k in ('architecture','normalization','split_ids','config','initial_state_sha256')}
    state.update(model_state_dict=model.state_dict(), epoch=50, validation={'final_relative_l2':.1})
    torch.save(state, folder/'best.pt')
    selection = {'sha256':sha256(folder/'best.pt'),'epoch':50,'completed_budget':budget(60),
                 'validation':state['validation'],'runtime_seconds':1.,'training_only_seconds':.8}
    gate = {'runtime_seconds':.1,'updates':1,'passed':True}
    for name,value in [('protocol',protocol),('selection',selection),('gate',gate)]:
        write_json(folder/f'{name}.json',value)
    cp.freeze_manifest(folder,protocol,selection,gate,folder/'best.pt')
    original = cp.verify_hash
    monkeypatch.setattr(cp,'verify_hash',lambda p,h:None if h in ('archive','protocol') else original(p,h))
    loaded,norm,manifest = cp.load_member(folder,frozen)
    assert state_hash(loaded) == state_hash(model)
    assert norm.to_dict() == NORM.to_dict()
    manifest['config']['seed'] = 999
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='seeds'):cp.load_member(folder,frozen)


def test_full_budget_validation_only_and_fresh_restart(tmp_path, monkeypatch):
    import stage8_uq.training as training
    config = member_config(1)
    model = fresh_model(config, SMALL)
    protocol = {'architecture':SMALL,'normalization':NORM.to_dict(),
                'split_ids':trajectory_split(),'config':config.to_dict(),
                'initial_state_sha256':state_hash(model),'member_id':1}
    calls = []
    def fake_train(model, batches, optimizer):
        if not calls:
            assert state_hash(model) == protocol['initial_state_sha256']
        calls.append('train')
        return 1.
    def fake_validation(model, data, ids, norm):
        assert ids == protocol['split_ids']['validation']
        assert set(ids).isdisjoint(protocol['split_ids']['test'])
        return {'final_relative_l2':1.,'final_mass_error':0.}
    monkeypatch.setattr(training,'train_epoch',fake_train)
    monkeypatch.setattr(training,'validation_metrics',fake_validation)
    monkeypatch.setattr(training,'loader',lambda *args,**kwargs: [])
    selection = training.full_training(tmp_path,protocol,{},[],protocol['split_ids']['validation'],NORM,config)
    assert len(calls) == 60
    assert selection['completed_budget']['optimizer_updates'] == 3600
    assert selection['epoch'] == 1  # Exact ties prefer the earlier epoch.
    assert not selection['test_or_ood_used']
