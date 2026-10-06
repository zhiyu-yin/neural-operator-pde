"""Fixed final-horizon examples and all-regime context; plots stay in ignored runs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def select_examples(data):
    velocity=data['high_velocity']
    return [('predetermined_fresh','fresh_id:0'),('predetermined_high_nu','high_nu:0'),
            ('predetermined_velocity','velocity_pp:0'),
            ('worst_velocity_error',str(velocity['qualified_ids'][np.argmax(velocity['error'][:,-1])])),
            ('highest_velocity_U_rel',str(velocity['qualified_ids'][np.argmax(velocity['U_rel'][:,-1])]))]


def make_plots(root,data,summary,load_raw,q):
    folder=root/'plots';folder.mkdir(exist_ok=False)
    times=np.arange(1,11)/10
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for role in ('fresh_id','high_nu','high_velocity'):
        rows=summary['regimes'][role]['by_horizon']
        for ax,key in zip(axes,['error','U_rel']):
            ax.plot(times,[r['mean_'+key] for r in rows],marker='o',label=role)
            ax.set(xlabel='Horizon',ylabel='Mean '+key)
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(folder/'regime_trends.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for role in ('fresh_id','high_nu','high_velocity'):
        rows=[r for r in summary['regimes'][role]['coverage'] if r['nominal']==.9]
        axes[0].plot(times,[r['coverage'] for r in rows],marker='o',label=role)
        axes[1].plot(times,[r['mean_W_rel'] for r in rows],marker='o',label=role)
    axes[0].axhline(.9,ls='--',color='gray',label='ID nominal reference only')
    axes[0].set(xlabel='Horizon',ylabel='Frozen-ID 90% whole-field coverage',ylim=(0,1.02))
    axes[1].set(xlabel='Horizon',ylabel='Mean frozen-ID 90% full width W_rel')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(folder/'coverage_width_stress.png',dpi=140);plt.close(fig)
    examples=[]
    for rule,qualified in select_examples(data):
        role,raw_id=qualified.split(':');raw_id=int(raw_id)
        raw=load_raw(role);i=int(np.flatnonzero(raw['trajectory_ids']==raw_id)[0])
        truth=raw['truth'][i,-1];mean=raw['mean'][i,-1];spread=raw['spread'][i,-1]
        fields=[truth,mean,np.abs(truth-mean),spread,2*q[-1,3]*spread]
        fig,axes=plt.subplots(1,5,figsize=(17,3.6))
        for k,(ax,field,title) in enumerate(zip(axes,fields,['Ground Truth','Ensemble Mean','Absolute Error','Raw Ensemble Spread','Frozen-ID 90% Full Width'])):
            lo=min(fields[0].min(),fields[1].min()) if k<2 else 0
            hi=max(fields[0].max(),fields[1].max()) if k<2 else max(fields[2].max(),fields[3].max()) if k<4 else fields[4].max()
            im=ax.imshow(field,origin='lower',vmin=lo,vmax=hi,cmap='viridis' if k<2 else 'magma')
            ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f'{rule}: {qualified}, t=1.0')
        fig.tight_layout();fig.savefig(folder/(rule+'.png'),dpi=140);plt.close(fig)
        examples.append({'rule':rule,'qualified_id':qualified,'final_error':float(data[role]['error'][i,-1]),
                         'final_U_rel':float(data[role]['U_rel'][i,-1]),'final_W_rel90':float(data[role]['W_rel'][i,-1,3]),
                         'covered90':bool(data[role]['covered'][i,-1,3])})
    return examples
