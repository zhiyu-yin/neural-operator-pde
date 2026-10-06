"""Predeclared coverage curves and audit field examples, all kept in ignored runs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_plots(root, rows, aggregate, data, predictions, q, diagnostics):
    folder=root/'plots'; folder.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,5,figsize=(16,7),sharex=True,sharey=True)
    for h,ax in enumerate(axes.flat):
        current=rows[h*5:(h+1)*5]
        nominal=np.array([r['nominal'] for r in current])
        coverage=np.array([r['simultaneous_coverage'] for r in current])
        lo=np.array([r['coverage_ci95_low'] for r in current])
        hi=np.array([r['coverage_ci95_high'] for r in current])
        ax.plot([.45,1],[.45,1],'--',color='gray',label='Reference only')
        ax.errorbar(nominal,coverage,yerr=[coverage-lo,hi-coverage],fmt='o-',capsize=3)
        ax.set(title=f't={(h+1)/10:.1f}',xlim=(.45,1),ylim=(.35,1.01))
    fig.supxlabel('Nominal simultaneous-field coverage')
    fig.supylabel('Empirical audit coverage (95% Wilson interval)')
    fig.tight_layout();fig.savefig(folder/'reliability_by_horizon.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    nominal=np.array([r['nominal'] for r in aggregate])
    coverage=np.array([r['mean_horizon_coverage'] for r in aggregate])
    ci=np.array([r['trajectory_bootstrap_ci95'] for r in aggregate])
    ax.plot([.45,1],[.45,1],'--',color='gray')
    ax.errorbar(nominal,coverage,yerr=[coverage-ci[:,0],ci[:,1]-coverage],fmt='o-',capsize=4)
    ax.set(xlabel='Nominal field coverage',ylabel='Mean horizon coverage',title='Whole-trajectory bootstrap; 256 audit units')
    fig.tight_layout();fig.savefig(folder/'reliability_aggregate.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for l,level in enumerate((.5,.7,.8,.9,.95)):
        ax.plot(np.arange(1,11)/10,[rows[h*5+l]['mean_W_rel'] for h in range(10)],label=f'{level:.0%}')
    ax.set(xlabel='Horizon',ylabel='Mean normalized full band width W_rel');ax.legend(title='Nominal level')
    fig.tight_layout();fig.savefig(folder/'normalized_width.png',dpi=140);plt.close(fig)
    examples=[]
    selections=[('predetermined',0),('largest_final_error',int(np.argmax(predictions['error'][:,-1]))),
                ('largest_final_U_rel',int(np.argmax(predictions['U_rel'][:,-1])))]
    for rule,i in selections:
        example={'rule':rule,'audit_trajectory_id':i,'final_error':float(predictions['error'][i,-1]),
                 'final_U_rel':float(predictions['U_rel'][i,-1]),'horizons':[]}
        for h in (1,5,10):
            mean=predictions['mean'][i,h];truth=data['fields'][i,h];spread=predictions['spread'][i,h]
            error=np.abs(truth-mean);width=2*q[h-1,3]*spread
            fields=[truth,mean,error,spread,width]
            fig,axes=plt.subplots(1,5,figsize=(17,3.6))
            for k,(ax,field,title) in enumerate(zip(axes,fields,['Ground Truth','Ensemble Mean','Absolute Error','Raw Ensemble Spread','90% Band Full Width'])):
                lo=min(truth.min(),mean.min()) if k<2 else 0
                hi=max(truth.max(),mean.max()) if k<2 else max(error.max(),spread.max()) if k<4 else width.max()
                im=ax.imshow(field,origin='lower',vmin=lo,vmax=hi,cmap='viridis' if k<2 else 'magma')
                ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
            covered=bool(diagnostics['field_covered'][i,h-1,3])
            fig.suptitle(f'{rule}: audit ID {i}, t={h/10:.1f}; 90% whole field covered: {covered}')
            fig.tight_layout();fig.savefig(folder/f'{rule}_t{h:02d}.png',dpi=140);plt.close(fig)
            example['horizons'].append({'time':h/10,'field_covered_90':covered,
                                       'W_rel_90':float(diagnostics['W_rel'][i,h-1,3]),
                                       'physical_error':float(predictions['error'][i,h])})
        examples.append(example)
    return examples
