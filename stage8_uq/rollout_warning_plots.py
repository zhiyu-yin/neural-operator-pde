"""Predeclared diagnostic plots for the existing ten prediction horizons."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def select_examples(arrays,result):
    error,u=arrays['error'],arrays['U_rel']
    examples=[{'rule':'predetermined','trajectory_id':0},
              {'rule':'largest_final_error','trajectory_id':int(np.argmax(error[:,-1]))},
              {'rule':'largest_final_U_rel','trajectory_id':int(np.argmax(u[:,-1]))},
              {'rule':'largest_error_increase_0.5_to_1.0','trajectory_id':int(np.argmax(error[:,-1]-error[:,4]))}]
    for row in result['early_warning']:
        missed=row['new_failure']['missed_event_indices']
        if missed:
            examples.append({'rule':'first_missed_new_failure','trajectory_id':min(missed),
                             'source_time':row['time'],'target_time':row['target_time'],'lag':row['lag'],
                             'event_count_in_comparison':row['new_failure']['event_count'],
                             'comparison_status':row['new_failure']['status']})
            break
    return examples


def make_plots(folder,arrays,result,primary=False):
    folder.mkdir(exist_ok=False)
    times=arrays['times'];rows=result['by_horizon']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,label in zip(axes,['E','U_rel','W_rel90'],['Physical relative L2 error','Raw U_rel','90% full width W_rel']):
        ax.plot(times,[r['mean_'+key] for r in rows],'o-',label='Mean')
        ax.plot(times,[r['median_'+key] for r in rows],'--',label='Median')
        ax.set(xlabel='Horizon',ylabel=label);ax.legend()
    fig.tight_layout();fig.savefig(folder/'rollout_trends.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['U_rel','W_rel90']):
        ax.fill_between(times,[r[key+'_q10'] for r in rows],[r[key+'_q90'] for r in rows],alpha=.25,label='10th–90th percentile')
        ax.plot(times,[r[key+'_median'] for r in rows],'o-',label='Median')
        ax.set(xlabel='Horizon',ylabel=key);ax.legend()
    fig.tight_layout();fig.savefig(folder/'saturation.png',dpi=140);plt.close(fig)
    examples=select_examples(arrays,result) if primary else []
    for e in examples:
        i=e['trajectory_id'];fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,key,label in zip(axes,['error','U_rel','width'],['E(t)','U_rel(t)','90% W_rel(t)']):
            ax.plot(times,arrays[key][i],'o-');ax.set(xlabel='Horizon',ylabel=label)
            if 'source_time' in e:
                ax.axvspan(e['source_time'],e['target_time'],alpha=.15,color='orange')
        fig.suptitle(f"{e['rule']}: audit trajectory {i}")
        fig.tight_layout();fig.savefig(folder/(e['rule']+'.png'),dpi=140);plt.close(fig)
        e.update(final_error=float(arrays['error'][i,-1]),final_U_rel=float(arrays['U_rel'][i,-1]),
                 final_W_rel90=float(arrays['width'][i,-1]),error_increase_0_5_to_1=float(arrays['error'][i,-1]-arrays['error'][i,4]))
    return examples
