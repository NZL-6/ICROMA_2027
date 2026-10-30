"""Build identically defined, audited event-dependency networks for the optima."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json, math, sys
import networkx as nx

from common import ROOT,ENGINE
HERE=Path(__file__).resolve().parent
sys.path.append(str(ENGINE))
from audit import audit_result

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

sc=read(HERE/'scenario.json');START=sc['start_minutes'];END=sc['end_minutes'];SID=sc['case']
raw=read(HERE/'common_input.json')['data'];emergency=read(HERE/'emergency_state.json')['records']
train_index={t['id']:t for t in raw['trains']}

def build(result):
    schedule=result['schedule'];nodes=[];edges=[];byid={}
    def node(n):nodes.append(n);byid[n['id']]=n
    def ident(t,n,p):return f'{t}:{n}:{p}'
    for t in raw['trains']:
        tid=t['id'];direction=t['runs'][0]['resource'].split('::')[-1]
        for n,(v,e) in enumerate(zip(t['visits'],schedule[tid])):
            for p in ['arrival','departure']:
                node(dict(id=ident(tid,n,p),train=tid,visit=n,phase=p,station=v['station'],direction=direction,
                          planned=v['plan_'+p],time=e[p],delay=max(0,e[p]-v['plan_'+p]),
                          track=e.get('track'),stop=e['stop'],history='fixed_'+p in v,source=False))
    for e in emergency:
        tid=e['train'];node(dict(id='SOURCE:'+tid,train=tid,visit=e['n'],phase='source',station=e['station'],
             direction=train_index[tid]['runs'][0]['resource'].split('::')[-1],planned=None,time=END,delay=None,
             source=True,history=False,track=e['track'],stop=1))
    def edge(a,b,kind,lag,label,resource=None):
        u,v=byid[a],byid[b];slack=v['time']-u['time']-lag
        assert slack>=0,(a,b,kind,slack)
        pressure=max(0,u['time']+lag-v['planned']) if v['planned'] is not None else 0
        active=slack==0 and pressure>0 and (u['source'] or u['delay']>0)
        edges.append(dict(id=f'{kind}|{a}|{b}',source=a,target=b,kind=kind,lag=lag,slack=slack,
              pressure=pressure,active=active,label=label,resource=resource,cross_train=u['train']!=v['train']))
    for t in raw['trains']:
        tid=t['id']
        for n,v in enumerate(t['visits']):
            minimum=0 if n in (0,len(t['visits'])-1) else v.get('minimum_dwell',1)*schedule[tid][n]['stop']
            edge(ident(tid,n,'arrival'),ident(tid,n,'departure'),'dwell',minimum,'同车停靠/通过')
        for n,r in enumerate(t['runs']):
            edge(ident(tid,n,'departure'),ident(tid,n+1,'arrival'),'run',r['minimum'],'同车区间运行',r['resource'])
    groups=defaultdict(list)
    for t in raw['trains']:
        for n,r in enumerate(t['runs']):groups[r['resource']].append((t['id'],n))
    for resource,items in groups.items():
        items.sort(key=lambda z:(schedule[z[0]][z[1]]['departure'],z[0]))
        for (a,n),(b,k) in zip(items,items[1:]):
            edge(ident(a,n,'departure'),ident(b,k,'departure'),'entry_headway',raw['running_resources'][resource]['entry_headway'],'同向区间入口追踪',resource)
            edge(ident(a,n+1,'arrival'),ident(b,k+1,'arrival'),'exit_headway',raw['running_resources'][resource]['exit_headway'],'同向区间出口追踪',resource)
    tracks=defaultdict(list)
    for tid,visits in schedule.items():
        for n,v in enumerate(visits[1:-1],1):tracks[v['station'],v['track']].append((tid,n))
    for (station,track),items in tracks.items():
        items.sort(key=lambda z:(schedule[z[0]][z[1]]['arrival'],z[0]))
        for (a,n),(b,k) in zip(items,items[1:]):
            edge(ident(a,n,'departure'),ident(b,k,'arrival'),'station_track',raw['stations'][station]['clearance'],'同股道释放与接续',station+'::'+track)
    for e in emergency:
        edge('SOURCE:'+e['train'],ident(e['train'],e['n'],'arrival' if e['n']==0 else 'departure'),
             'emergency',0,'规定紧急停车站的恢复后释放约束')
    graph=nx.DiGraph();graph.add_nodes_from(byid);graph.add_edges_from((e['source'],e['target']) for e in edges)
    assert nx.is_directed_acyclic_graph(graph)
    active=nx.DiGraph();active.add_nodes_from(byid);active.add_edges_from((e['source'],e['target']) for e in edges if e['active'])
    for n in nodes:
        n['active_in']=active.in_degree(n['id']);n['active_out']=active.out_degree(n['id'])
    lengths=nx.dag_longest_path_length(active)
    counters=Counter(e['kind'] for e in edges if e['active'])
    cross=[e for e in edges if e['active'] and e['cross_train']]
    train_graph=nx.DiGraph();train_graph.add_nodes_from(schedule)
    train_graph.add_edges_from((byid[e['source']]['train'],byid[e['target']]['train']) for e in cross)
    metrics=dict(event_nodes=sum(not n['source'] for n in nodes),external_source_copies=len(emergency),all_edges=len(edges),
          active_edges=sum(e['active'] for e in edges),active_cross_train_edges=len(cross),
          active_train_pairs=train_graph.number_of_edges(),active_edges_by_type=dict(counters),
          longest_active_path_edges=lengths,max_event_delay=max(n['delay'] for n in nodes if not n['source']),
          mean_terminal_delay=result['objective']/len(schedule),
          last_arrival=max(v[-1]['arrival'] for v in schedule.values()),
          changed_running_durations=sum(schedule[t['id']][n+1]['arrival']-schedule[t['id']][n]['departure']!=r['original_duration']
                                      for t in raw['trains'] for n,r in enumerate(t['runs'])),
          changed_dwell_durations=sum(schedule[t['id']][n]['departure']-schedule[t['id']][n]['arrival']!=v['plan_departure']-v['plan_arrival']
                                      for t in raw['trains'] for n,v in enumerate(t['visits'][1:-1],1)))
    order_changes=0;resources_compared=0
    for resource,items in groups.items():
        for a,b in __import__('itertools').combinations(items,2):
            i,n=a;j,k=b
            old=train_index[i]['visits'][n]['plan_departure']<train_index[j]['visits'][k]['plan_departure']
            new=schedule[i][n]['departure']<schedule[j][k]['departure']
            order_changes+=old!=new;resources_compared+=1
    metrics['changed_unique_orders']=order_changes
    return dict(policy=result['policy'],name=result['name'],objective=result['objective'],lower_bound=result['lower_bound'],
                proven_exact=result['proven_exact'],nodes=nodes,edges=edges,metrics=metrics,schedule=schedule)

def main():
    networks=[];checks=[]
    for p in ['P1','P2','P3']:
        result=read(HERE/f'{p}_optimal.json');mask=read(HERE/'masks'/f'{p}.json')
        check=audit_result(raw,dict(schedule=result['schedule'],objective=result['objective']),mask)
        assert check['passed'] and result['proven_exact'] and result['objective']==round(result['lower_bound'])
        # Explicitly verify the same emergency state, beyond generic rail rules.
        for e in emergency:
            s=result['schedule'][e['train']][e['n']]
            assert s['departure']>=END
            if e['n']>0:assert s['arrival']==e['arrival'] and s['track']==e['track'] and s['stop']==1
        network=build(result);save(HERE/f'{p}_network.json',network);networks.append(network)
        if p in ('P1','P2'):assert network['metrics']['changed_unique_orders']==0
        if p=='P1':assert network['metrics']['changed_running_durations']==0
        with (HERE/f'{p}_network_edges.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(network['edges'][0]));writer.writeheader();writer.writerows(network['edges'])
        checks.append(dict(policy=p,objective=result['objective'],lower_bound=result['lower_bound'],exact=True,
             physical_and_policy_audit=check,network_dag=True,edge_slack_nonnegative=True,metrics=network['metrics']))
    assert networks[0]['objective']>=networks[1]['objective']>=networks[2]['objective']
    bundle=dict(case=SID,now=START,reopening=END,clock_origin='08:00',horizon=336,
         emergency_rule='全部未完成列车就近到站停车；未始发列车停发；08:13后恢复',
         trains=[dict(id=t['id'],direction=t['runs'][0]['resource'].split('::')[-1],fast=t['runs'][0]['original_duration']==4,
              route=[v['station'] for v in t['visits']]) for t in raw['trains']],
         networks=networks,method=dict(node='Fixed arrival/departure event set, 648 nodes per policy; per-train copies of the common 08:13 external release source are additional.',
             edge='Train run/dwell, consecutive same-direction section entry/exit headway, consecutive actual same-track release, common emergency release.',
             slack='actual_target - actual_source - minimum_lag',
             active='slack=0, positive target delay support, and a delayed predecessor or an external source',
             caution='Tight timetable dependencies are observed delay-support relationships, not experimentally identified causal effects. One representative optimum per policy; networks need not be unique.'))
    save(HERE/'comparison_data.json',bundle)
    save(HERE/'validation.json',dict(passed=True,case=SID,checks=checks,input_sha256=hashlib.sha256((HERE/'common_input.json').read_bytes()).hexdigest()))
    template=(HERE/'viewer_template.html').read_text(encoding='utf-8')
    (HERE/'three_policy_delay_networks.html').write_text(template.replace('__DATA__',json.dumps(bundle,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')),encoding='utf-8')
    print(json.dumps([dict(policy=n['policy'],objective=n['objective'],metrics=n['metrics']) for n in networks],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
