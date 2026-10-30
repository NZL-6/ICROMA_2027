"""Target-blind covariates from P1 (and P2 for the staged comparison).

All windows and transforms are fixed here before reading target decisions.
No feature selection on associations. See feature_manifest.json for provenance.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,sys,math,itertools,time
import numpy as np
import pandas as pd
import networkx as nx
ROOT=Path(__file__).resolve().parent.parent
ENGINE=ROOT/'vendor'
sys.path.insert(0,str(ENGINE))
from preprocess import preprocess
CAT=json.loads((ROOT/'candidate_catalog.json').read_text(encoding='utf-8'))['records']
REG={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def put(d,k,v,definition=None):
    if isinstance(v,(bool,np.bool_)):v=int(v)
    if isinstance(v,(np.integer,np.floating)):v=v.item()
    d[k]=v;REG[k]=dict(catalog_id=k.split('_')[0],definition=definition or k,source='reference timetable / original plan / physical input only')
def ratio(a,b):return a/b if b else np.nan
def stats(d,k,x):
    x=np.array(list(x),dtype=float);x=x[np.isfinite(x)]
    for suffix,v in [('count',len(x)),('sum',x.sum() if len(x) else 0),('mean',x.mean() if len(x) else np.nan),('max',x.max() if len(x) else np.nan),('min',x.min() if len(x) else np.nan),('sd',x.std() if len(x) else np.nan),('median',np.median(x) if len(x) else np.nan),('iqr',np.quantile(x,.75)-np.quantile(x,.25) if len(x) else np.nan),('q90',np.quantile(x,.9) if len(x) else np.nan),('zero_fraction',np.mean(x==0) if len(x) else np.nan)]:put(d,k+'_'+suffix,v)
def entropy(counts):
    x=np.array(list(counts),dtype=float);x=x[x>0];p=x/x.sum() if len(x) else x
    return float(-np.sum(p*np.log(p)))
def eid(t,n,p):return f'{t}:{n}:{p}'
def branch(s):return s[:2] if s[0] in 'UD' else 'trunk'

class Case:
    def __init__(self,folder,reference):
        self.folder=folder;self.sc=read(folder/'scenario.json');self.B=self.sc['start_minutes'];self.E=self.sc['end_minutes'];self.L=self.E-self.B;self.closed=self.sc['section']
        self.raw=read(folder/'common_input.json')['data'];self.data,_=preprocess(self.raw,{})
        self.trains={t['id']:t for t in self.raw['trains']};self.prep={t['id']:t for t in self.data['trains']}
        self.sch=read(folder/f'{reference}_optimal.json')['schedule'];self.reference=reference
        self.em={e['train']:e for e in read(folder/'emergency_state.json')['records']}
        self.units=read(folder/'physical_variables.json');self.net=read(folder/f'{reference}_network.json')
        self.nodes={n['id']:n for n in self.net['nodes'] if not n['source']}
        self.g=nx.DiGraph();self.g.add_nodes_from(self.nodes);self.g.add_node('MASTER')
        self.inc=defaultdict(list);self.out=defaultdict(list)
        for edge in self.net['edges']:
            e=dict(edge);e['source']='MASTER' if e['source'].startswith('SOURCE:') else e['source']
            self.g.add_edge(e['source'],e['target'],**e);self.inc[e['target']].append(e);self.out[e['source']].append(e)
        self.active=nx.DiGraph();self.active.add_nodes_from(self.g);self.active.add_edges_from((u,v) for u,v,d in self.g.edges(data=True) if d['active'])
        self.anc={u:nx.ancestors(self.active,u) for u in self.nodes};self.desc={u:nx.descendants(self.active,u) for u in self.nodes}
        self.ug=self.active.to_undirected();self.ug.remove_nodes_from(list(nx.isolates(self.ug)))
        self.between=nx.betweenness_centrality(self.active,normalized=True)
        self.pr=nx.pagerank(self.active,alpha=.85)
        self.core=nx.core_number(self.ug) if len(self.ug) else {};self.clust=nx.clustering(self.ug)
        self.comp={u:c for c in nx.weakly_connected_components(self.active) for u in c}
        self.communities=list(nx.community.greedy_modularity_communities(self.ug)) if len(self.ug.edges) else []
        self.community={u:i for i,c in enumerate(self.communities) for u in c}
        self.dist={u:dict(nx.single_source_shortest_path_length(self.active,u)) for u in self.active}
        self.source_dist=nx.single_source_shortest_path_length(self.active,'MASTER')
        self.longest={u:0 for u in self.active};self.crosslong={u:0 for u in self.active};self.lagdist={'MASTER':0}
        for u in nx.topological_sort(self.active):
            for v in self.active.successors(u):
                self.longest[v]=max(self.longest[v],self.longest[u]+1)
                self.crosslong[v]=max(self.crosslong[v],self.crosslong[u]+int(self.g[u][v]['cross_train']))
                if u in self.lagdist:self.lagdist[v]=min(self.lagdist.get(v,1e9),self.lagdist[u]+self.g[u][v]['lag'])
        self.fac=nx.Graph();self.fac.add_nodes_from(self.raw['stations'])
        for r in self.raw['running_resources'].values():self.fac.add_edge(*r['physical_section'].split('--'))
        self.fbet=nx.betweenness_centrality(self.fac);self.cut=set(nx.articulation_points(self.fac))
        self.vis=defaultdict(list);self.runs=defaultdict(list);self.planruns=defaultdict(list)
        for tid,t in self.trains.items():
            direction=t['runs'][0]['resource'].split('::')[-1]
            for n,(v,s) in enumerate(zip(t['visits'],self.sch[tid])):
                if n not in (0,len(t['visits'])-1):self.vis[v['station'],direction].append((s['arrival'],s['departure']+1,tid,n))
            for n,r in enumerate(t['runs']):
                self.runs[r['resource']].append((self.sch[tid][n]['departure'],self.sch[tid][n+1]['arrival'],tid,n))
                self.planruns[r['resource']].append((t['visits'][n]['plan_departure'],t['visits'][n+1]['plan_arrival'],tid,n))
        for values in itertools.chain(self.runs.values(),self.planruns.values()):values.sort()
        self.base_global=self.global_features();self.node_cache={};self.visit_cache={}

    def global_features(self):
        d={};B,E,L=self.B,self.E,self.L
        for k,v in {'D01_start':B,'D02_end':E,'D03_duration':L,'D04_start_phase40':B%40,'D05_end_phase40':E%40,'D06_decision_lag':0,'D07_elapsed':0,'D08_remaining':L,'D11_H1_C':int(self.closed=='H1--C'),'D11_U13_H1':int(self.closed=='U13--H1'),'D11_U11_U12':int(self.closed=='U11--U12'),'D12_section':1,'D14_sections':1,'D15_directions':2,'D16_run_resources':2,'D17_station_tracks_lost':0,'D18_capacity_loss_fraction':1,'D19_residual_fraction':0,'D20_lost_resource_minutes':2*L,'D21_reopen_stages':1,'D22_step_recovery':1,'D23_post_reopen_restriction':0,'D24_borrow_opposite':0,'D25_simultaneous_disruptions':1,'I04_known_reopen':1,'I05_predicted_remaining':L,'I06_lower':L,'I06_upper':L,'I07_prediction_width':0,'I08_variance':0,'I09_q10':L,'I09_q50':L,'I09_q90':L,'I10_updates':0,'I14_known_location':1,'I15_known_step':1}.items():put(d,k,v)
        kinds=Counter(e['kind'] for e in self.em.values());stations=Counter(e['station'] for e in self.em.values() if e['n']>0)
        put(d,'E12_origin_waiting',kinds['origin_hold']);put(d,'E13_station_waiting',sum(stations.values()));put(d,'E16_backlog',len(self.em))
        release=[self.sch[t][e['n']]['departure'] for t,e in self.em.items()]
        put(d,'E17_release_first5_fraction',np.mean(np.array(release)<E+5) if release else 0)
        put(d,'E18_parking_HHI',sum((v/sum(stations.values()))**2 for v in stations.values()) if stations else 0)
        put(d,'E19_fixed_events',sum('fixed_'+p in v for t in self.trains.values() for v in t['visits'] for p in ['arrival','departure']))
        put(d,'E20_planned_starts_before_reopen',sum(B<t['visits'][0]['plan_departure']<E for t in self.trains.values()))
        for kind in ['origin_hold','finish_committed_run_then_hold','hold_current_station']:put(d,'E21_'+kind,kinds[kind])
        put(d,'E21_completed',sum(t['visits'][-1]['plan_arrival']<=B for t in self.trains.values()))
        put(d,'E22_in_closed_section',0);put(d,'E24_prior_delay',0);put(d,'E25_prior_delay_mean',0)
        put(d,'E26_committed_runs',sum(t['visits'][n]['plan_departure']<B<t['visits'][n+1]['plan_arrival'] for t in self.trains.values() for n in range(len(t['runs']))))
        delays=[max(0,self.sch[t][-1]['arrival']-v['visits'][-1]['plan_arrival']) for t,v in self.trains.items()]
        put(d,'B10_affected_fraction',np.mean(np.array(delays)>0));stats(d,'B11_root_longest_paths',self.longest.values())
        for label,key in [('station','station'),('time20','time')]:
            c=Counter()
            for n in self.nodes.values():c[n[key] if key=='station' else int(n[key]//20)]+=n['delay']
            put(d,'B12_'+label+'_entropy',entropy(c.values()));put(d,'B12_'+label+'_HHI',sum((x/sum(c.values()))**2 for x in c.values()) if sum(c.values()) else 0)
        put(d,'B14_physical_sources',1);put(d,'B14_policy_sources',1);put(d,'B14_shared_release_sources',1)
        stats(d,'B15_in_degree',[self.active.in_degree(u) for u in self.nodes]);stats(d,'B15_out_degree',[self.active.out_degree(u) for u in self.nodes])
        finite=[z for u,dd in self.dist.items() for v,z in dd.items() if u!=v]
        put(d,'B16_reachable_pair_fraction',len(finite)/(len(self.active)*(len(self.active)-1)));put(d,'B16_mean_reachable_distance',np.mean(finite) if finite else np.nan)
        # Overlap at train-pair layer: event relation layers can share train pairs.
        layers={k:set() for k in ['entry_headway','exit_headway','station_track']}
        for e in self.net['edges']:
            if e['kind'] in layers:layers[e['kind']].add(tuple(sorted((self.nodes[e['source']]['train'],self.nodes[e['target']]['train']))))
        for a,b in itertools.combinations(layers,2):put(d,'B17_'+a+'_'+b+'_Jaccard',ratio(len(layers[a]&layers[b]),len(layers[a]|layers[b])))
        for k,v in {'M07_global_emergency_rule':0,'M08_reference_free_classes':1 if self.reference=='P1' else 3,'M09_reference_P2':int(self.reference=='P2'),'M10_terminal_delay_weight':1,'M12_tick':1,'M13_one_representative':1,'M14_seed':20261026,'M14_hint_previous_policy':1,'M15_strict_optimum':0}.items():put(d,k,v)
        return d

    def graph_node(self,u):
        if u in self.node_cache:return self.node_cache[u]
        n=self.nodes[u];d={};inc=self.inc[u];out=self.out[u];neigh=inc+out
        for k,v in {'G05_in':len(inc),'G05_out':len(out),'G06_in_cross':sum(e['cross_train'] for e in inc),'G06_out_cross':sum(e['cross_train'] for e in out),'G08_in_active':sum(e['active'] for e in inc),'G08_out_active':sum(e['active'] for e in out),'G09_active_cross':sum(e['active'] and e['cross_train'] for e in neigh),'G09_active_cross_fraction':ratio(sum(e['active'] and e['cross_train'] for e in neigh),len(neigh)),'G11_near1':sum(e['slack']<=1 for e in neigh),'G11_near2':sum(e['slack']<=2 for e in neigh),'G16_degree_balance':len(out)-len(inc),'G16_pressure_balance':sum(e['pressure'] for e in out)-sum(e['pressure'] for e in inc),'G17_type_entropy':entropy(Counter(e['kind'] for e in neigh).values()),'G18_source_fraction':ratio(sum(e['source']=='MASTER' for e in inc),len(inc))}.items():put(d,k,v)
        for side,es in [('in',inc),('out',out)]:
            for typ in ['dwell','run','entry_headway','exit_headway','station_track','emergency']:put(d,'G07_'+side+'_'+typ,sum(e['kind']==typ for e in es))
            stats(d,'G10_'+side+'_slack',[e['slack'] for e in es]);stats(d,'G12_'+side+'_support',[e['pressure'] for e in es])
        for code,es,endpoint in [('G13',inc,'source'),('G14',out,'target')]:
            bytrain=defaultdict(list)
            for e in es:
                v=e[endpoint]
                if v!='MASTER' and self.nodes[v]['train']!=n['train']:bytrain[self.nodes[v]['train']].append(self.nodes[v]['delay'])
            stats(d,code+'_neighbor_delay',[max(z) for z in bytrain.values()])
        vals=[self.nodes[v]['delay'] for v in set(self.active.predecessors(u))|set(self.active.successors(u)) if v!='MASTER']
        put(d,'G15_delay_range',max(vals)-min(vals) if vals else np.nan);put(d,'G15_delay_variance',np.var(vals) if vals else np.nan)
        put(d,'G15_delay_CV',ratio(np.std(vals),np.mean(vals)) if vals else np.nan)
        anc=self.anc[u]-{'MASTER'};desc=self.desc[u]-{'MASTER'}
        put(d,'H01_source_edges',self.source_dist.get(u,np.nan));put(d,'H01_source_minlag',self.lagdist.get(u,np.nan));put(d,'H02_longest_path',self.longest[u]);put(d,'H03_time_from_release',n['time']-self.E)
        for prefix,vs in [('up',anc),('down',desc)]:
            for code,value in [('H04' if prefix=='up' else 'H05',len(vs)),('H06' if prefix=='up' else 'H07',len({self.nodes[v]['train'] for v in vs}))]:put(d,code+'_'+prefix,value)
            put(d,'H08_'+prefix+'_stations',len({self.nodes[v]['station'] for v in vs}));put(d,'H08_'+prefix+'_branches',len({branch(self.nodes[v]['station']) for v in vs}))
            put(d,'H09_'+prefix+'_event_delay',sum(self.nodes[v]['delay'] for v in vs))
            tt=defaultdict(list)
            for v in vs:tt[self.nodes[v]['train']].append(self.nodes[v]['delay'])
            put(d,'H09_'+prefix+'_train_maxdelay_sum',sum(max(z) for z in tt.values()))
        put(d,'H10_betweenness',self.between[u]);put(d,'H11_out_harmonic',sum(1/z for v,z in self.dist[u].items() if z)/max(1,len(self.nodes)-1))
        put(d,'H11_in_harmonic',sum(1/dd[u] for v,dd in self.dist.items() if v!=u and u in dd)/max(1,len(self.nodes)-1))
        put(d,'H12_pagerank',self.pr[u]);comp=self.comp[u]-{'MASTER'};put(d,'H13_component_events',len(comp));put(d,'H13_component_trains',len({self.nodes[v]['train'] for v in comp}));put(d,'H14_SCC',1)
        put(d,'H15_longest_crossings',self.crosslong[u]);put(d,'H15_longest_chain',self.longest[u])
        predecessors=set(self.active.predecessors(u));successors=set(self.active.successors(u))
        put(d,'H16_fanin_pairs',len(predecessors)*(len(predecessors)-1)//2);put(d,'H16_fanout_pairs',len(successors)*(len(successors)-1)//2)
        put(d,'H16_diamonds_at_source',sum(len(set(self.active.successors(a))&set(self.active.successors(b))) for a,b in itertools.combinations(successors,2)))
        put(d,'H23_undirected_clustering',self.clust.get(u,0));put(d,'H24_core',self.core.get(u,0))
        nbr=[v for v in predecessors|successors if v!='MASTER']
        put(d,'H25_local_same_type_fraction',np.mean([self.trains[self.nodes[v]['train']]['speed_class']==self.trains[n['train']]['speed_class'] for v in nbr]) if nbr else np.nan)
        put(d,'H25_local_delay_absdiff',np.mean([abs(self.nodes[v]['delay']-n['delay']) for v in nbr]) if nbr else np.nan)
        cid=self.community.get(u,-1);c=self.communities[cid] if cid>=0 else {u}
        put(d,'B08_community_events',len(c));put(d,'B08_cross_community_edges',sum(self.community.get(v,-1)!=cid for v in predecessors|successors))
        self.node_cache[u]=d;return d

    def dynamic(self,station,direction,anchor,resource,d):
        cap=len(self.raw['stations'][station]['direction_tracks'][direction]);intervals=self.vis[station,direction]
        opp='R' if direction=='F' else 'F'
        def window(center,w,ints):
            lo,hi=center-w/2,center+w/2;cuts=sorted({lo,hi}|{max(lo,min(hi,z)) for a,b,*_ in ints for z in [a,b]})
            pieces=[(a,b,sum(x<=a<y for x,y,*_ in ints)) for a,b in zip(cuts,cuts[1:])]
            avg=sum((b-a)*c for a,b,c in pieces)/w;peak=max([c for a,b,c in pieces],default=0);sat=sum(b-a for a,b,c in pieces if c>=cap)/w
            longest=cur=0;free=[];run=0
            for a,b,c in pieces:
                cur=cur+b-a if c>=cap else 0;longest=max(longest,cur)
                if c<cap:run+=b-a
                elif run:free.append(run);run=0
            if run:free.append(run)
            return avg,peak,sat,longest,free
        for w in [5,10,20]:
            avg,peak,sat,longest,free=window(anchor,w,intervals);tag=f'_w{w}'
            for k,v in {'R01_mean_occupancy':avg,'R02_peak_occupancy':peak,'R03_min_free':cap-peak,'R04_utilization':avg/cap,'R05_saturated_fraction':sat,'R06_longest_saturation':longest,'R12_demand_capacity':avg/cap,'R15_free_window_count':len(free),'R15_free_window_total':sum(free),'R15_free_window_max':max(free,default=0)}.items():put(d,k+tag,v)
            lo,hi=anchor-w/2,anchor+w/2;waiting=[b-1-a for a,b,t,n in intervals if a<hi and b>lo]
            stats(d,'R10_waiting_duration'+tag,waiting)
            arr=sum(lo<=a<hi for a,b,*_ in intervals);dep=sum(lo<=b-1<hi for a,b,*_ in intervals)
            put(d,'R11_arrivals_minus_departures'+tag,arr-dep)
            nearby=[(t,n) for a,b,t,n in intervals if a<hi and b>lo];put(d,'R14_window_competitors'+tag,len(set(t for t,n in nearby)))
            put(d,'H17_competitor_trains'+tag,len(set(t for t,n in nearby)));put(d,'H18_competition_set_memberships'+tag,len(nearby));put(d,'H19_peak_demand_capacity'+tag,peak/cap)
            put(d,'R18_opposite_utilization'+tag,window(anchor,w,self.vis[station,opp])[0]/cap)
            prev=window(anchor-w,w,intervals)[0];future=window(anchor+w,w,intervals)[0]
            put(d,'R17_occupancy_rate'+tag,(future-prev)/(2*w));put(d,'B13_adjacent_occupancy_diff'+tag,future-prev)
            plannedarr=[v['plan_arrival'] for t in self.trains.values() for n,v in enumerate(t['visits'][1:-1],1) if v['station']==station and t['runs'][0]['resource'].endswith('::'+direction)]
            put(d,'R16_plan_arrival_release_overlap'+tag,sum(lo<=a<hi and any(x<=a<y for x,y,*_ in intervals) for a in plannedarr))
            if resource:
                rr=self.runs[resource];put(d,'R07_entry_load'+tag,sum(lo<=a<hi for a,b,*_ in rr)*2/w);put(d,'R08_exit_load'+tag,sum(lo<=b<hi for a,b,*_ in rr)*2/w)
                pp=self.planruns[resource];put(d,'T12_plan_flow'+tag,sum(lo<=a<hi for a,b,*_ in pp)/w)
                types=[self.trains[t]['speed_class']=='fast' for a,b,t,n in pp if lo<=a<hi]
                put(d,'T13_fast_fraction'+tag,np.mean(types) if types else np.nan);put(d,'T13_alternation'+tag,np.mean(np.diff(types)!=0) if len(types)>1 else np.nan)
            # Local spatiotemporal region = this station, same direction, symmetric window.
            vn={u for u,n in self.nodes.items() if n['station']==station and n['direction']==direction and lo<=n['time']<hi}
            inside=[e for e in self.net['edges'] if e['source'] in vn and e['target'] in vn]
            entering=[e for e in self.net['edges'] if e['target'] in vn and e['source'] not in vn];leaving=[e for e in self.net['edges'] if e['source'] in vn and e['target'] not in vn]
            put(d,'B01_region_events'+tag,len(vn));put(d,'B01_region_trains'+tag,len({self.nodes[u]['train'] for u in vn}));put(d,'B02_edges'+tag,len(inside));put(d,'B02_density'+tag,ratio(len(inside),len(vn)*(len(vn)-1)))
            put(d,'B03_active_density'+tag,ratio(sum(e['active'] for e in inside),len(vn)*(len(vn)-1)));put(d,'B04_entering'+tag,len(entering));put(d,'B04_leaving'+tag,len(leaving))
            stats(d,'B05_region_delay'+tag,[self.nodes[u]['delay'] for u in vn]);put(d,'B06_delay_variance'+tag,np.var([self.nodes[u]['delay'] for u in vn]) if vn else np.nan)
            put(d,'B07_saturated_fraction'+tag,sat);put(d,'B09_cross_boundary_pressure_balance'+tag,sum(e['pressure'] for e in leaving)-sum(e['pressure'] for e in entering))

    def visit(self,tid,n):
        if (tid,n) in self.visit_cache:return dict(self.visit_cache[tid,n])
        t=self.trains[tid];v=t['visits'][n];s=self.sch[tid][n];pv=self.prep[tid]['visits'][n];N=len(t['visits']);direction=t['runs'][0]['resource'].split('::')[-1];station=v['station'];d=dict(self.base_global)
        def add(k,x):put(d,k,x)
        anchor=s['departure'];pr=v['plan_departure']
        closure=next((k for k,r in enumerate(t['runs']) if self.raw['running_resources'][r['resource']]['physical_section']==self.closed),None)
        e=self.em.get(tid)
        current=0 if t['visits'][0]['plan_departure']>=self.B else next((k for k,vv in enumerate(t['visits']) if vv['plan_departure']>=self.B),N-1)
        em_n=e['n'] if e else current
        for k,x in {'D09_base_from_start':anchor-self.B,'D09_plan_from_start':pr-self.B,'D10_base_from_release':anchor-self.E,'D10_plan_from_release':pr-self.E,
            'X01_crosses_closed':int(closure is not None),'X13_reaches_stop_before_reopen':int(e is not None and (e.get('arrival') or self.B)<=self.E),
            'X17_remaining_route_fraction':(N-1-n)/(N-1),'X18_origin_emergency':int(e is not None and e['n']==0),
            'X19_direct_stop_type':0 if not e else 1 if e['n']==0 else 2 if e['kind']=='finish_committed_run_then_hold' else 3}.items():add(k,x)
        for key in ['X02_passed_at_onset','X03_plan_clear_before','X04_plan_closure_overlap','X05_plan_overlap_fraction','X06_plan_entry_from_onset','X07_plan_exit_from_onset',
            'X08_intervals_to_entry','X09_minrun_to_entry','X10_signed_distance_to_closure','X11_at_closed_entry','X12_can_clear_without_emergency',
            'X14_earliest_entry_no_other_conflicts','X15_earliest_entry_minus_release','X16_closure_only_wait']:add(key,np.nan)
        if closure is None:
            add('X04_plan_closure_overlap',0);add('X05_plan_overlap_fraction',0);add('X11_at_closed_entry',0);add('X16_closure_only_wait',0)
        else:
            ent=t['visits'][closure]['plan_departure'];ext=t['visits'][closure+1]['plan_arrival'];overlap=max(0,min(ext,self.E)-max(ent,self.B))
            for k,x in {'X02_passed_at_onset':int(ext<=self.B),'X03_plan_clear_before':int(ext<=self.B),'X04_plan_closure_overlap':overlap,'X05_plan_overlap_fraction':overlap/(ext-ent),
                'X06_plan_entry_from_onset':ent-self.B,'X07_plan_exit_from_onset':ext-self.B,'X08_intervals_to_entry':max(0,closure-current),
                'X09_minrun_to_entry':sum(r['minimum'] for r in t['runs'][min(current,closure):closure]),'X10_signed_distance_to_closure':closure-n,
                'X11_at_closed_entry':int(n==closure),'X12_can_clear_without_emergency':int(ext<=self.B)}.items():add(k,x)
            cv=t['visits'][current];minimum=cv.get('minimum_dwell',1) if cv.get('stop')=='required' and 0<current<N-1 else 0
            earliest=max(self.B,cv['plan_arrival']+minimum,cv['plan_departure'] if cv.get('commercial') or current==0 else self.B)
            for k0 in range(current,closure):
                nv=t['visits'][k0+1];arrival=max(earliest+t['runs'][k0]['minimum'],nv['plan_arrival'])
                minimum=nv.get('minimum_dwell',1) if nv.get('stop')=='required' else 0
                earliest=max(arrival+minimum,nv['plan_departure'] if nv.get('commercial') else arrival)
            if closure<current:earliest=ent
            add('X14_earliest_entry_no_other_conflicts',earliest);add('X15_earliest_entry_minus_release',earliest-self.E)
            add('X16_closure_only_wait',max(0,self.E-earliest) if ext>self.B else 0)
        state=0 if t['visits'][0]['plan_departure']>=self.B else 3 if t['visits'][-1]['plan_arrival']<=self.B else 1 if any(vv['plan_arrival']<=self.B<vv['plan_departure'] for vv in t['visits']) else 2
        for state_id in range(4):add('E01_state_'+str(state_id),int(state==state_id))
        add('E02_executed_fraction',sum(vv['plan_departure']<self.B for vv in t['visits'])/N);add('E03_current_route_index',min(em_n,N-1));add('E04_committed_remaining',sum(max(0,t['visits'][k+1]['plan_arrival']-self.B) for k in range(N-1) if t['visits'][k]['plan_departure']<self.B<t['visits'][k+1]['plan_arrival']))
        add('E05_first_stop_index',em_n if e else np.nan);add('E06_time_to_stop',max(0,e['arrival']-self.B) if e and e['arrival'] is not None else 0 if e else np.nan);add('E07_emergency_arrival',e['arrival'] if e and e['arrival'] is not None else self.B if e else np.nan)
        add('E08_min_emergency_hold',max(0,self.E-max(self.B,e['arrival'] or self.B)) if e else 0);add('E09_already_stopped',int(state==1));add('E10_emergency_pool_capacity',len(self.raw['stations'][e['station']]['direction_tracks'][direction]) if e else np.nan)
        q=[(self.sch[tt][ee['n']]['departure'],tt) for tt,ee in self.em.items() if e and ee['station']==e['station'] and tt.startswith(direction)]
        add('E11_same_stop_count',len(q));add('E14_release_queue_rank',next((i for i,z in enumerate(sorted(q)) if z[1]==tid),np.nan));add('E15_queue_fast_fraction',np.mean([self.trains[tt]['speed_class']=='fast' for _,tt in q]) if q else np.nan)
        occupied=sum(a<=self.B<b for a,b,*_ in self.vis[station,direction]);add('E23_initial_station_occupied',occupied)
        for k,x in {'T01_plan_arrival':v['plan_arrival'],'T02_plan_departure':pr,'T03_plan_dwell':pr-v['plan_arrival'],'T04_plan_run_next':t['runs'][n]['original_duration'] if n<N-1 else np.nan,'T05_plan_stop':v['plan_stop'],'T06_plan_track_index':int(v['plan_track'][1:]) if v.get('plan_track') else np.nan,'T07_plan_run_buffer':t['runs'][n]['original_duration']-t['runs'][n]['minimum'] if n<N-1 else np.nan,'T08_commercial_dwell_buffer':pr-v['plan_arrival']-v.get('minimum_dwell',1) if v.get('commercial') else np.nan,'T14_origin_wave40':t['visits'][0]['plan_departure']//40,'T15_fast':int(t['speed_class']=='fast'),'T16_objective_weight':t['weight'],'T17_plan_total':t['visits'][-1]['plan_arrival']-t['visits'][0]['plan_departure'],'T18_remaining_plan':t['visits'][-1]['plan_arrival']-pr,'T19_remaining_min':sum(r['minimum'] for r in t['runs'][n:])+sum(vv.get('minimum_dwell',1) for vv in t['visits'][n+1:-1] if vv['stop']=='required'),'T20_remaining_commercial':sum(vv['commercial'] for vv in t['visits'][n+1:]),'T21_prev_commercial_distance':n-max([k for k,vv in enumerate(t['visits'][:n]) if vv['commercial']],default=0),'T21_next_commercial_distance':min([k for k,vv in enumerate(t['visits'][n+1:],n+1) if vv['commercial']],default=N-1)-n,'T22_route_edges':N-1,'T22_remaining_edges':N-1-n,'T23_remaining_buffer':0,'T24_plan_terminal':t['visits'][-1]['plan_arrival'],'T24_terminal_from_release':t['visits'][-1]['plan_arrival']-self.E}.items():add(k,x)
        peers=[(self.trains[tt]['visits'][nn]['plan_arrival'],self.trains[tt]['visits'][nn]['plan_departure'],tt) for a,b,tt,nn in self.vis[station,direction]]
        ar=sorted(peers);dr=sorted(peers,key=lambda z:(z[1],z[2]));add('T26_station_rank_change',next((i for i,z in enumerate(dr) if z[2]==tid),0)-next((i for i,z in enumerate(ar) if z[2]==tid),0))
        add('T25_original_reorder',int(d['T26_station_rank_change']!=0))
        predtrack=[self.trains[tt]['visits'][nn]['plan_departure']+1 for a,b,tt,nn in self.vis[station,direction] if tt!=tid and self.trains[tt]['visits'][nn].get('plan_track')==v.get('plan_track') and self.trains[tt]['visits'][nn]['plan_arrival']<v['plan_arrival']]
        add('T11_plan_track_buffer',v['plan_arrival']-max(predtrack) if predtrack else np.nan)
        cap=len(self.raw['stations'][station]['direction_tracks'][direction]);deg=self.fac.degree(station)
        for k,x in {'F01_hub':int(station in ['H1','H2']),'F01_trunk':int(station in ['H1','C','H2']),'F02_undirected_degree':deg,'F03_branch_count':max(0,deg-1),'F04_betweenness':self.fbet[station],'F05_articulation':int(station in self.cut),'F06_direction_pool':cap,'F07_compatible_tracks':len(v.get('tracks',[])) if n not in [0,N-1] else 0,'F08_tracks_per_direction':1,'F09_entry_headway':2,'F10_exit_headway':2,'F11_clearance':1,'F12_min_dwell':v.get('minimum_dwell',1),'F13_min_run':t['runs'][n]['minimum'] if n<N-1 else np.nan,'F14_optional':int(v['stop']=='optional' and 'fixed_stop' not in v),'F14_required':int(v['stop']=='required'),'F15_parallel_capacity':cap,'F16_next_capacity_delta':len(self.raw['stations'][t['visits'][n+1]['station']]['direction_tracks'][direction])-cap if n<N-1 else np.nan,'F17_hub_distance':min(nx.shortest_path_length(self.fac,station,h) for h in ['H1','H2']),'F18_next_stop_distance':min([k-n for k,vv in enumerate(t['visits']) if k>n and vv.get('plan_stop')],default=np.nan),'F19_external_origin_wait':1,'F22_direction_coupling':0,'F23_alternative_routes':0}.items():add(k,x)
        add('G01_arrival_delay',s['arrival']-v['plan_arrival']);add('G02_departure_delay',s['departure']-pr);add('G03_dwell_delay_increment',d['G02_departure_delay']-d['G01_arrival_delay']);add('G04_dwell_increment',s['departure']-s['arrival']-(pr-v['plan_arrival']))
        a=self.graph_node(eid(tid,n,'arrival'));b=self.graph_node(eid(tid,n,'departure'))
        for k,x in a.items():put(d,k+'_arrival',x)
        for k,x in b.items():put(d,k+'_departure',x)
        for k in ['G05_in','G05_out','G08_in_active','G08_out_active','H04_up','H05_down','H10_betweenness']:put(d,'G19_'+k+'_end_difference',b[k]-a[k])
        resource=t['runs'][n]['resource'] if n<N-1 else t['runs'][-1]['resource']
        self.dynamic(station,direction,anchor,resource,d)
        for tag,key in [('T09','entry'),('T10','exit')]:
            rr=self.planruns[resource];ii=next((i for i,z in enumerate(rr) if z[2]==tid),None);phase=0 if key=='entry' else 1
            add(tag+'_previous_buffer',rr[ii][phase]-rr[ii-1][phase]-2 if ii is not None and ii>0 else np.nan)
        rr=self.runs[resource];ii=next((i for i,z in enumerate(rr) if z[2]==tid),None)
        for phase in [0,1]:add('R09_'+str(phase)+'_previous_buffer',rr[ii][phase]-rr[ii-1][phase]-2 if ii is not None and ii>0 else np.nan)
        prevs=self.vis[t['visits'][max(0,n-1)]['station'],direction];nexts=self.vis[t['visits'][min(N-1,n+1)]['station'],direction]
        add('R13_upstream_occupancy',sum(a<=anchor<b for a,b,*_ in prevs));add('R13_downstream_occupancy',sum(a<=anchor<b for a,b,*_ in nexts));add('R13_occupancy_difference',d['R13_downstream_occupancy']-d['R13_upstream_occupancy'])
        # Candidate coupling counts are structural, before observing changes.
        support={eid(tid,n,'arrival'),eid(tid,n,'departure')};otherS=[eid(tid,x['n'],'departure') for x in self.units['S'].values() if x['train']==tid]
        add('H20_nearest_optional_stop_index',min([abs(n-x['n']) for x in self.units['S'].values() if x['train']==tid and x['n']!=n],default=np.nan))
        add('H21_same_visit_classes',int(f'{tid}:{n}' in self.units['S'])+int(f'D:{tid}:{n}' in self.units['T'])+int(f'R:{tid}:{n}' in self.units['T'])+sum(x['left_train']==tid and x['left_n']==n or x['right_train']==tid and x['right_n']==n for x in self.units['O'].values()))
        da=self.desc[eid(tid,n,'arrival')];db=self.desc[eid(tid,n,'departure')];add('H22_arrival_departure_downstream_Jaccard',ratio(len(da&db),len(da|db)))
        for typ in ['S','T','O','K']:add('M06_remaining_'+typ,sum((x.get('train')==tid and x.get('n',-1)>=n) or (x.get('left_train')==tid and x.get('left_n',-1)>=n) or (x.get('right_train')==tid and x.get('right_n',-1)>=n) for x in self.units[typ].values()))
        add('M02_fixed_arrival',int('fixed_arrival' in v));add('M02_fixed_departure',int('fixed_departure' in v));add('M11_horizon_from_base',self.raw['horizon']-anchor);add('M11_horizon_from_release',self.raw['horizon']-self.E)
        self.visit_cache[tid,n]=dict(d);return d

    def closure_rank_difference(self,i,j,direction):
        order={tid:k for k,(_,_,tid,_) in enumerate(self.planruns[self.closed+'::'+direction])}
        return order[j]-order[i] if i in order and j in order else np.nan

    def rows(self):
        rows=[]
        for typ,units in self.units.items():
            if typ=='K':continue
            for key,u in units.items():
                if typ=='O':
                    i,ni,j,nj=u['left_train'],u['left_n'],u['right_train'],u['right_n']
                    # Canonical order: original planned front then rear.
                    if self.trains[i]['visits'][ni]['plan_departure']>self.trains[j]['visits'][nj]['plan_departure']:i,ni,j,nj=j,nj,i,ni
                    left=self.visit(i,ni);right=self.visit(j,nj);d={}
                    for k,a in left.items():
                        b=right[k]
                        if k in self.base_global:put(d,k,a)
                        else:
                            for suf,x in [('front',a),('rear',b),('difference',b-a if a is not None and b is not None else np.nan),('absdifference',abs(b-a) if a is not None and b is not None else np.nan)]:put(d,k+'_'+suf,x)
                    ti,tj=self.trains[i],self.trains[j];vi,vj=ti['visits'][ni],tj['visits'][nj];resource=u['resource'];pi=self.planruns[resource];ranks={z[2]:r for r,z in enumerate(pi)}
                    common=set(r['resource'] for r in ti['runs'])&set(r['resource'] for r in tj['runs']);commonstations=set(v['station'] for v in ti['visits'])&set(v['station'] for v in tj['visits'])
                    aa=eid(i,ni,'departure');bb=eid(j,nj,'departure');da=self.desc[aa];db=self.desc[bb]
                    pair={
                        'P01_front_first':1,'P02_plan_rank_distance':ranks[j]-ranks[i],
                        'P03_plan_entry_gap':vj['plan_departure']-vi['plan_departure'],'P04_plan_exit_gap':tj['visits'][nj+1]['plan_arrival']-ti['visits'][ni+1]['plan_arrival'],
                        'P05_plan_run_difference':tj['runs'][nj]['original_duration']-ti['runs'][ni]['original_duration'],
                        'P07_remaining_plan_difference':right['T18_remaining_plan']-left['T18_remaining_plan'],'P07_remaining_min_difference':right['T19_remaining_min']-left['T19_remaining_min'],
                        'P08_terminal_difference':tj['visits'][-1]['plan_arrival']-ti['visits'][-1]['plan_arrival'],
                        'P09_stop_disagreements':sum(a['plan_stop']!=b['plan_stop'] for a in ti['visits'] for b in tj['visits'] if a['station']==b['station']),
                        'P10_common_sections':len(common),'P10_common_min_time':sum(r['minimum'] for r in ti['runs'] if r['resource'] in common),
                        'P11_common_station_pools':len(commonstations),'P12_merge_distance':min([k for k,v in enumerate(ti['visits']) if v['station'] in commonstations])-ni,
                        'P12_split_remaining':max([k for k,v in enumerate(ti['visits']) if v['station'] in commonstations])-ni,
                        'P13_potential_catchup':max(0,(ti['runs'][ni]['minimum']-tj['runs'][nj]['minimum'])*len(common)-(vj['plan_departure']-vi['plan_departure'])),
                        'P14_free_order_positions':sum({x['left_train'],x['right_train']}=={i,j} for x in self.units['O'].values()),
                        'P15_parallel_common_stations':sum(len(self.raw['stations'][s]['direction_tracks'][resource.split('::')[-1]])>=2 for s in commonstations),
                        'P16_base_departure_delay_difference':right['G02_departure_delay']-left['G02_departure_delay'],
                        'P17_reference_same_track':int(self.sch[i][ni]['track']==self.sch[j][nj]['track'] and self.sch[i][ni]['track'] is not None),
                        'P18_reference_interval_overlap':max(0,min(self.sch[i][ni+1]['arrival'],self.sch[j][nj+1]['arrival'])-max(self.sch[i][ni]['departure'],self.sch[j][nj]['departure'])),
                        'X20_same_exposure':int(left['X19_direct_stop_type']==right['X19_direct_stop_type']),
                        'X21_entry_relative_release_difference':right['X15_earliest_entry_minus_release']-left['X15_earliest_entry_minus_release'],
                        'X22_closure_wait_difference':right['X16_closure_only_wait']-left['X16_closure_only_wait'],
                        'X23_closure_plan_rank_difference':self.closure_rank_difference(i,j,resource.split('::')[-1]),
                        'H20_pair_graph_distance':self.dist[aa].get(bb,np.nan),'H22_pair_downstream_Jaccard':ratio(len(da&db),len(da|db)),
                        'M03_original_order_choices':2,'M04_order_choices':2,'M05_base_order':int(self.sch[i][ni]['departure']<self.sch[j][nj]['departure'])}
                    for combo in ['fast_fast','fast_slow','slow_fast','slow_slow']:pair['P06_'+combo]=int(combo==ti['speed_class']+'_'+tj['speed_class'])
                    for k,x in pair.items():put(d,k,x)
                    unit=dict(train=i,visit=ni,other_train=j,other_visit=nj,resource=resource,station=vi['station'],direction=resource.split('::')[-1],kind='O')
                else:
                    tid,n=u['train'],u['n'];d=self.visit(tid,n);v=self.trains[tid]['visits'][n];pv=self.prep[tid]['visits'][n]
                    kind='S' if typ=='S' else 'Td' if u['kind']=='dwell' else 'Tr'
                    if kind=='S':lo,hi,base=0,1,self.sch[tid][n]['stop'];rawlo,rawhi=0,1
                    elif kind=='Td':
                        lo,hi=u['bounds'];base=self.sch[tid][n]['departure']-self.sch[tid][n]['arrival'];rawlo,rawhi=0,self.raw['horizon']
                    else:
                        lo,hi=u['bounds'];base=self.sch[tid][n+1]['arrival']-self.sch[tid][n]['departure'];r=self.trains[tid]['runs'][n];rawlo,rawhi=r['minimum'],r['maximum']
                        end=self.visit(tid,n+1)
                        for k in ['G01_arrival_delay','G02_departure_delay','H10_betweenness_departure','R01_mean_occupancy_w10','F06_direction_pool','T18_remaining_plan']:
                            put(d,k+'_next_endpoint',end[k]);put(d,'G19_run_'+k+'_difference',end[k]-d[k])
                        put(d,'G03_run_delay_increment',self.sch[tid][n+1]['arrival']-self.trains[tid]['visits'][n+1]['plan_arrival']-(self.sch[tid][n]['departure']-v['plan_departure']))
                        put(d,'G04_run_increment',base-r['original_duration'])
                    for k,x in {'M03_raw_domain_width':rawhi-rawlo,'M04_domain_lower':lo,'M04_domain_upper':hi,'M04_domain_width':hi-lo,'M05_compressible':base-lo,'M05_extendable':hi-base}.items():put(d,k,x)
                    unit=dict(train=tid,visit=n,other_train='',other_visit=-1,resource=self.trains[tid]['runs'][min(n,len(self.trains[tid]['runs'])-1)]['resource'],station=v['station'],direction=self.trains[tid]['runs'][0]['resource'].split('::')[-1],kind=kind)
                for kind0 in ['S','Td','Tr','O']:put(d,'M01_'+kind0,int(unit['kind']==kind0))
                rows.append(dict(case=self.sc['case'],start=self.B,duration=self.L,reference=self.reference,unit_id=key,**unit,**d))
        return pd.DataFrame(rows)

def main():
    start=time.perf_counter();out=ROOT/'features';out.mkdir(exist_ok=True);done=[]
    for f in sorted((ROOT/'cases').iterdir()):
        for ref in ['P1','P2']:
            if not (f/f'{ref}_network.json').exists():continue
            target=out/f'{f.name}_{ref}.csv.gz';manifest=out/f'{f.name}_{ref}_manifest.json'
            if target.exists() and manifest.exists():REG.update(read(manifest));done.append(target.name);continue
            c=Case(f,ref);df=c.rows();df.to_csv(target,index=False,compression='gzip');save(manifest,REG);done.append(target.name)
            print(f.name,ref,len(df),len(df.columns),round(time.perf_counter()-start,1),flush=True)
    save(ROOT/'feature_manifest.json',REG)
    coverage=[]
    for item in CAT:
        columns=[k for k,v in REG.items() if v['catalog_id']==item['id']]
        state='computed' if columns else 'unavailable_in_current_model' if item['current_status']=='X' else 'pair_only_or_not_yet_materialized'
        coverage.append(dict(**item,execution_status=state,columns=columns,reason=item['definition'] if not columns else 'See feature_manifest.json; constant and missingness checks are performed per response.'))
    save(ROOT/'catalog_coverage.json',coverage);save(ROOT/'feature_progress.json',dict(files=done,seconds=time.perf_counter()-start))
if __name__=='__main__':main()
