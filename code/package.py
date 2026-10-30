"""Package evidence without claiming the manuscript has already been updated."""
import datetime,os,shutil,zipfile
from common import *
def main():
    assert read(ROOT/'final_validation.json')['passed']
    out=ROOT.parent/(ROOT.name+'_evidence');out.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'reports',out/'reports',dirs_exist_ok=True)
    for name in ['REPORT.md','README.md','protocol.json','completion.json','final_validation.json','case_manifest.json','folds.json','phase_ledger.json','training_costs.json','source_lock.json','execution_lock.json','model_lock.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,out/name)
    files=[]
    for parent,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in ['__pycache__','.git']]
        for name in names:
            p=Path(parent)/name
            if p.suffix in ['.tmp','.lock','.pyc'] or name=='delivery_complete.json':continue
            files.append(p)
    evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    save(out/'evidence_manifest.json',dict(files=evidence))
    archive=out/'experiment_source_results_evidence.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        z.write(out/'evidence_manifest.json','evidence_manifest.json')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    save(ROOT/'delivery_complete.json',dict(experiment_package_complete=True,overall_complete=False,paper_update_pending=True,
        archive=str(archive),sha256=sha(archive),files=len(files),at=datetime.datetime.now().astimezone().isoformat()))
    print('EVIDENCE_PACKAGE_READY_PENDING_PAPER',str(archive),flush=True)
if __name__=='__main__':main()

