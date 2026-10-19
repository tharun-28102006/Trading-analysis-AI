"""Safe terminal orchestration for Nivesh data refresh and eligible research."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from .acquisition import list_sources, run_acquisition
from .config import DATA_DIR
from .db import connect, init_db


def _report_files(report: dict) -> dict:
    folder=DATA_DIR/'reports'; folder.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stem=f"run-all-{stamp}-{report['run_id']}"
    json_path=folder/f'{stem}.json'; text_path=folder/f'{stem}.txt'
    temp=json_path.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(temp,json_path)
    lines=[f"Nivesh run-all: {report['status']}",f"Run ID: {report['run_id']}",f"Started: {report['started_at']}",f"Finished: {report['finished_at']}"]
    for stage in report['stages']:
        detail=stage.get('reason') or stage.get('message') or stage.get('status','')
        lines.append(f"- {stage['name']}: {stage['status']} — {detail}")
    text_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'json':str(json_path),'text':str(text_path)}


def run_all(reconcile_history=False) -> dict:
    """Run only dependencies supported by the current data and permissions."""
    started=datetime.now(timezone.utc).isoformat(); run_id=started.replace(':','').replace('-','')
    stages=[]
    with connect() as conn:
        integrity=conn.execute('PRAGMA integrity_check').fetchone()[0]
    stages.append({'name':'database_integrity','status':'succeeded' if integrity=='ok' else 'failed','message':integrity})
    if integrity!='ok':
        report={'run_id':run_id,'status':'failed','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'stages':stages}
        report['reports']=_report_files(report); return report

    sources=list_sources()
    configured=[s['source_key'] for s in sources if s.get('permission_status') in {'authorized_by_local_user_attestation','authorized_by_provider_dataset_license'}]
    stages.append({'name':'source_discovery','status':'succeeded','message':f"{len(sources)} catalog entries; configured eligible sources: {', '.join(configured) or 'none'}"})
    from . import market
    started_refresh=market.start_refresh('run-all',reconcile_history)
    if started_refresh.get('status')=='running':
        while True:
            acquisition=market.refresh_job(started_refresh['run_id'])
            if acquisition.get('status')!='running': break
            time.sleep(.5)
    else:
        acquisition=started_refresh
    stages.append({'name':'market_refresh','status':acquisition.get('status','unknown'),'message':acquisition.get('error') or f"{acquisition.get('rows_written',0)} validated observations published",'result':acquisition})

    from . import models
    from .features import training_eligibility
    with connect() as conn:
        symbols=[r[0] for r in conn.execute("SELECT DISTINCT symbol FROM published_prices WHERE quality='USER_CONFIRMED_UNVERIFIED' AND symbol<>'NIFTY50' ORDER BY symbol")]
        benchmark_rows=conn.execute("SELECT COUNT(*) FROM published_prices WHERE symbol='NIFTY50' AND quality='USER_CONFIRMED_UNVERIFIED'").fetchone()[0]
        model_versions={r['model_key']:r['id'] for r in conn.execute("SELECT id,model_key FROM model_versions WHERE status='eligible' ORDER BY id DESC")}
        membership_rows=conn.execute("SELECT COUNT(*) FROM published_constituent_membership WHERE quality='USER_CONFIRMED_UNVERIFIED'").fetchone()[0]
        pooled_exists=bool(conn.execute("SELECT 1 FROM model_versions WHERE model_key='pooled' AND status='eligible'").fetchone())
        aggregator_exists=bool(conn.execute("SELECT 1 FROM model_versions WHERE model_key='master_aggregator' AND status='eligible'").fetchone())

    training=[]; skipped=[]; held=[]; failed=[]
    if not benchmark_rows:
        skipped.append({'scope':'company_models','reason':'Genuine NIFTY50 benchmark history is required for aligned five-session excess-return targets.'})
    else:
        auto_retrain=os.getenv('NIVESH_AUTO_RETRAIN','0').strip().lower() in {'1','true','yes'}
        for symbol in symbols:
            eligibility=training_eligibility(symbol)
            if not eligibility['eligible']:
                skipped.append({'symbol':symbol,'reason':eligibility['reason']}); continue
            existing=[key for key in model_versions if key==f'company:{symbol}']
            if existing and not auto_retrain:
                held.append({'symbol':symbol,'reason':'An eligible model already exists; set NIVESH_AUTO_RETRAIN=1 to replace it during run-all.'}); continue
            try: training.append(models.train_company(symbol))
            except Exception as exc: failed.append({'symbol':symbol,'reason':str(exc)[:400]})
    stages.append({'name':'company_models','status':'succeeded' if training and not failed else ('partial_success' if training else 'blocked' if skipped else 'skipped'),'trained':training,'skipped':skipped,'existing_models_kept':held,'failed':failed})

    if membership_rows:
        if pooled_exists and os.getenv('NIVESH_AUTO_RETRAIN','0').lower() not in {'1','true','yes'}:
            stages.append({'name':'pooled_model','status':'skipped','reason':'Eligible pooled model already exists; automatic retraining is disabled.'})
        else:
            try: stages.append({'name':'pooled_model','status':'succeeded','result':models.train_universe()})
            except Exception as exc: stages.append({'name':'pooled_model','status':'blocked','reason':str(exc)[:400]})
    else:
        stages.append({'name':'pooled_model','status':'blocked','reason':'No user-confirmed historical constituent membership intervals; survivorship-safe pooled evaluation is unavailable.'})

    with connect() as conn:
        oos_evidence=conn.execute("SELECT COUNT(*) n,COUNT(DISTINCT origin_date) dates,COUNT(DISTINCT model_key) streams FROM oos_predictions WHERE model_key<>'master_aggregator'").fetchone()
    enough_agg_evidence=oos_evidence['n']>=100 and oos_evidence['dates']>=30 and oos_evidence['streams']>=2
    if aggregator_exists and os.getenv('NIVESH_AUTO_RETRAIN','0').lower() not in {'1','true','yes'}:
        stages.append({'name':'master_aggregator','status':'skipped','reason':'Eligible aggregator already exists; automatic retraining is disabled.'})
    elif not enough_agg_evidence:
        stages.append({'name':'master_aggregator','status':'blocked','reason':f"Needs at least 100 base OOS rows, 30 distinct prediction dates and two base-model streams; currently {oos_evidence['n']} rows, {oos_evidence['dates']} dates, {oos_evidence['streams']} streams."})
    else:
        try:
            agg=models.train_aggregator()
            stages.append({'name':'master_aggregator','status':agg.get('status','unknown'),'result':agg})
        except Exception as exc: stages.append({'name':'master_aggregator','status':'blocked','reason':str(exc)[:400]})

    with connect() as conn:
        latest=conn.execute("SELECT model_key,COUNT(DISTINCT symbol) symbols,COUNT(*) rows,MAX(origin_date) latest FROM oos_predictions WHERE model_key<>'master_aggregator' GROUP BY model_key ORDER BY MAX(id) DESC").fetchall()
        previous={}
        for row in conn.execute('SELECT config_json,results_json FROM backtest_runs WHERE status=\'completed\' ORDER BY id DESC'):
            try:
                key=json.loads(row['config_json']).get('model_key')
                previous.setdefault(key,json.loads(row['results_json']).get('end_date',''))
            except (TypeError,ValueError): continue
    candidate=next((r for r in latest if r['symbols']>=5 and r['rows']>=45),None)
    if candidate and candidate['latest']>previous.get(candidate['model_key'],''):
        try:
            from .backtest import run_backtest
            result=run_backtest(model_key=candidate['model_key'],top_n=5,max_weight=.2)
            stages.append({'name':'walk_forward_backtest','status':result.get('status','unknown'),'result':result})
        except Exception as exc: stages.append({'name':'walk_forward_backtest','status':'failed','reason':str(exc)[:400]})
    elif candidate:
        stages.append({'name':'walk_forward_backtest','status':'skipped','reason':'No new OOS predictions since the last recorded backtest.'})
    else:
        stages.append({'name':'walk_forward_backtest','status':'blocked','reason':'Requires at least 45 OOS predictions across five symbols for a constrained cross-sectional backtest.'})

    blocked=[s for s in stages if s['status'] in {'blocked','failed'}]
    failed_stage=any(s['status']=='failed' for s in stages)
    did_work=any(s['name'] not in {'database_integrity','source_discovery'} and s['status'] in {'succeeded','partial_success'} for s in stages)
    status='failed' if failed_stage else ('partial_success' if did_work and blocked else 'succeeded' if did_work and not blocked else 'blocked')
    report={'run_id':run_id,'status':status,'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),
            'database_integrity':integrity,'reconcile_history':bool(reconcile_history),'stages':stages,
            'data_note':'Model training remains limited to the existing USER_CONFIRMED_UNVERIFIED policy gate. TejHQ provider rows are not silently promoted into that category; NIFTY 50 history and point-in-time membership are still required.'}
    report['reports']=_report_files(report)
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description='Nivesh local data and research maintenance')
    parser.add_argument('command',choices=('discover-sources','refresh-data','run-research','run-all'))
    parser.add_argument('--reconcile-history',action='store_true',help='Re-fetch configured provider history from its configured start date.')
    args=parser.parse_args(); init_db()
    if args.command=='discover-sources':
        result={'sources':list_sources()}
    elif args.command in {'run-all','run-research'}:
        result=run_all(args.reconcile_history)
    else:
        from . import market
        started=market.start_refresh('cli',args.reconcile_history)
        if started.get('status')=='running':
            while True:
                result=market.refresh_job(started['run_id'])
                if result.get('status')!='running': break
                time.sleep(.5)
        else: result=started
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    return 0 if args.command=='discover-sources' or result.get('status') in {'succeeded','partial_success','already_running'} else 2


if __name__=='__main__':
    raise SystemExit(main())
