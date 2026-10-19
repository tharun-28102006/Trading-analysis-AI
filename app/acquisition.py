"""Source discovery and safe acquisition gate.

The catalog tracks data categories independently. A source's open license does
not imply that an API adapter exists or that a network fetch has been tested.
"""
from __future__ import annotations

import json
from datetime import timedelta
from datetime import datetime, timezone

from .db import connect

CHECKED_AT = '2026-10-18'

SOURCE_CATALOG = (
    {
        'source_key': 'tejhq_indian_markets',
        'name': 'TejHQ Indian Markets NSE EOD data',
        'owner': 'TejHQ',
        'data_types': ['equity_eod_ohlcv'],
        'coverage': ['NSE EQ/BE/BZ equities; source reports 2010-01-04 to current trading date; roughly 2,300 instruments/day'],
        'access_method': 'Keyless public REST endpoint for per-symbol inclusive date-range OHLCV and a public status endpoint',
        'permission_status': 'authorized_by_provider_dataset_license',
        'authentication': 'No key or account required for /v1/ohlcv and /v1/status',
        'pricing_summary': 'Keyless OHLCV and status endpoints are listed as Open at ₹0; no paid subscription is used.',
        'terms_url': 'https://huggingface.co/datasets/tejhq/indian-markets',
        'docs_url': 'https://tejhq.dev/docs',
        'request_limits': 'The public keyless tier has no published quota; Nivesh serializes calls, limits to 1 request/second and retries at most twice. The separate free-key tier documents 1,000 calls/day and 300/minute.',
        'attribution': 'TejHQ Indian Markets; NSE official EOD bhavcopy dataset; retrieved at UTC timestamp; MIT dataset license as published by TejHQ.',
        'notes': 'TejHQ publishes the dataset with MIT metadata and states cleaned data is redistributable. Nivesh contacts only api.tejhq.dev, never NSE website endpoints. NSE website terms prohibit direct website automation and simulation use; TejHQ is the data publisher and its dataset license is the basis for this private local research integration. The dataset does not include NIFTY 50 index OHLCV or NIFTY 50 membership history.',
        'documentation_status': 'reviewed', 'access_status': 'provider_dataset_mit_license', 'adapter_status': 'implemented',
        'connectivity_status': 'not_tested', 'access_cost_type': 'free_open_api',
    },
    {
        'source_key': 'nse_research_channel',
        'name': 'NSE research data sharing channel',
        'owner': 'National Stock Exchange of India / NSE Data & Analytics',
        'data_types': ['equity_eod_ohlcv', 'index_ohlc', 'index_constituents'],
        'coverage': ['NSE daily/monthly reports including bhavcopy; exact files and history subject to approval'],
        'access_method': 'Research request routed through NSE Economic Policy Research Department; possible NDA and periodic usage reports',
        'permission_status': 'accreditation_and_approval_required',
        'authentication': 'Accredited academic/research institution eligibility and NSE approval required',
        'pricing_summary': 'NSE permits qualifying research-oriented data access without cost up to the applicable 2 GB aggregate threshold; not general public access.',
        'terms_url': 'https://www.nseindia.com/static/nse-disclaimer',
        'docs_url': 'https://www.nseindia.com/static/research/research-initiatives',
        'request_limits': '2 GB aggregate threshold for qualifying research-oriented data; periodic progress reports may be required.',
        'attribution': 'Terms, NDA and source-specific attribution are determined by NSE approval.',
        'notes': 'NSE terms restrict free access to accredited academic institutions, recognized research organizations and think tanks, and exclude commercial, speculative or investigative uses. Public website terms also prohibit automated collection and use in simulation. Nivesh has no approved research agreement; no requests are made.',
        'documentation_status': 'reviewed', 'access_status': 'blocked_accreditation_and_approval', 'adapter_status': 'not_implemented',
        'connectivity_status': 'not_tested', 'access_cost_type': 'free_restricted',
    },
    {
        'source_key': 'nse_public_website',
        'name': 'NSE public reports and website',
        'owner': 'National Stock Exchange of India',
        'data_types': ['public_reports', 'equity_eod_ohlcv', 'index_ohlc'],
        'coverage': ['Public daily bhavcopy and other reports; daily reports do not by themselves establish historical index or membership rights'],
        'access_method': 'Public webpage/download for human access; automation not used',
        'permission_status': 'not_permitted_for_automated_research_ingestion',
        'authentication': 'Website terms apply; no approved automated access',
        'pricing_summary': 'Some research-oriented access may be free only under NSE eligibility and approval; this route is not applicable to the current app.',
        'terms_url': 'https://www.nseindia.com/static/nse-terms-of-use',
        'docs_url': 'https://www.nseindia.com/all-reports',
        'request_limits': 'Automated website collection is prohibited; no API quota for this use is published.',
        'attribution': 'Use and storage require applicable written permission; website terms exclude simulation use.',
        'notes': 'NSE currently publishes daily UDiFF bhavcopy reports. Current website terms prohibit systematic automated collection and simulation use, so Nivesh does not download these files.',
        'documentation_status': 'reviewed', 'access_status': 'prohibited_for_current_use', 'adapter_status': 'not_implemented',
        'connectivity_status': 'not_tested', 'access_cost_type': 'free_restricted',
    },
    {
        'source_key': 'bse_public_reports',
        'name': 'BSE public reports and market data',
        'owner': 'BSE Limited',
        'data_types': ['equity_eod_ohlcv', 'index_ohlc', 'corporate_data'],
        'coverage': ['BSE report pages and self-data-feed products; fields/history vary by report and product'],
        'access_method': 'Public report pages exist; automated access and personal-research reuse rights for the required dataset were not established',
        'permission_status': 'reuse_terms_unverified',
        'authentication': 'No public credential-free API for the required licensed use was verified',
        'pricing_summary': 'BSE Self Data Feed is a commercial subscription; no paid route is used. Public-report reuse terms remain unverified.',
        'terms_url': 'https://www.bseindia.com/static/about/disclaimer.aspx',
        'docs_url': 'https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx',
        'request_limits': 'No documented automated-download limit for a permitted personal research feed was verified.',
        'attribution': 'Requires applicable report-specific or feed terms.',
        'notes': 'Not integrated: public visibility does not establish permission for automated collection, local retention and backtesting. BSE paid self-data feed is excluded.',
        'documentation_status': 'reviewed', 'access_status': 'reuse_terms_unverified', 'adapter_status': 'not_implemented',
        'connectivity_status': 'not_tested', 'access_cost_type': 'unknown_or_paid',
    },
    {
        'source_key': 'mospi_cpi_api',
        'name': 'MoSPI Consumer Price Index API',
        'owner': 'Ministry of Statistics and Programme Implementation',
        'data_types': ['macro_monthly_cpi'],
        'coverage': ['Official CPI index and inflation series; available series and period are API-specific'],
        'access_method': 'Public MoSPI e-Sankhyiki CPI API; base year 2024, current series',
        'permission_status': 'open_access_with_attribution',
        'authentication': 'No account credentials documented by the MoSPI-maintained e-Sankhyiki client for this endpoint',
        'pricing_summary': 'Free public current CPI API access is described by the MoSPI-maintained e-Sankhyiki client; no subscription price is listed.',
        'terms_url': 'https://data.gov.in/Godl',
        'docs_url': 'https://github.com/nso-india/mospi-esankhyiki',
        'request_limits': 'Public quota not stated in the manual; client must use conservative rate limiting.',
        'attribution': 'Attribute MoSPI, CPI dataset/API, retrieval date, and Government Open Data License - India where applicable.',
        'notes': 'The MoSPI-maintained e-Sankhyiki client documents the 2024-base current CPI API. CPI is macro context only; it cannot supply equity or NIFTY observations. Adapter stores CPI separately and does not expose values to models without verified publication timestamps.',
        'documentation_status': 'reviewed', 'access_status': 'free_open_api', 'adapter_status': 'implemented',
        'connectivity_status': 'not_tested', 'access_cost_type': 'free_account_required',
    },
)



def sync_source_registry() -> None:
    """Persist the documented source inventory without contacting providers."""
    with connect() as conn:
        for source in SOURCE_CATALOG:
            fields = dict(source)
            fields['checked_at'] = CHECKED_AT
            conn.execute(
                '''INSERT INTO source_registry(source_key,name,owner,data_types_json,coverage_json,access_method,
                   permission_status,authentication,pricing_summary,terms_url,docs_url,request_limits,attribution,
                   notes,checked_at,documentation_status,access_status,adapter_status,connectivity_status,access_cost_type)
                   VALUES(:source_key,:name,:owner,:data_types_json,:coverage_json,:access_method,
                   :permission_status,:authentication,:pricing_summary,:terms_url,:docs_url,:request_limits,
                   :attribution,:notes,:checked_at,:documentation_status,:access_status,:adapter_status,:connectivity_status,:access_cost_type)
                   ON CONFLICT(source_key) DO UPDATE SET name=excluded.name,owner=excluded.owner,
                   data_types_json=excluded.data_types_json,coverage_json=excluded.coverage_json,
                   access_method=excluded.access_method,permission_status=excluded.permission_status,
                   authentication=excluded.authentication,pricing_summary=excluded.pricing_summary,
                   terms_url=excluded.terms_url,docs_url=excluded.docs_url,request_limits=excluded.request_limits,
                   attribution=excluded.attribution,notes=excluded.notes,checked_at=excluded.checked_at,
                   documentation_status=excluded.documentation_status,access_status=excluded.access_status,
                   adapter_status=excluded.adapter_status,connectivity_status=excluded.connectivity_status,
                   access_cost_type=excluded.access_cost_type''',
                {**fields, 'data_types_json': json.dumps(fields['data_types']),
                 'coverage_json': json.dumps(fields['coverage'])},
            )
        current_keys = [source['source_key'] for source in SOURCE_CATALOG]
        placeholders = ','.join('?' for _ in current_keys)
        conn.execute(f'DELETE FROM source_registry WHERE source_key NOT IN ({placeholders})', current_keys)
        # Earlier registry migrations copied explanatory notes into last_error.
        conn.execute("UPDATE source_registry SET last_error=NULL WHERE last_attempt_status='blocked_no_adapter' AND last_error=notes")


def list_sources() -> list[dict]:
    sync_source_registry()
    with connect() as conn:
        rows = [dict(row) for row in conn.execute('SELECT * FROM source_registry ORDER BY source_key')]
    current_keys = {source['source_key'] for source in SOURCE_CATALOG}
    rows = [row for row in rows if row['source_key'] in current_keys]
    for row in rows:
        row['data_types'] = json.loads(row.pop('data_types_json'))
        row['coverage'] = json.loads(row.pop('coverage_json'))
        if row['source_key']=='mospi_cpi_api':
            row['last_attempt_status']=row.get('last_attempt_status') or 'not_tested'
            with connect() as conn:
                coverage=conn.execute("SELECT COUNT(*) n,MIN(year*100+month) first_period,MAX(year*100+month) last_period FROM macro_observations WHERE source='MoSPI CPI API'").fetchone()
            row['coverage'].append(f"Stored rows: {coverage['n']}; period range: {coverage['first_period'] or 'none'} to {coverage['last_period'] or 'none'}; publication timestamps are used only when supplied by source.")
    return rows


def run_acquisition(triggered_by: str = 'manual', reconcile_history: bool = False) -> dict:
    """Refresh configured permitted sources and record each outcome durably."""
    sync_source_registry()
    started = datetime.now(timezone.utc).isoformat()
    sources = list_sources()
    permitted_statuses={'authorized_by_local_user_attestation','open_access_with_attribution'}
    enabled = [source for source in sources if source['permission_status'] in permitted_statuses
               and source['adapter_status'] == 'implemented'
               ]
    with connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        pending=conn.execute("SELECT id,started_at FROM acquisition_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
        if pending:
            try: age=datetime.now(timezone.utc)-datetime.fromisoformat(pending['started_at'])
            except ValueError: age=timedelta(days=1)
            if age < timedelta(hours=6):
                return {'run_id':pending['id'],'status':'already_running','network_requests_made':0,
                        'data_acquired':False,'reason':'A refresh is already active; no duplicate acquisition was started.'}
            conn.execute("UPDATE acquisition_runs SET finished_at=?,status='interrupted',errors_json=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(),json.dumps(['Previous acquisition exceeded the six-hour recovery window; it was marked interrupted.']),pending['id']))
        for source in sources:
            conn.execute('UPDATE source_registry SET last_attempt_at=?,last_attempt_status=? WHERE source_key=?',
                         (started, 'not_attempted_no_adapter', source['source_key']))
        cursor = conn.execute(
            'INSERT INTO acquisition_runs(started_at,status,attempted_sources_json,errors_json,summary_json) VALUES(?,?,?,?,?)',
            (started, 'running', json.dumps([source['source_key'] for source in enabled]), '[]',
             json.dumps({'triggered_by': triggered_by, 'network_requests_made': 0})),
        )
        run_id = cursor.lastrowid
    if not enabled:
        error = 'No permitted source adapter is eligible. NSE public website collection is prohibited for this automated simulation/research use; NSE free research access requires qualifying institutional accreditation and approval. BSE reuse terms for automated research are unverified. No provider was contacted.'
        summary = {'network_requests_made': 0, 'authorized_sources': 0, 'blocked_sources': len(sources),
                   'data_acquired': False, 'reason': error}
        with connect() as conn:
            conn.execute('UPDATE acquisition_runs SET finished_at=?,status=?,errors_json=?,summary_json=? WHERE id=?',
                         (datetime.now(timezone.utc).isoformat(), 'blocked', json.dumps([error]),
                          json.dumps(summary), run_id))
            conn.execute("UPDATE source_registry SET last_attempt_status='blocked_no_adapter' WHERE adapter_status<>'implemented'")
            conn.execute("UPDATE source_registry SET last_attempt_status='blocked_permission_or_adapter',last_error=NULL WHERE source_key IN ('nse_research_channel','nse_public_website','bse_public_reports')")
        return {'run_id': run_id, 'status': 'blocked', **summary}
    # The only implemented free adapter is optional CPI context. It is ingested
    # independently and never presented as an equity/index observation.
    errors=[]; successes=[]
    for source in enabled:
        try:
            from .providers.mospi import MoSPIClient, store_cpi
            payload,rows=MoSPIClient().fetch_cpi()
            stored=store_cpi(payload,rows)
            successes.append({'source_key':source['source_key'],'category':'macro_monthly_cpi',**stored})
            with connect() as conn:
                conn.execute("UPDATE source_registry SET last_success_at=?,last_error=NULL,last_attempt_status='succeeded' WHERE source_key=?",(datetime.now(timezone.utc).isoformat(),source['source_key']))
        except Exception as exc:
            message=f'{type(exc).__name__}: {str(exc)[:300]}'
            errors.append(message)
            with connect() as conn:
                conn.execute("UPDATE source_registry SET last_error=?,last_attempt_status='failed' WHERE source_key=?",(message,source['source_key']))
    if successes:
        rows_received=sum(item['rows_received'] for item in successes)
        new_rows=sum(item['new_rows'] for item in successes)
        status='partial_success' if errors else 'succeeded'
        summary={'network_requests_made':len(enabled),'authorized_sources':len(enabled),'blocked_sources':len(sources)-len(enabled),
                 'data_acquired':bool(rows_received),'equity_market_data_acquired':False,'rows_received':rows_received,
                 'new_rows':new_rows,'categories':successes,'errors':errors,
                 'reason':'Free MoSPI CPI context was refreshed; no equity or NIFTY 50 market feed is eligible.'}
    else:
        status='failed'
        summary={'network_requests_made':len(enabled),'authorized_sources':len(enabled),'blocked_sources':len(sources)-len(enabled),
                 'data_acquired':False,'equity_market_data_acquired':False,'errors':errors,
                 'reason':'MoSPI CPI retrieval failed; no equity or index feed was contacted.'}
    with connect() as conn:
        conn.execute('UPDATE acquisition_runs SET finished_at=?,status=?,errors_json=?,summary_json=? WHERE id=?',
                     (datetime.now(timezone.utc).isoformat(),status,json.dumps(errors),json.dumps(summary),run_id))
    return {'run_id':run_id,'status':status,**summary}


def acquisition_history(limit: int = 50) -> list[dict]:
    with connect() as conn:
        return [dict(row) for row in conn.execute(
            'SELECT * FROM acquisition_runs ORDER BY id DESC LIMIT ?', (min(max(limit, 1), 200),)
        )]

