"""Chronologically separated calibration for the event target P(excess return > 0)."""
import numpy as np
import pandas as pd

from .db import connect

MIN_FIT=200
MIN_EVAL=100
MIN_PER_CLASS=20


def _fit_platt(scores, labels):
    x=np.asarray(scores,float); y=np.asarray(labels,float)
    a=0.0; b=float(np.log((y.mean()+1e-6)/(1-y.mean()+1e-6)))
    for _ in range(100):
        z=np.clip(a*x+b,-30,30); p=1/(1+np.exp(-z)); w=np.maximum(p*(1-p),1e-8)
        grad=np.array([np.sum((p-y)*x),np.sum(p-y)])
        h=np.array([[np.sum(w*x*x)+1e-6,np.sum(w*x)],[np.sum(w*x),np.sum(w)+1e-6]])
        delta=np.linalg.solve(h,grad); a-=delta[0]; b-=delta[1]
        if np.max(np.abs(delta))<1e-8: break
    return float(a),float(b)


def _reliability(labels, probabilities, bins=10):
    y=np.asarray(labels,float); p=np.clip(np.asarray(probabilities,float),1e-9,1-1e-9)
    order=np.argsort(p); groups=np.array_split(order,min(bins,len(y)))
    ece=sum(len(g)/len(y)*abs(float(y[g].mean())-float(p[g].mean())) for g in groups if len(g))
    return {'brier_score':float(np.mean((p-y)**2)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            'expected_calibration_error':float(ece),'reliability_bins':[{'count':len(g),'mean_probability':float(p[g].mean()),'observed_rate':float(y[g].mean())} for g in groups if len(g)]}


def calibration_status(model_key:str):
    with connect() as c:
        rows=c.execute("SELECT origin_date,target_end_date,prediction,actual_excess,price_basis FROM oos_predictions WHERE model_key=? AND price_basis IN ('raw_unadjusted','adjusted_close') ORDER BY origin_date,symbol",(model_key,)).fetchall()
    if not rows: return {'status':'not_calibrated','model_key':model_key,'reason':'No eligible walk-forward predictions are available.'}
    frame=pd.DataFrame([dict(r) for r in rows]); dates=sorted(frame.origin_date.unique())
    if len(dates)<15: return {'status':'not_calibrated','model_key':model_key,'sample_count':len(frame),'reason':'Too few distinct prediction dates for separated calibration and evaluation.'}
    split=int(len(dates)*.7); fit_dates=dates[:max(0,split-5)]; eval_dates=dates[split:]
    fit=frame[frame.origin_date.isin(fit_dates)]; test=frame[frame.origin_date.isin(eval_dates)]
    fit_y=(fit.actual_excess.to_numpy(float)>0).astype(int); test_y=(test.actual_excess.to_numpy(float)>0).astype(int)
    counts={'fit':len(fit),'evaluation':len(test),'fit_positive':int(fit_y.sum()),'fit_negative':int(len(fit_y)-fit_y.sum()),'evaluation_positive':int(test_y.sum()),'evaluation_negative':int(len(test_y)-test_y.sum())}
    if len(fit)<MIN_FIT or len(test)<MIN_EVAL or min(counts['fit_positive'],counts['fit_negative'],counts['evaluation_positive'],counts['evaluation_negative'])<MIN_PER_CLASS:
        return {'status':'not_calibrated','model_key':model_key,'sample_count':len(frame),'counts':counts,'fit_date_range':[min(fit_dates),max(fit_dates)] if fit_dates else None,'evaluation_date_range':[min(eval_dates),max(eval_dates)] if eval_dates else None,'method':'Platt logistic calibration with a five-session date purge','reason':f'Requires at least {MIN_FIT} fit and {MIN_EVAL} evaluation observations, with at least {MIN_PER_CLASS} examples of each event outcome in both periods.'}
    a,b=_fit_platt(fit.prediction.to_numpy(float),fit_y)
    test_p=1/(1+np.exp(-np.clip(a*test.prediction.to_numpy(float)+b,-30,30)))
    # For a future forecast, refit using all available historical OOS rows only
    # after the disjoint evaluation report has been computed.
    all_y=(frame.actual_excess.to_numpy(float)>0).astype(int); final_a,final_b=_fit_platt(frame.prediction.to_numpy(float),all_y)
    return {'status':'calibrated','model_key':model_key,'event':'five-session excess return > 0','probability_semantics':'Historical model-conditional event frequency estimate; not a guarantee or investment recommendation.',
            'sample_count':len(frame),'counts':counts,'fit_date_range':[min(fit_dates),max(fit_dates)],'evaluation_date_range':[min(eval_dates),max(eval_dates)],
            'method':'Platt logistic calibration; earliest 70% of dates fit, five-session date purge, latest 30% held out for reliability evaluation.',
            'evaluation_reliability':_reliability(test_y,test_p),'final_calibrator':{'a':final_a,'b':final_b,'fit_date_range':[min(dates),max(dates)]},
            'price_bases':sorted(frame.price_basis.unique().tolist()),'out_of_sample_only':True}


def calibrated_probability(model_key:str,prediction:float):
    result=calibration_status(model_key)
    if result['status']!='calibrated': return result
    cal=result['final_calibrator']; p=1/(1+np.exp(-np.clip(cal['a']*float(prediction)+cal['b'],-30,30)))
    return {**result,'probability_excess_positive':float(p)}
