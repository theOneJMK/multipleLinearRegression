import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json

YOUTUBE = "YOUTUBE"
FACEBOOK = "FACEBOOK"
NEWSPAPER = "NEWSPAPER"
SALES = "SALES"


def create_designmatrix(dataset):
    x_values = pd.DataFrame(
        {
            YOUTUBE: dataset[YOUTUBE],
            FACEBOOK: dataset[FACEBOOK],
            NEWSPAPER: dataset[NEWSPAPER],
        }
    )
    x_values = sm.add_constant(x_values)
    return x_values


def perform_test_regression(dataset):
    model = sm.OLS(endog=marketing[SALES], exog=create_designmatrix(dataset))
    return model.fit()


def perform_test_residual_diagnostics(dataset, marketing_result):
    x_values = create_designmatrix(dataset)
    # RESET
    reset_power = 4
    reset = sms.linear_reset(marketing_result, reset_power, use_f=True)
    reset_fval = reset.fvalue.squeeze().tolist()
    reset_pval = reset.pvalue.squeeze().tolist()

    print("RESET: ", reset)

    # goldfeld quandt for homoskedasticity
    gq_fval, gq_pval, gq_order = sms.het_goldfeldquandt(
        dataset[SALES], x_values)
    print("Goldfeld-Quandt: fval: {}, pval {}, order: {}".format(gq_fval, gq_pval, gq_order))

    # durbin watson for first order autocorrelation
    dw = sms.durbin_watson(resids=marketing_result.resid)
    print("Durbin-Watson: {}".format(dw))

    # breusch godfrey for autocorrelation
    bg_nlags = 3
    bg_lm, bg_lmpval, bg_fval, bg_fpval = sms.acorr_breusch_godfrey(
        marketing_result, bg_nlags)
    print("Breusch-Godfrey: lm: {}, lmpval: {}, fval: {}, fpval: {}".format(bg_lm,
                                                                            bg_lmpval, bg_fval, bg_fpval))

    vifs = [variance_inflation_factor(x_values.values, i)
            for i in range(x_values.shape[1])]
    print("VIFs: ", vifs)
    return {'RESET': {'fValue': reset_fval, 'pValue': reset_pval, 'resetPower': reset_power},
            'Goldfeld-Quandt': {'fValue': gq_fval, 'pValue': gq_pval, 'order': gq_order},
            'Durbin-Watson': dw,
            'Breusch-Godfrey': {'lagrangeMultiplier': bg_lm, 'lagrangeMultiplierPValue': bg_lmpval, 'fValue': bg_fval, 'pValue': bg_fpval, 'bgNLags': bg_nlags},
            'VIF': vifs}


def write_result_as_json(marketing_result, diagnostics):
    print(marketing_result.summary())
    marketing_result_to_json = {
        'noOfObservations': marketing_result.nobs,
        'coefficients': marketing_result.params.values.tolist(),
        'noOfCoefficients': marketing_result.params.size,
        'stdErrorOfCoefficients': marketing_result.bse.tolist(),
        'tValues': marketing_result.tvalues.values.tolist(),
        'pValues': marketing_result.pvalues.values.tolist(),
        'predicted': marketing_result.fittedvalues.values.tolist(),
        'residuals': marketing_result.resid.values.tolist(),
        'sSquare': marketing_result.mse_resid,
        'rSquared': marketing_result.rsquared,
        'fValue': marketing_result.fvalue,
        'pValueOfFValue': marketing_result.f_pvalue,
        'residualDegreesOfFreedom': marketing_result.df_resid,
        'modelDegreesOfFreedom': marketing_result.df_model,
        # ssr in statsmodels = squared sum of residuals; in js quared sum of error
        'sse': marketing_result.ssr,
        'sst': marketing_result.centered_tss,  # squared sum of total
        # ess in python = squared sum of regression (ssr)
        'ssr': marketing_result.ess,
        'mseResid': marketing_result.mse_resid,  # mean squared error / dof_resid
        'mseTotal': marketing_result.mse_total,  # mean squared error / centered_tss
        'mseModel': marketing_result.mse_model,  # mean squared error / dof_model
        'residualDiagnostic': diagnostics
    }

    with open('data/test/marketingPythonResult.json', 'w') as outfile:
        json.dump(marketing_result_to_json, outfile)


if __name__ == "__main__":
    marketing = pd.read_csv(
        "data/test/marketingData.csv",
        sep=",",
        decimal=".",
        header=0,
        names=[YOUTUBE, FACEBOOK, NEWSPAPER, SALES],
    )

    marketing_result = perform_test_regression(marketing)

    diagnostics = perform_test_residual_diagnostics(
        marketing, marketing_result)

    write_result_as_json(marketing_result, diagnostics)
