/**
 * Class for storing the result of a linear Regression.
 */
class LinearResult {
    /**
     * The observed y values as Matrix.
     */
    endogen;

    /**
     * The observed variables as Matrix.
     */
    exogen;

    /**
     * The number of observations.
     */
    noOfObservations;

    /**
     * Flag indicating if an intercept is present in exogen.
     */
    constantPresent;

    /**
     * The coefficients of the endogen Variables.
     */
    coefficients;

    /**
     * The number of coefficients used for this regression including the intercept.
     */
    noOfCoefficients;

    /**
     * Standard error of the coefficients.
     */
    stdErrorOfCoefficients;
    
    /**
     * tValues of the coefficients.
     */
    tValues;
    
    /**
     * p test values of the tValues of the coefficients.
     */
    pValues;
    
    /**
     * The predicted values.
     */
    predicted;

    /**
     *  The residuals.
    */
    residuals;
    
    /**
     * Variance of the residuals.
     */
    sSquare;

    /**
     * R squared. The coefficient of determination.
     */
    rSquared;
    
    /**
     * F-statistic for the regression model.
     */
    fValue;

    /**
     * The pValue for the F-statistic.
     */
    pValueOfFValue;

    /**
     * Degrees of freedom for the residuals = number of observations - number of coefficients.
     */
    residualDegreesOfFreedom;

    /**
     * Degrees of freedom for model = number of coefficients excluding the constant.
     */
    modelDegreesOfFreedom;

    /**
     * Squared sum of errors (squared sum of residuals).
     */
    sse;

    /**
     * Squared sum of total = squared sum of errors + squared sum of regression.
     */
    sst;

    /**
     * Squared sum of regression.
     */
    ssr;

    /** 
     * Mean squared error.
     */
    mse;

    /**
     * The equation of the Regression build from the coefficients.
     */
    regressionEquation;

    /**
     * Empty constructor
     */
    LinearResult() {    
    }
}

module.exports = { LinearResult: LinearResult };