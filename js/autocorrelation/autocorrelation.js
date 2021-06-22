const math = require('mathjs');
const linReg = require('../linear/linearRegression.js');
const jstat = require('jstat');
const tools = require('../tools/tools.js');
const { LinearResult } = require('../linear/linearResult.js');
const { isInteger, isPositive } = require('mathjs');
const { default: Matrix } = require('ml-matrix');

/**
 * Calculate the Durbin-Watson test statistic for firsrt order autocorelation.
 * @param {Array} regressionResult A Result from a performed linearRegression.
 * @param {boolean} logging true for logging computational information.
 */
function durbinWatson(regressionResult, logging) {
    if (!(regressionResult instanceof LinearResult)) {
        throw new Error("The parameter regressionResult must be of type LinearResult.");
    }

    let enumerator = 0;
    let residuals = regressionResult.residuals;
    for (let i = 1, len = residuals.length; i < len; i++) {
        enumerator += math.square(residuals[i] - residuals[i - 1]);
    }

    let d = enumerator / regressionResult.sse;
    if (logging) {
        console.log("\ndurbinWatson: " + d);
    }
    return d;
}

class BreuschGodfreyResult {

    /**
     * The calculated lagrange multiplier
     */
    lagrangeMultiplier

    /**
     * The P-Value of the lagrange multiplier
     */
    lagrangeMultiplierPValue

    /**
     * The F-Value of thr Breusch-Godfrey test.
     */
    fValue

    /**
     * The P-Value of the F-Test
     */
    pValue

    /**
     * Constructor
     */
    BreuschGodfreyResult() {

    }
}

/**
 * Calculate the Berusch-Godfrey statistic for autocorrelation in the nlags order.
 * Note autocorrelation is sometimes referred to as serialcorrelation.
 * @param {LinearResult} regressionResult A Result from a performed linearRegression.
 * @param {int} nlags The orde of the autocorrelation to check.
 * @param {boolean} logging true for logging computational information.
 */
function breuschGodfrey(regressionResult, nlags, logging) {
    if (!(regressionResult instanceof LinearResult)) {
        throw new Error("The parameter regressionResult must be of type LinearResult.");
    }

    if (!(nlags && isInteger(nlags) && isPositive(nlags))) {
        throw new Error("nlags must a positive Integer.");
    }

    let lagMatrix = tools.lagMatrix(regressionResult.residuals, nlags, logging);

    lagMatrix = tools.addConstant(lagMatrix);

    let exogen = new Matrix(regressionResult.exogen);
    for (let i = 0; i < lagMatrix.columns; i++) {
        exogen.addColumn(lagMatrix.getColumnVector(i));
    }

    if (logging) {
        console.log("Breush-Godfrey exogen: " + exogen);
    }

    let breuschGodfreyRegressionResult = linReg.linearRegression(regressionResult.residuals, exogen, logging);

    let lagrangeMultiplier = regressionResult.noOfObservations * breuschGodfreyRegressionResult.rSquared;
    let lagrangeMultiplierPValue = 1 - jstat.chisquare.cdf(lagrangeMultiplier, nlags);//so called survival function

    let dofEn = nlags;
    let dofDen =
        breuschGodfreyRegressionResult.residualDegreesOfFreedom + 1; //a constant is always present. Therefore the dof must be corrected.
    let dividend = regressionResult.sse - breuschGodfreyRegressionResult.sse;
    let fTestEnumerator = dividend / dofEn;
    let fTestDenumerator = breuschGodfreyRegressionResult.sse / dofDen;

    let fValue = fTestEnumerator / fTestDenumerator;
    let pValue = jstat.ftest(fValue, dofEn, dofDen);

    if (logging) {
        console.log(`Breusch-Godfrey, lm: ${lagrangeMultiplier}, lmPVal: ${lagrangeMultiplierPValue}, FValue: ${fValue}, FPValue: ${pValue}`);
    }

    let result = new BreuschGodfreyResult();
    result.lagrangeMultiplier = lagrangeMultiplier;
    result.lagrangeMultiplierPValue = lagrangeMultiplierPValue;
    result.fValue = fValue;
    result.pValue = pValue;
    return result;
}

module.exports = { durbinWatson: durbinWatson, BreuschGodfreyResult: BreuschGodfreyResult, breuschGodfrey: breuschGodfrey };