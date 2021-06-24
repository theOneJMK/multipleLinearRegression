import jstat from 'jstat';
const { ftest } = jstat;
import { linearRegression } from '../linear/linearRegression.js';
import { isInteger } from 'mathjs';
import { LinearResult } from '../linear/linearResult.js';
import { Matrix } from 'ml-matrix';

class ResetResult {
    /**
     * The taken power for the reset Test
     */
    power;

    /**
     * The result of the F-Test
     */
    fValue;

    /**
     * The P-value of the F-Value
     */
    pValue;

    /**
     * Constructor
     */
    ResetResult() {

    }
}
/**
 * Calculate Ramseys Regression Specification and Error test with the given power of the fitted values.
 * @param {LinearResult} regressionResult A Result from a performed linearRegression.
 * @param {int} power The power to take from teh fitted values. Must be above 1.
 * @param {boolean} logging true for logging computational information.
 */
function ramsey(regressionResult, power, logging) {
    if (!isInteger(power) || power <= 1) {
        throw new Error("In order to perform a reset test the power must be given and be greater than 1.");
    }

    if (!(regressionResult instanceof LinearResult)) {
        throw new Error("The parameter regressionResult must be of type LinearResult.");
    }

    let ramseyExogen = new Matrix(regressionResult.exogen);
    for (let i = 2; i <= power; i++) {
        let powerOfFittedValues = new Array(regressionResult.noOfObservations);
        for (let j = 0; j < regressionResult.noOfObservations; j++) {
            powerOfFittedValues[j] = regressionResult.predicted[j] ** i;
        }
        ramseyExogen.addColumn(powerOfFittedValues);
    }

    let ramseyResult = linearRegression(regressionResult.endogen, ramseyExogen, logging);

    let dofEn = power - 1;//only the taken powers, so subtract one
    let dofDen = ramseyResult.residualDegreesOfFreedom;
    let fTestEnumerator = (regressionResult.sse - ramseyResult.sse) / dofEn;
    let fTestDenumerator = ramseyResult.sse / dofDen;

    let fScore = fTestEnumerator / fTestDenumerator;
    let pValueOfFtest = jstat.ftest(fScore, dofEn, dofDen);

    if (logging) {
        console.log(`Ramsey Reset FTest with power ${power}: ${fScore}, pValue: ${pValueOfFtest}`);
    }

    let result = new ResetResult();
    result.power = power;
    result.fValue = fScore;
    result.pValue = pValueOfFtest;

    return result;
}

export { ramsey, ResetResult };