"use strict";
const math = require('mathjs');
const jstat = require('jstat');
const tools = require('../tools/tools.js');
const { LinearResult } = require('./linearResult.js');
const { Matrix, pseudoInverse } = require('ml-matrix');
const { isPositive, setMultiplicity } = require('mathjs');

/**
 * Performs a linear regression on the observed values y and the variables given in x using the least squares algorythm.
 * y and y need to have the same number of recors. Otherwise an Error will be risen.
 * @param {Matrix|Array} y The observed values. May be a one dimensonal Array or math.Matrix.
 * @param {Matrix|Array} x The observed variabels. May be a two dimensonal Array or math.Matrix.
 * @param {boolean} logging true for logging computational information.
 * @returns {LinearResult} Information about the performed regression. 
 */
function linearRegression(y, x, logging) {
    let yMatrix;
    if (y instanceof Array) {
        yMatrix = Matrix.columnVector(y);
    } else if (y instanceof Matrix) {
        yMatrix = y;
    } else {
        throw new Error('y needs to be an array or a matrix');
    }

    let designMatrix;
    if (x instanceof Array) {
        designMatrix = new Matrix(x);
    } else if (x instanceof Matrix) {
        designMatrix = x;
    } else {
        throw new Error('x needs to be an array or a matrix');
    }

    if (designMatrix.rows != yMatrix.rows) {
        throw new Error("Number of observations differs between y and x");
    }

    //log a warning, if there are more coefficients than observations
    if (yMatrix.rows - designMatrix.columns <= 0) {
        console.warn("More obserbations than coefficients are given. Some results may not hold true.");
    }

    let output = new LinearResult();
    output.constantPresent = tools.isConstantPresent(designMatrix);

    let coefficientResult = calculateCoefficients(designMatrix, yMatrix, logging);
    output.coefficients = coefficientResult.coefficients;
    output.noOfCoefficients = coefficientResult.coefficients.rows;

    let fitted = fit(designMatrix, yMatrix, output.coefficients, logging);
    output.predicted = fitted.predicted;
    output.noOfObservations = fitted.predicted.length;
    output.residuals = fitted.residuals;

    let rSquared = rSquare(fitted.predicted, fitted.residuals, yMatrix, output.noOfObservations, output.constantPresent, logging);
    output.rSquared = rSquared.rSquared;
    output.sst = rSquared.sst;
    output.ssr = rSquared.ssr;
    output.sse = rSquared.sse;
    output.mse = rSquared.mse;

    output.residualDegreesOfFreedom = tools.degreesOfFreedom(output.noOfObservations, output.noOfCoefficients, logging);
    output.modelDegreesOfFreedom = output.constantPresent ? output.noOfCoefficients - 1 : output.noOfCoefficients;

    output.fValue = fStatistic(output.noOfCoefficients, output.residualDegreesOfFreedom, rSquared.sse, rSquared.sst);
    output.pValueOfFValue = jstat.ftest(output.fValue, output.modelDegreesOfFreedom, output.residualDegreesOfFreedom);
    if (logging) {
        console.log("\nF-Statistics: " + output.fValue);
    }

    output.sSquare = varianceOfResiduals(rSquared.sse, output.residualDegreesOfFreedom, logging);
    output.stdErrorOfCoefficients = stdErrorOfCoefficients(coefficientResult.pseudoInverse, output.sSquare, logging);

    let tValues = tValue(output.coefficients, output.stdErrorOfCoefficients, output.residualDegreesOfFreedom, output.constantPresent, logging);
    output.tValues = tValues.tValues;
    output.pValues = tValues.pValues;

    output.exogen = designMatrix;
    output.endogen = yMatrix;

    output.regressionEquation = buildRegressionEquation(output.coefficients, output.constantPresent);
    return output;
}

/**
 * Calculcate the coefficients using the equation b = (X'X)^-1 X'y.
 * To calculte the inverse (X'X)^-1 ml-matrix is used, because it offers the option to calculate the Moore-Penrose pseudo-inverse.
 * @param {Matrix} designMatrix The design matrix.
 * @param {Matrix} yMatrix The observed y values.
 * @param {boolean} logging true for logging computational information.
 * @returns Object containing the coefficients and the used pseudo-inverse
 */
function calculateCoefficients(designMatrix, yMatrix, logging) {
    if (logging) {
        console.log("designMatrix" + designMatrix);
    }

    //transpose the designMatrix
    let transposedDesignMatrix = designMatrix.transpose();
    if (logging) {
        console.log("transposedDesignMatrix: " + transposedDesignMatrix);
    }

    // let p = transposedDesignMatrix.mmul(designMatrix);
    // if (logging) {
    //     console.log("p: " + p);
    // }

    let pseudoInv = pseudoInverse(transposedDesignMatrix.mmul(designMatrix));
    // let q = transposedDesignMatrix.mmul(yMatrix);

    // if (logging) {
    //     console.log("invP: " + pseudoInv);
    //     console.log("q: " + q);
    //}

    //Calculate the coefficients b
    // let regCoeff = pseudoInv.mmul(transposedDesignMatrix.mmul(yMatrix));
    // if (logging) {
    //     console.log("\nCoefficients: " + regCoeff);
    // }
    let result = {};
    result.coefficients = pseudoInv.mmul(transposedDesignMatrix.mmul(yMatrix));;
    result.pseudoInverse = pseudoInv;
    return result;
}

/**
 * Fits the values from the design matrix with the coefficients to calculate the fitted values.
 * Basically it performs y_hat = b0 + b1 * x1 + b2 * x2 ... or y_hat = b0 * x0 + b1 * x1 depentding on
 * wether a constant is Present or not. In addition teh residuals are calculated.
 * @param {Matrix} designMatrix The design matrix.
 * @param {Matrix} yMatrix The observed y values.
 * @param {Matrix} coefficients The regression coefficients.
 * @param {boolean} logging true for logging computational information.
 * @returns an object with the predicted (fitted) values as array and the residuals as array.
 */
function fit(designMatrix, yMatrix, coefficients, logging) {
    let predicted = new Array(yMatrix.rows);
    let residuals = new Array(yMatrix.rows);

    for (let i = 0; i < designMatrix.rows; i++) {
        let sum = calculateRow(
            coefficients, designMatrix.getRowVector(i), coefficients.rows - 1);

        predicted[i] = sum;
        let resid = yMatrix.get(i, 0) - sum;
        residuals[i] = resid;
    }

    if (logging) {
        console.log("\npredicted: " + predicted);
        console.log("\nresiduals: " + residuals);
    }
    let result = {};
    result.predicted = predicted;
    result.residuals = residuals;
    return result;
}

/**
 * Calculated the sum of a row with recursion.
 * @param {Matrix} coefficients the coefficients of the regression
 * @param {Matrix} row the current row itereated from fit()
 * @param {int} index number of columns in coefficients row.
 */
function calculateRow(coefficients, row, index) {
    let sum = coefficients.get(index, 0) * row.get(0, index);

    if (index > 0) {
        return calculateRow(coefficients, row, --index) + sum;
    }

    return sum;
}

/**
* Calculates the values for thr squared sum of errors (sse), squared sum of regression (ssr), squared sum of total (sst),
* rSquared (regresssion coefficient) and the mean squared error of sse.
* @param {Array} predicted The predicted values from fit().
* @param {Array} residuals The calculated residuals from fit().
* @param {Matrix} yMatrix The observed y values.
* @param {Integer} noOfObservations the number of observations.
* @param {boolean} constantPresent true if a constant is present.
* @param {boolean} logging true for logging computational information.
* @returns An Object containg the described values.
*/
function rSquare(predicted, residuals, yMatrix, noOfObservations, constantPresent, logging) {
    let meanY = yMatrix.mean();

    let sst = 0;
    let ssr = 0;
    let squaredErrors = new Array(residuals.length);
    let sse = 0;
    yMatrix.apply((row, column) => {
        sst += math.square(yMatrix.get(row, column) - meanY);
        ssr += math.square(predicted[row] - meanY);
        squaredErrors[row] = math.square(residuals[row]);
        sse += squaredErrors[row];
    });

    let rSquared;
    if (constantPresent) {
        let factor1 = 1 / (noOfObservations - 1);
        let enumerator = (factor1 * sse);
        let denumerator = (factor1 * sst);
        rSquared = 1 - (enumerator / denumerator);
    } else {
        let ySquared = math.square(yMatrix.to1DArray());
        rSquared = 1 - (sse / math.sum(ySquared));
    }

    if (logging) {
        console.log("\nrsquared " + rSquared);
    }

    let result = {};
    result.mse = math.mean(squaredErrors);
    result.rSquared = rSquared;
    result.sst = sst; //squared sum of total
    result.ssr = ssr; //squared sum of regression
    result.sse = sse; //squared sum of errors
    return result;
}

/**
 * F-statistic from the regression.
 * @param {Integer} noOfCoefficients the number of coefficients for calculation the degrees of freedom.
 * @param {Integet} degreesOfFreedom the residual degrees of freedom including the intercept.
 * @param {float} sse squared sum of errors.
 * @param {float} sst squared sum of total.
 * @param {boolean} logging true for logging computational information.
 * @returns the calculated F-value
 */
function fStatistic(noOfCoefficients, degreesOfFreedom, sse, sst) {
    let fStatisticEnumerator = (sst - sse) / (noOfCoefficients - 1);
    let fStatisticDenumerator = sse / degreesOfFreedom;
    return fStatisticEnumerator / fStatisticDenumerator;
}

/**
 * Calculates the variance of the residuals (sSquare) corrected by the degrees of freedom.
 * @param {float} sse the squared sum of errors (squared sum of residuals)
 * @param {Integer} degreesOfFreedom the residual degrees of freedom including the intercept.
 * @param {Boolean} logging true for logging computational information.
 * @returns the variance of the residuals (sSquare)
 */
function varianceOfResiduals(sse, degreesOfFreedom, logging) {
    let sSquare = (1 / degreesOfFreedom) * sse;
    if (logging) {
        console.log("\nVariance of the Residuals, corrected by the degrees of freedom: " + sSquare);
    }
    return sSquare;
}

/**
 * Calculates the standard error of the coefficients.
 * @param {Matrix} pseudoInverse the pseudo inverse which was calculated in "calculateCoefficients".
 * @param {float} sSquare variance of the residuals.
 * @param {Boolean} logging true for logging computational information.
 * @returns an Array containing the standard errors.
 */
function stdErrorOfCoefficients(pseudoInverse, sSquare, logging) {
    let diag = pseudoInverse.diag();

    let stdErrorOfCoefficients = new Array(diag.length);
    for (let i = 0, len = diag.length; i < len; i++) {
        let x = diag[i] * sSquare;
        stdErrorOfCoefficients[i] = math.sqrt(x);
    }

    if (logging) {
        console.log("\nStandard Error of Coefficients: " + stdErrorOfCoefficients);
    }
    return stdErrorOfCoefficients;
}

/**
 * Calculates the T-Value for the regression coefficients and the pValues for the given tValue under consideration of the degrees of freedom.
 * @param {Matrix} coefficients The regression coefficients.
 * @param {Array} stdErrorOfCoefficients The standard error (standard deviation) of the coefficients.
 * @param {Integer} residualDegreesOfFreedom the residual degrees of freedom including the intercept.
 * @param {boolean} constantPresent true if a constant is present.
 * @param {boolean} logging true for logging computational information.
 */
function tValue(coefficients, stdErrorOfCoefficients, residualDegreesOfFreedom, constantPresent, logging) {
    //if present, correct the degrees of freedom by the constant 
    if (constantPresent) {
        residualDegreesOfFreedom++;
    }

    let tValues = new Array(coefficients.rows);
    let pValues = new Array(coefficients.rows);
    coefficients.apply((row, column) => {
        let tValue = coefficients.get(row, column) / stdErrorOfCoefficients[row];
        tValues[row] = tValue;
        pValues[row] = jstat.ttest(tValue, residualDegreesOfFreedom);
    });

    if (logging) {
        console.log("\ntValue: " + tValues);
        console.log("\npValues: " + pValues);
    }
    let result = {};
    result.tValues = tValues;
    result.pValues = pValues;
    return result;
}

/**
 * Builds a string with the coeficcients that represents the regression equation.
 * @param {Matrix} coefficients The regression coefficients.
 * @param {boolean} constantPresent Flag indicating if an intercept is present in exogen.
 */
function buildRegressionEquation(coefficients, constantPresent) {
    let equation = "y = ";
    if (constantPresent) {
        equation += coefficients.get(0, 0);
    }

    coefficients.apply((row, column) => {
        if (constantPresent && row == 0) {
            return;
        }
        let actualCoefficient = coefficients.get(row, column);
        equation += isPositive(actualCoefficient) ? " + " : "";
        equation += actualCoefficient + `x${row}`;
    });
    return equation;
}

module.exports = { linearRegression: linearRegression };