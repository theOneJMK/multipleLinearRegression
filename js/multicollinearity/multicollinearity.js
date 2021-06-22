const math = require('mathjs');
const { default: Matrix } = require('ml-matrix');
const linReg = require('../linear/linearRegression.js');

class VIFResult {

    /**
     * The Variance Inflation Factor
     */
    vif;

    /**
     * The corresponding rSquared
     */
    rSquared;

    /**
     * The corresponding regression equation
     */
    regressionEquation;

    VIFResult() {

    }
}
/**
 * Calculates the VIFs (variance inflation factors) for the whole design matrix x.
 * @param {Array|Matrix} x The design matrix of the regression.
 * @returns An Array containing the VIFs.
 */
function varianceInflationFactors(x, logging) {
    switch (true) {
        case x instanceof Array:
            x = new Matrix(x);
            break;
        case x instanceof Matrix:
            break;
        default:
            throw new Error("x needs to be math.Matrix or an array");
    }

    let vIFResult = [];
    let noOfColumns = x.columns;
    let columnIndexes = math.range(0, x.columns).toArray();
    for (let i = 0; i < noOfColumns; i++) {
        let endogenVector = x.getColumnVector(i);

        let indexes = columnIndexes.filter((columnIndex) => {
            return columnIndex != i;
        });
        let exogen = x.subMatrixColumn(indexes, 0, x.rows - 1);

        let rSquared =
            linReg.linearRegression(endogenVector, exogen).rSquared;
        let vif = 1 / (1 - rSquared);

        if (logging) {
            console.log(`\nVIF ${i}: ${vif}, rSquare: ${rSquared}`);
        }

        let result = new VIFResult();
        result.vif = vif;
        result.rSquared = rSquared;
        //result.regressionEquation = regressionResult.regressionEquation;

        vIFResult.push(result);
    }
    return vIFResult;
}

module.exports = { varianceInflationFactors: varianceInflationFactors };