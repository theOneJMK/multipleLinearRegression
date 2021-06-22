const math = require('mathjs');
const { isInteger } = require('mathjs');
const { default: Matrix } = require('ml-matrix');

/**
 * Utility function to check wether x is an Array or a Matrix.
 * @param {any} designMatrix the value to check if it is an Array or a Matrix.
 * @returns true if designMatrix is an Array or a Matrix, false otherwise.
 */
function isArrayOrMatrix(designMatrix) {
    if (!(designMatrix instanceof Array) && !(designMatrix instanceof Matrix)) {
        return false;
    }
    return true;
}
/**
 * Adds the constant one as first column in the multidimensional Array x.
 * This may be neccessary to calculate the intercept.
 * @param {Array} designMatrix the observed vairables
 * @returns x with the added constant.
 */
function addConstant(designMatrix) {
    if (designMatrix instanceof Array) {
        let newDesignMatrix = designMatrix.slice();
        for (let i = 0; i < designMatrix.length; i++) {
            newDesignMatrix[i].unshift(1);
        }
        return newDesignMatrix;
    } else if (designMatrix instanceof Matrix) {
        let ones = Matrix.ones(designMatrix.rows, 1);
        designMatrix = designMatrix.addColumn(0, ones);
        return designMatrix;
    } else {
        new Error('designMatrix needs to be an array or a Matrix');
    }
}

/**
 * Check if all entries of the given array are ones.
 * Does not check sum == 1, because some an array of [0, 1, 1, 1] would have 1 as sum.
 * But this would not be a constant.
 * @param {Matrix} designMatrix The design matrix.
 * @returns true if a constant is present.
 */
function isConstantPresent(designMatrix) {
    if (!isArrayOrMatrix(designMatrix)) {
        new Error('designMatrix needs to be an array or a matrix');
    }
    //Transform to Matrix if neccessary
    if (designMatrix instanceof Array) {
        designMatrix = math.matrix(designMatrix);
    }

    let firstColumn = designMatrix.getColumnVector(0);
    for (let i = 0, len = firstColumn.rows; i < len; i++) {
        if (firstColumn.getRow(i)[0] != 1) {
            return false;
        }
    }
    return true;
}

/**
 * Creates a lag matrix for the given Array and the number of lags.
 * @param {Array} x the Array to create the lag matrix from.
 * @param {int} nLags the number of lags.
 * @param {boolean} logging  true for logging computational information.
 * @returns the built lag matrix.
 */
function lagMatrix(x, nLags, logging) {
    if (!isInteger(nLags) || math.isNegative(nLags)) {
        throw new Error("nlags needs to be a positive integer to calculate the lagMatrix.");
    }

    if (!(x instanceof Array)) {
        throw new Error('designMatrix needs to be an array');
    }

    if (x.length <= nLags) {
        throw new Error("the length of x must be greater than nlags.");
    }

    x = x.slice();//copy x to let the parameter untouched.
    for (let i = 0; i < nLags; i++) {
        x.unshift(0);
    }

    let lagMatrix = Matrix.zeros(x.length - nLags, nLags);
    for (let i = 1; i < lagMatrix.rows; i++) {
        let kRow = x.slice(i, i + nLags).reverse();
        lagMatrix.setRow(i, kRow);
    }
    if (logging) {
        console.log(lagMatrix);
    }
    return lagMatrix;
}

/**
 * Calculates the degrees of freedom (dof).
 * This dof includes the constant.
 * @param {int} noOfObservations The number of observed values.
 * @param {int} noOfCoefficients The number of used coefficients.
 * @param {boolean} logging true for logging computational information.
 * @returns the calculatesd degrees of freedom (dof)
 */
function degreesOfFreedom(noOfObservations, noOfCoefficients, logging) {
    let degreesOfFreedom = noOfObservations - noOfCoefficients;
    if (logging) {
        console.log("\ndegrees of Freedom: " + degreesOfFreedom);
    }
    return degreesOfFreedom;
}

module.exports = { isArrayOrMatrix: isArrayOrMatrix, degreesOfFreedom: degreesOfFreedom, lagMatrix: lagMatrix, addConstant: addConstant, isConstantPresent: isConstantPresent };
