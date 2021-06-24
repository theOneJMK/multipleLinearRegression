import { linearRegression } from '../linear/linearRegression.js';
import { isInteger, isPositive } from 'mathjs';
import { Matrix } from 'ml-matrix';
import jstat from 'jstat';
const { ftest } = jstat;

class GoldfeldQuandtResult {

    /**
     * The F=Value of the Goldfeld-Quandt test
     */
    fValue

    /**
     * The corresponding P-Value for the F-Value
     */
    pValue

    /**
     * The Index on which the data was sorted, if defined.
     */
    sortIndex

    /**
     * The used offset for splitting the dataset
     */
    split

    /**
     * constructor
     */
    GoldfeldQuandtResult() {
    }
}
/**
 * Calculates the Goldfeld-Quandt statistic for homoskedasticity to find out wether the regression has homo- or heteroskedasticy.
 * 
 * @param {Array|Matrix} bgY The observed y values.
 * @param {Array|Matrix} bgX The design matix.
 * @param {Integer} sortIndex The index on which the design matrix will be sorted, null if sorting is not neccessary.
 * @param {Integer} split Offset for the splitting.
 *  If 0 or not provided the dataset will be split in the middle of the observations.
 *  Otherwise the offset dertmines how many observations on the left and right side of the middle are skipped to perofrm the test.
 */
function goldfeldQuandt(y, x, sortIndex, split, logging) {
    if (!(y instanceof Matrix) && !(y instanceof Array)) {
        throw new Error("y needs to be Matrix or an array");
    }
    if (!(x instanceof Matrix) && !(x instanceof Array)) {
        throw new Error("x needs to be Matrix or an array");
    }

    if (sortIndex != null && isInteger(sortIndex) && isPositive(sortIndex)) {
        //When y is a matrix, make it an array for sorting
        if (y instanceof Matrix) {
            y = y.to1DArray();
        }
        //When x is a matrix, make it an array for sorting
        if (x instanceof Matrix) {
            x = x.to2DArray();
        }
        sortDesignMatrix(x, y, sortIndex);
    } else {
        console.log("No numerical index for sorting is given.");
    }

    let gqY = new Matrix(y);
    if (gqY.columns != 1) {
        throw new Error("y needs to be a vector (1 dimensional math.Matrix)");
    }

    let gqX = new Matrix(x);
    if (gqX.rows < 2) {
        throw new Error("At least 2 observations must be given for the Goldfeld-Quandt test.");
    }

    let offset = 0;
    let splitNumberOfObservations = Math.floor(gqY.rows / 2);
    if (split && isInteger(split) && isPositive(split)) {
        offset = split;
        if (splitNumberOfObservations - offset < 0 || splitNumberOfObservations + offset > gqY.rows - 1) {
            throw new Error("The offset exceeds the dataset. The Goldfeld-Quandt-Test can therefor not be executed.");
        }
    } else {
        console.log("Splitting in the middle since split is not given.");
    }

    let firstSubsetSplitter = splitNumberOfObservations - offset - 1;
    let secondSubsetSplitter = splitNumberOfObservations + offset;

    let y1 = gqY.subMatrixColumn([0], 0, firstSubsetSplitter);
    let y2 = gqY.subMatrixColumn([0], secondSubsetSplitter, gqY.rows - 1);

    let x1 = gqX.subMatrix(0, firstSubsetSplitter, 0, gqX.columns - 1);
    let x2 = gqX.subMatrix(secondSubsetSplitter, gqX.rows - 1, 0, gqX.columns - 1);

    let result1 = linearRegression(y1, x1, logging);
    let result2 = linearRegression(y2, x2, logging);

    if (logging) {
        console.log("1st SSE: " + result1.sse);
        console.log("2nd SSE: " + result2.sse);
    }

    let fValDividend = result2.sse / result2.residualDegreesOfFreedom;
    let fValDivisor = result1.sse / result1.residualDegreesOfFreedom;
    let fVal = fValDividend / fValDivisor;
    let pVal = ftest(fVal, result1.residualDegreesOfFreedom, result2.residualDegreesOfFreedom);
    if (logging) {
        console.log(`fVal:${fVal}, pval:${pVal}, order:increasing`);
    }

    let result = new GoldfeldQuandtResult();
    result.fValue = fVal;
    result.pValue = pVal;
    result.sortIndex = sortIndex;
    result.split = split;
    return result;
}
/**
 * Sorts the design matrix and the observed y values on the given index.
 * @param {Array} x the design matix as array for sorting purposes
 * @param {Array} y the observed y values as array
 * @param {int} sortIndex the index on which the design matrix will be sorted
 */
function sortDesignMatrix(x, y, sortIndex) {
    //Add y to the two dimensional x Array
    for (let i = 0; i < y.length; i++) {
        x[i].push(y[i]);
    }

    //Sort x
    x = x.sort(function (s1, s2) {
        return s1[sortIndex] - s2[sortIndex];
    });

    //Pull y out of x
    for (let i = 0; i < y.length; i++) {
        y[i] = x[i][x[i].length - 1];
        x[i].pop();
    }
}

export { GoldfeldQuandtResult, goldfeldQuandt };