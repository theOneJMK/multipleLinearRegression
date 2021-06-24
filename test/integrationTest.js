import { test } from 'tape';
import { readFileSync} from 'fs';// file system
import { convertCSVToArray } from 'convert-csv-to-array';
import { linearRegression } from '../js/linear/linearRegression.js';
import { addConstant } from '../js/tools/tools.js';
import { ramsey } from '../js/specification/specification.js';
import { goldfeldQuandt } from '../js/heteroskedasticity/heteroskedasticity.js';
import { durbinWatson, breuschGodfrey } from '../js/autocorrelation/autocorrelation.js';
import { varianceInflationFactors } from '../js/multicollinearity/multicollinearity.js';
import { round } from 'mathjs';
import { Matrix } from 'ml-matrix';
import { URL } from 'url'; 
const __dirname = new URL('.', import.meta.url).pathname;

const testFolder = 'data/';

/**
 * Integration test for the whole library.
 * It reads the input from 'data/marketingData.csv' and computes a multiple linerat regression.
 * The result is the compared to the results fom the python library 'stats-model', which is stored in  'data/marketingPythonResult.json'
 */
test('integrationTest', function (t) {
    let testdata;
    try {
        let data = readFileSync(__dirname + testFolder + 'marketingData.csv');
        testdata = convertCSVToArray(data.toString(), { header: true, type: 'array', seperator: ',' });
    } catch (err) {
        console.log(err);
        t.fail('failed to read testdata.');
        t.end()
    }

    testdata.shift();//remove first row
    let testMatrix = new Matrix(testdata);
    let testY = testMatrix.getColumnVector(3);
    let testX = testMatrix.subMatrixColumn([0, 1, 2]);
    testX = addConstant(testX);
    let regressionResult = linearRegression(testY, testX, false);
    let compareResults = readPyhtonResultFromJson(t);

    t.comment('testing regression results:');
    t.looseEqual(regressionResult.noOfObservations, compareResults['noOfObservations']);
    t.deepLooseEqual(round(regressionResult.coefficients.to1DArray(), 10),
        round(compareResults['coefficients'], 10));
    t.looseEqual(regressionResult.noOfCoefficients, compareResults['noOfCoefficients']);
    t.deepLooseEqual(round(regressionResult.stdErrorOfCoefficients, 10),
        round(compareResults['stdErrorOfCoefficients'], 10));
    t.deepLooseEqual(round(regressionResult.tValues, 10),
        round(compareResults['tValues'], 10));
    t.deepLooseEqual(round(regressionResult.pValues, 6),
        round(compareResults['pValues'], 6));
    t.deepLooseEqual(round(regressionResult.predicted, 10),
        round(compareResults['predicted'], 10));
    t.deepLooseEqual(round(regressionResult.residuals, 10),
        round(compareResults['residuals'], 10));
    t.looseEqual(round(regressionResult.sSquare, 10),
        round(compareResults['sSquare'], 10));
    t.looseEqual(round(regressionResult.rSquared, 10),
        round(compareResults['rSquared'], 10));
    t.looseEqual(round(regressionResult.fValue, 10),
        round(compareResults['fValue'], 10));
    t.looseEqual(round(regressionResult.pValueOfFValue, 10),
        round(compareResults['pValueOfFValue'], 10));
    t.looseEqual(regressionResult.modelDegreesOfFreedom,
        round(compareResults['modelDegreesOfFreedom'], 10));
    t.looseEqual(regressionResult.residualDegreesOfFreedom,
        round(compareResults['residualDegreesOfFreedom'], 10));
    t.looseEqual(round(regressionResult.sse, 10),
        round(compareResults['sse'], 10));
    t.looseEqual(round(regressionResult.sst, 10),
        round(compareResults['sst'], 10));
    t.looseEqual(round(regressionResult.ssr, 8),
        round(compareResults['ssr'], 8));
    t.looseEqual(round(regressionResult.sse / regressionResult.residualDegreesOfFreedom, 10),
        round(compareResults['mseResid'], 10));
    t.looseEqual(round(regressionResult.sst / (regressionResult.residualDegreesOfFreedom + regressionResult.modelDegreesOfFreedom), 10),
        round(compareResults['mseTotal'], 10));
    t.looseEqual(round(regressionResult.ssr / regressionResult.modelDegreesOfFreedom, 9),
        round(compareResults['mseModel'], 9));
    t.comment("regression results ok!");

    //residual diagnostik
    testResidualDiagnostics(t, testY, testX, regressionResult, compareResults['residualDiagnostic']);
    t.end()
})

function testResidualDiagnostics(t, testY, testX, regressionResult, compareResidualDiagnostic) {
    t.comment('testing residual diagnostics');

    t.comment('testing RESET');
    let compareReset = compareResidualDiagnostic['RESET'];
    let ramseyResult = ramsey(regressionResult, compareReset['resetPower'], false);
    t.looseEqual(round(ramseyResult.fValue, 10),
        round(compareReset['fValue'], 10));
    t.looseEqual(round(ramseyResult.pValue, 10),
        round(compareReset['pValue'], 10));

    t.comment('testing Goldfeld-Quandt for homoskedasticy');
    let compareGQ = compareResidualDiagnostic['Goldfeld-Quandt'];
    let gqResult = goldfeldQuandt(testY, testX, null, 0, false);
    t.looseEqual(round(gqResult.fValue, 9),
        round(compareGQ['fValue'], 9));
    t.looseEqual(round(gqResult.pValue, 10),
        round(compareGQ['pValue'], 10));

    t.comment('testing Durbin-Watson for first order autocrorrelation');
    let dwResult = durbinWatson(regressionResult, false);
    t.looseEqual(round(dwResult, 10),
        round(compareResidualDiagnostic['Durbin-Watson'], 10));

    t.comment('testing Breusch-Godfrey');
    let compareBG = compareResidualDiagnostic['Breusch-Godfrey'];
    let bgResult = breuschGodfrey(regressionResult, compareBG['bgNLags'], false);
    t.looseEqual(round(bgResult.lagrangeMultiplier, 10),
        round(compareBG['lagrangeMultiplier'], 10));
    t.looseEqual(round(bgResult.lagrangeMultiplierPValue, 10),
        round(compareBG['lagrangeMultiplierPValue'], 10));
    t.looseEqual(round(bgResult.fValue, 10),
        round(compareBG['fValue'], 10));
    t.looseEqual(round(bgResult.pValue, 10),
        round(compareBG['pValue'], 10));

    t.comment('testing VIFs');
    let compareVif = compareResidualDiagnostic['VIF'];
    let vifResult = varianceInflationFactors(testX, false);
    let vifResultArray = vifResult.map(vif => vif.vif);
    t.deepLooseEqual(round(vifResultArray, 10), round(compareVif, 10));
}

function readPyhtonResultFromJson(t) {
    let data;
    try {
        data = readFileSync(__dirname + testFolder + 'marketingPythonResult.json');
    } catch (err) {
        console.log(err);
        t.fail('failed to read compare data.');
        t.end();
    }
    return JSON.parse(data);
}