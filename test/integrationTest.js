const test = require('tape');
const fs = require('fs');// file system
const csvToArray = require('convert-csv-to-array');
const linReg = require('../js/linear/linearRegression');
const regressionTools = require('../js/tools/tools');
const regressionSpecification = require('../js/specification/specification');
const heteroskedasticy = require('../js/heteroskedasticity/heteroskedasticity');
const autocorrelation = require('../js/autocorrelation/autocorrelation');
const multicollinearity = require('../js/multicollinearity/multicollinearity');
const math = require('mathjs');
const { Matrix } = require('ml-matrix');

const testFolder = '/data/';

/**
 * Integration test for the whole library.
 * It reads the input from 'data/marketingData.csv' and computes a multiple linerat regression.
 * The result is the compared to the results fom the python library 'stats-model', which is stored in  'data/marketingPythonResult.json'
 */
test('integrationTest', function (t) {
    let testdata;
    try {
        console.log(__dirname);
        let data = fs.readFileSync(__dirname + testFolder + 'marketingData.csv');
        testdata = csvToArray.convertCSVToArray(data.toString(), { header: true, type: 'array', seperator: ',' });
    } catch (err) {
        console.log(err);
        t.fail('failed to read testdata.');
        t.end()
    }

    testdata.shift();//remove first row
    let testMatrix = new Matrix(testdata);
    let testY = testMatrix.getColumnVector(3);
    let testX = testMatrix.subMatrixColumn([0, 1, 2]);
    testX = regressionTools.addConstant(testX);
    let regressionResult = linReg.linearRegression(testY, testX, false);
    let compareResults = readPyhtonResultFromJson(t);

    t.comment('testing regression results:');
    t.looseEqual(regressionResult.noOfObservations, compareResults['noOfObservations']);
    t.deepLooseEqual(math.round(regressionResult.coefficients.to1DArray(), 10),
        math.round(compareResults['coefficients'], 10));
    t.looseEqual(regressionResult.noOfCoefficients, compareResults['noOfCoefficients']);
    t.deepLooseEqual(math.round(regressionResult.stdErrorOfCoefficients, 10),
        math.round(compareResults['stdErrorOfCoefficients'], 10));
    t.deepLooseEqual(math.round(regressionResult.tValues, 10),
        math.round(compareResults['tValues'], 10));
    t.deepLooseEqual(math.round(regressionResult.pValues, 6),
        math.round(compareResults['pValues'], 6));
    t.deepLooseEqual(math.round(regressionResult.predicted, 10),
        math.round(compareResults['predicted'], 10));
    t.deepLooseEqual(math.round(regressionResult.residuals, 10),
        math.round(compareResults['residuals'], 10));
    t.looseEqual(math.round(regressionResult.sSquare, 10),
        math.round(compareResults['sSquare'], 10));
    t.looseEqual(math.round(regressionResult.rSquared, 10),
        math.round(compareResults['rSquared'], 10));
    t.looseEqual(math.round(regressionResult.fValue, 10),
        math.round(compareResults['fValue'], 10));
    t.looseEqual(math.round(regressionResult.pValueOfFValue, 10),
        math.round(compareResults['pValueOfFValue'], 10));
    t.looseEqual(regressionResult.modelDegreesOfFreedom,
        math.round(compareResults['modelDegreesOfFreedom'], 10));
    t.looseEqual(regressionResult.residualDegreesOfFreedom,
        math.round(compareResults['residualDegreesOfFreedom'], 10));
    t.looseEqual(math.round(regressionResult.sse, 10),
        math.round(compareResults['sse'], 10));
    t.looseEqual(math.round(regressionResult.sst, 10),
        math.round(compareResults['sst'], 10));
    t.looseEqual(math.round(regressionResult.ssr, 8),
        math.round(compareResults['ssr'], 8));
    t.looseEqual(math.round(regressionResult.sse / regressionResult.residualDegreesOfFreedom, 10),
        math.round(compareResults['mseResid'], 10));
    t.looseEqual(math.round(regressionResult.sst / (regressionResult.residualDegreesOfFreedom + regressionResult.modelDegreesOfFreedom), 10),
        math.round(compareResults['mseTotal'], 10));
    t.looseEqual(math.round(regressionResult.ssr / regressionResult.modelDegreesOfFreedom, 9),
        math.round(compareResults['mseModel'], 9));
    t.comment("regression results ok!");

    //residual diagnostik
    testResidualDiagnostics(t, testY, testX, regressionResult, compareResults['residualDiagnostic']);
    t.end()
})

function testResidualDiagnostics(t, testY, testX, regressionResult, compareResidualDiagnostic) {
    t.comment('testing residual diagnostics');

    t.comment('testing RESET');
    let compareReset = compareResidualDiagnostic['RESET'];
    let ramseyResult = regressionSpecification.ramsey(regressionResult, compareReset['resetPower'], false);
    t.looseEqual(math.round(ramseyResult.fValue, 10),
        math.round(compareReset['fValue'], 10));
    t.looseEqual(math.round(ramseyResult.pValue, 10),
        math.round(compareReset['pValue'], 10));

    t.comment('testing Goldfeld-Quandt for homoskedasticy');
    let compareGQ = compareResidualDiagnostic['Goldfeld-Quandt'];
    let gqResult = heteroskedasticy.goldfeldQuandt(testY, testX, null, 0, false);
    t.looseEqual(math.round(gqResult.fValue, 9),
        math.round(compareGQ['fValue'], 9));
    t.looseEqual(math.round(gqResult.pValue, 10),
        math.round(compareGQ['pValue'], 10));

    t.comment('testing Durbin-Watson for first order autocrorrelation');
    let dwResult = autocorrelation.durbinWatson(regressionResult, false);
    t.looseEqual(math.round(dwResult, 10),
        math.round(compareResidualDiagnostic['Durbin-Watson'], 10));

    t.comment('testing Breusch-Godfrey');
    let compareBG = compareResidualDiagnostic['Breusch-Godfrey'];
    let bgResult = autocorrelation.breuschGodfrey(regressionResult, compareBG['bgNLags'], false);
    t.looseEqual(math.round(bgResult.lagrangeMultiplier, 10),
        math.round(compareBG['lagrangeMultiplier'], 10));
    t.looseEqual(math.round(bgResult.lagrangeMultiplierPValue, 10),
        math.round(compareBG['lagrangeMultiplierPValue'], 10));
    t.looseEqual(math.round(bgResult.fValue, 10),
        math.round(compareBG['fValue'], 10));
    t.looseEqual(math.round(bgResult.pValue, 10),
        math.round(compareBG['pValue'], 10));

    t.comment('testing VIFs');
    let compareVif = compareResidualDiagnostic['VIF'];
    let vifResult = multicollinearity.varianceInflationFactors(testX, false);
    let vifResultArray = vifResult.map(vif => vif.vif);
    t.deepLooseEqual(math.round(vifResultArray, 10), math.round(compareVif, 10));
}

function readPyhtonResultFromJson(t) {
    let data;
    try {
        data = fs.readFileSync(__dirname + testFolder + 'marketingPythonResult.json');
    } catch (err) {
        console.log(err);
        t.fail('failed to read compare data.');
        t.end();
    }
    return JSON.parse(data);
}