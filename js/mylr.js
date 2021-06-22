var math = require('mathjs');
var jStat = require('jstat');
var multicollinearity = require('./multicollinearity/multicollinearity.js');
var autocorrelation = require('./autocorrelation/autocorrelation.js');
var specification = require('./specification/specification.js');
var homoskedasticty = require('./heteroskedasticy/heteroskedasticy.js');
var linReg = require('./linear/linearRegression.js');
var tools = require('./tools/tools.js');

let x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]];
let y = [4, 5, 20, 14, 32, 22, 38, 43];

//add a 1 at beginning of every y for calculating the constant coefficient.
x = tools.addConstant(x);

//calculate variance inflation factors
vifs = multicollinearity.varianceInflationFactors(math.matrix(x), true);
console.log(`VIFs:\n${vifs}`);

let regressionResult = linReg.linearRegression(y, x, false);

autocorrelation.durbinWatson(regressionResult, true);
autocorrelation.breuschGodfrey(regressionResult, 3, true);

console.log("\nGoldfeld-Quandt:");
homoskedasticty.goldfeldQuandt(y, x, null, null, true);

specification.ramsey(regressionResult, 4, false)