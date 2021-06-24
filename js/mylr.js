var math = require('mathjs');
import { multicollinearity } from './multicollinearity/multicollinearity.js';
import { autocorrelation } from './autocorrelation/autocorrelation.js';
import { specification } from './specification/specification.js';
import { homoskedasticty } from './heteroskedasticy/heteroskedasticy.js';
import { linReg } from './linear/linearRegression.js';
import { tools } from './tools/tools.js';

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