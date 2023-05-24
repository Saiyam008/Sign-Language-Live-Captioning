const path = require('path');

module.exports = {
  entry: './popup.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js',
  },
};
