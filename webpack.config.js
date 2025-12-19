/**
 * Webpack configuration for Greed.js v2.0 - Single unified build
 * Modular architecture with full backward compatibility
 */
const path = require('path');

// Create two separate configs: one for main bundle, one for worker
const mainConfig = {
  entry: {
    'greed': './src/core/greed-v2.js',
    'greed.min': './src/core/greed-v2.js'
  },

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js',
    library: {
      name: 'Greed',
      type: 'umd',
      export: 'default',
      umdNamedDefine: true
    },
    globalObject: 'globalThis',
    clean: false // Don't clean - let webpack clean plugin handle it
  },

  optimization: {
    // Disable code splitting for npm package compatibility
    // All code will be bundled into a single file
    splitChunks: false,
    usedExports: true,
    sideEffects: false,
    minimize: true
  },

  resolve: {
    extensions: ['.js', '.mjs'],
    alias: {
      '@core': path.resolve(__dirname, 'src/core'),
      '@compute': path.resolve(__dirname, 'src/compute'),
      '@utils': path.resolve(__dirname, 'src/utils')
    }
  },

  module: {
    rules: [
      {
        test: /\.m?js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', {
                targets: {
                  browsers: ['> 1%', 'last 2 versions', 'not dead']
                },
                modules: false, // Let webpack handle modules for tree shaking
                useBuiltIns: 'usage',
                corejs: 3
              }]
            ]
          }
        }
      }
    ]
  },

  externals: {
    // Pyodide should be loaded separately
    'pyodide': {
      commonjs: 'pyodide',
      commonjs2: 'pyodide',
      amd: 'pyodide',
      root: 'loadPyodide'
    }
  },

  plugins: [],

  devtool: process.env.NODE_ENV === 'development' ? 'source-map' : false,

  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    compress: true,
    port: 8080,
    hot: true,
    open: true,
    headers: {
      // Security headers for development
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  },

  performance: {
    hints: 'warning',
    maxAssetSize: 500000, // 500KB per asset
    maxEntrypointSize: 1000000 // 1MB total
  },

  target: ['web', 'es2020']
};

// Worker configuration - separate bundle without UMD wrapper
const workerConfig = {
  entry: {
    'pyodide-worker': './src/compute/worker/pyodide-worker.js'
  },

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js',
    globalObject: 'self', // Workers use 'self' instead of 'window'
    clean: false // Don't clean, main config already does that
  },

  optimization: {
    splitChunks: false,
    minimize: true
  },

  resolve: {
    extensions: ['.js', '.mjs']
  },

  devtool: process.env.NODE_ENV === 'development' ? 'source-map' : false,

  target: ['webworker', 'es2020']
};

// Export both configurations
module.exports = [mainConfig, workerConfig];