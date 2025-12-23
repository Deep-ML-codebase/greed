module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
    jest: true
  },
  extends: [
    'eslint:recommended'
  ],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  },
  rules: {
    'no-unused-vars': ['error', { 'argsIgnorePattern': '^_', 'varsIgnorePattern': '^_' }],
    'no-console': 'off', // We have proper logging in place
    'prefer-const': 'error',
    'no-var': 'error',
    'semi': ['error', 'always'],
    'quotes': ['error', 'single', { 'allowTemplateLiterals': true }],
    'no-case-declarations': 'off', // Allow lexical declarations in case blocks
    'no-useless-catch': 'off', // Allow try/catch for clarity
    'no-empty': ['error', { 'allowEmptyCatch': true }], // Allow empty catch blocks
    'no-useless-escape': 'off' // Allow escapes in Python code strings
  },
  globals: {
    'GPUBufferUsage': 'readonly',
    'GPUShaderStage': 'readonly',
    'GPUMapMode': 'readonly',
    'navigator': 'readonly',
    'performance': 'readonly',
    'window': 'readonly',
    'loadPyodide': 'readonly',
    'importScripts': 'readonly',
    'self': 'readonly'
  }
};