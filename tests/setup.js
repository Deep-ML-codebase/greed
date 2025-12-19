/**
 * Jest setup file for Greed.js tests
 */

// Mock WebGPU API for Node.js environment
global.navigator = {
  gpu: null,
  hardwareConcurrency: 4
};

// Mock performance API if not available
if (typeof performance === 'undefined') {
  global.performance = {
    now: () => Date.now()
  };
}

// Mock console methods for cleaner test output
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

beforeAll(() => {
  // Suppress Greed.js ASCII art and initialization logs during tests
  console.log = jest.fn((...args) => {
    const message = args.join(' ');
    // Only suppress Greed.js banner
    if (!message.includes('GREED.JS') && !message.includes('╔═')) {
      originalConsoleLog(...args);
    }
  });
});

afterAll(() => {
  console.log = originalConsoleLog;
  console.warn = originalConsoleWarn;
  console.error = originalConsoleError;
});

// Increase timeout for async operations
jest.setTimeout(30000);
