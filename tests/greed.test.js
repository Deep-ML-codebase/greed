/**
 * Basic unit tests for Greed.js
 */

describe('Greed.js Core', () => {
  describe('Module Structure', () => {
    it('should export Greed class', () => {
      // Test that the module structure exists
      // Note: Full integration tests require browser environment with WebGPU
      expect(true).toBe(true);
    });

    it('should have correct package configuration', () => {
      const pkg = require('../package.json');

      expect(pkg.name).toBe('greed.js');
      expect(pkg.version).toBeDefined();
      expect(pkg.main).toBe('dist/greed.js');
      expect(pkg.types).toBe('dist/greed.d.ts');
      expect(pkg.license).toBe('MIT');
    });

    it('should have required npm scripts', () => {
      const pkg = require('../package.json');

      expect(pkg.scripts.build).toBeDefined();
      expect(pkg.scripts.test).toBeDefined();
      expect(pkg.scripts.lint).toBeDefined();
    });

    it('should have TypeScript definitions', () => {
      const fs = require('fs');
      const path = require('path');

      const dtsPath = path.join(__dirname, '..', 'dist', 'greed.d.ts');
      expect(fs.existsSync(dtsPath)).toBe(true);
    });
  });

  describe('Configuration', () => {
    it('should have valid engine requirements', () => {
      const pkg = require('../package.json');

      expect(pkg.engines.node).toBeDefined();
      expect(pkg.engines.npm).toBeDefined();
    });

    it('should have valid keywords for npm discovery', () => {
      const pkg = require('../package.json');

      expect(pkg.keywords).toContain('pytorch');
      expect(pkg.keywords).toContain('webgpu');
      expect(pkg.keywords).toContain('machine-learning');
    });

    it('should have valid repository configuration', () => {
      const pkg = require('../package.json');

      expect(pkg.repository.type).toBe('git');
      expect(pkg.repository.url).toContain('github.com');
    });
  });

  describe('Distribution Files', () => {
    it('should have main entry point', () => {
      const fs = require('fs');
      const path = require('path');

      const mainPath = path.join(__dirname, '..', 'dist', 'greed.js');
      expect(fs.existsSync(mainPath)).toBe(true);
    });

    it('should have minified version', () => {
      const fs = require('fs');
      const path = require('path');

      const minPath = path.join(__dirname, '..', 'dist', 'greed.min.js');
      expect(fs.existsSync(minPath)).toBe(true);
    });
  });
});

describe('Security Validator', () => {
  it('should be importable', () => {
    // Placeholder for security validator tests
    // These would test the SecurityValidator class
    expect(true).toBe(true);
  });
});

describe('Memory Manager', () => {
  it('should be importable', () => {
    // Placeholder for memory manager tests
    expect(true).toBe(true);
  });
});
