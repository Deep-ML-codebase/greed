/**
 * SecurityValidator - Comprehensive security validation for Python code and tensor operations
 * Prevents code injection, validates inputs, and enforces security policies
 */
import EventEmitter from '../core/event-emitter.js';

class SecurityValidator extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      // Security levels
      strictMode: config.strictMode !== false,
      allowEval: config.allowEval === true,
      allowFileSystem: config.allowFileSystem === true,
      allowNetwork: config.allowNetwork === true,
      allowSubprocess: config.allowSubprocess === true,
      
      // Input validation limits
      maxCodeLength: config.maxCodeLength || 100000, // 100KB
      maxTensorSize: config.maxTensorSize || 100_000_000, // 100M elements
      maxTensorCount: config.maxTensorCount || 100,
      maxStringLength: config.maxStringLength || 10000,
      
      // Resource limits
      maxMemoryMB: config.maxMemoryMB || 1024,
      maxExecutionTimeMs: config.maxExecutionTimeMs || 30000,
      
      // Package whitelist
      allowedPackages: new Set(config.allowedPackages || [
        'numpy', 'math', 'random', 'json', 'datetime', 'collections',
        'itertools', 'functools', 'operator', 're', 'scipy', 'matplotlib',
        'pandas', 'scikit-learn', 'sklearn', 'plotly', 'seaborn',
        'statsmodels', 'sympy', 'networkx'
      ]),
      
      // Custom validation patterns
      customDangerousPatterns: config.customDangerousPatterns || [],
      customAllowedPatterns: config.customAllowedPatterns || [],
      
      ...config
    };

    // Security pattern definitions
    this.dangerousPatterns = this._initializeDangerousPatterns();
    this.suspiciousPatterns = this._initializeSuspiciousPatterns();
    this.fileSystemPatterns = this._initializeFileSystemPatterns();
    this.networkPatterns = this._initializeNetworkPatterns();
    
    // Validation statistics
    this.stats = {
      totalValidations: 0,
      blockedOperations: 0,
      warningsIssued: 0,
      lastValidation: null,
      threatCategories: {}
    };
  }

  /**
   * Validate Python code for security threats
   */
  validatePythonCode(code, options = {}) {
    const startTime = performance.now();
    this.stats.totalValidations++;
    
    try {
      this.emit('validation:start', { 
        codeLength: code.length, 
        strict: this.config.strictMode 
      });

      // Basic input validation
      this._validateCodeInput(code);
      
      // Security pattern analysis
      const threats = this._analyzeSecurityThreats(code);
      
      // Risk assessment
      const riskLevel = this._assessRiskLevel(threats);
      
      // Policy enforcement
      const result = this._enforceSecurityPolicy(code, threats, riskLevel, options);
      
      // Update statistics
      this._updateValidationStats(threats, riskLevel, result);
      
      const validationTime = performance.now() - startTime;
      this.emit('validation:complete', {
        threats: threats.length,
        riskLevel,
        allowed: result.allowed,
        validationTime
      });
      
      return result;
    } catch (error) {
      this.emit('validation:error', { error, codePreview: code.substring(0, 100) });
      throw error;
    }
  }

  /**
   * Validate tensor data for security and resource constraints
   */
  validateTensorData(tensors, options = {}) {
    const tensorArray = Array.isArray(tensors) ? tensors : [tensors];
    
    this.emit('tensor:validation:start', { tensorCount: tensorArray.length });
    
    try {
      // Count and size validation
      if (tensorArray.length > this.config.maxTensorCount) {
        throw new SecurityError(`Too many tensors: ${tensorArray.length} > ${this.config.maxTensorCount}`);
      }
      
      let totalElements = 0;
      let totalMemoryMB = 0;
      
      for (let i = 0; i < tensorArray.length; i++) {
        const tensor = tensorArray[i];
        
        // Type validation
        if (!this._isValidTensorType(tensor)) {
          throw new SecurityError(`Invalid tensor type at index ${i}: ${typeof tensor}`);
        }
        
        // Size validation
        const elementCount = this._getTensorElementCount(tensor);
        if (elementCount > this.config.maxTensorSize) {
          throw new SecurityError(`Tensor too large at index ${i}: ${elementCount} > ${this.config.maxTensorSize}`);
        }
        
        // Memory estimation
        const memoryMB = this._estimateTensorMemoryMB(tensor);
        totalMemoryMB += memoryMB;
        totalElements += elementCount;
        
        this.emit('tensor:validated', { 
          index: i, 
          elements: elementCount, 
          memoryMB: Math.round(memoryMB * 100) / 100 
        });
      }
      
      // Total resource validation
      if (totalMemoryMB > this.config.maxMemoryMB) {
        throw new SecurityError(`Total tensor memory too large: ${Math.round(totalMemoryMB)}MB > ${this.config.maxMemoryMB}MB`);
      }
      
      this.emit('tensor:validation:complete', {
        tensorCount: tensorArray.length,
        totalElements,
        totalMemoryMB: Math.round(totalMemoryMB * 100) / 100
      });
      
      return {
        valid: true,
        tensorCount: tensorArray.length,
        totalElements,
        totalMemoryMB,
        warnings: []
      };
    } catch (error) {
      this.emit('tensor:validation:error', { error });
      throw error;
    }
  }

  /**
   * Validate operation parameters and options
   */
  validateOperation(operation, params, options = {}) {
    this.emit('operation:validation:start', { operation });
    
    try {
      // Operation name validation
      if (!operation || typeof operation !== 'string') {
        throw new SecurityError('Operation must be a non-empty string');
      }
      
      if (operation.length > 100) {
        throw new SecurityError(`Operation name too long: ${operation.length} > 100`);
      }
      
      // Check for dangerous operation patterns
      const dangerousOps = ['eval', 'exec', 'compile', '__import__', 'subprocess'];
      if (dangerousOps.some(danger => operation.toLowerCase().includes(danger))) {
        throw new SecurityError(`Dangerous operation detected: ${operation}`);
      }
      
      // Validate parameters
      const validatedParams = this._validateOperationParams(params);
      
      // Validate options
      const validatedOptions = this._validateOperationOptions(options);
      
      this.emit('operation:validation:complete', { 
        operation, 
        paramCount: Object.keys(validatedParams).length 
      });
      
      return {
        valid: true,
        operation,
        params: validatedParams,
        options: validatedOptions
      };
    } catch (error) {
      this.emit('operation:validation:error', { operation, error });
      throw error;
    }
  }

  /**
   * Validate URL for safe external requests
   */
  validateURL(url, options = {}) {
    const { allowedDomains = [], blockedDomains = [], requireHTTPS = true } = options;
    
    try {
      const urlObj = new URL(url);
      
      // Protocol validation
      if (requireHTTPS && urlObj.protocol !== 'https:') {
        throw new SecurityError(`HTTPS required for URL: ${url}`);
      }
      
      // Domain validation
      if (blockedDomains.includes(urlObj.hostname)) {
        throw new SecurityError(`Blocked domain: ${urlObj.hostname}`);
      }
      
      if (allowedDomains.length > 0 && !allowedDomains.includes(urlObj.hostname)) {
        throw new SecurityError(`Domain not in allowlist: ${urlObj.hostname}`);
      }
      
      // Suspicious pattern detection
      const suspiciousPatterns = [
        /localhost/i, /127\.0\.0\.1/, /0\.0\.0\.0/, /\[::\]/,
        /internal/i, /private/i, /admin/i, /\.local$/i
      ];
      
      if (suspiciousPatterns.some(pattern => pattern.test(url))) {
        throw new SecurityError(`Suspicious URL pattern detected: ${url}`);
      }
      
      return { valid: true, url: urlObj.href, domain: urlObj.hostname };
    } catch (error) {
      if (error instanceof SecurityError) {
        throw error;
      }
      throw new SecurityError(`Invalid URL: ${url} - ${error.message}`);
    }
  }

  /**
   * Get security statistics
   */
  getStats() {
    return {
      ...this.stats,
      config: {
        strictMode: this.config.strictMode,
        maxTensorSize: this.config.maxTensorSize,
        maxMemoryMB: this.config.maxMemoryMB,
        allowedPackages: Array.from(this.config.allowedPackages)
      }
    };
  }

  /**
   * Update security configuration
   */
  updateConfig(newConfig) {
    const oldConfig = { ...this.config };
    this.config = { ...this.config, ...newConfig };
    
    // Update allowed packages set
    if (newConfig.allowedPackages) {
      this.config.allowedPackages = new Set(newConfig.allowedPackages);
    }
    
    this.emit('config:updated', { oldConfig, newConfig: this.config });
  }

  /**
   * Reset security statistics
   */
  resetStats() {
    this.stats = {
      totalValidations: 0,
      blockedOperations: 0,
      warningsIssued: 0,
      lastValidation: null,
      threatCategories: {}
    };
    
    this.emit('stats:reset');
  }

  // Private methods
  _validateCodeInput(code) {
    if (typeof code !== 'string') {
      throw new SecurityError('Code must be a string');
    }
    
    if (code.length === 0) {
      throw new SecurityError('Code cannot be empty');
    }
    
    if (code.length > this.config.maxCodeLength) {
      throw new SecurityError(`Code too long: ${code.length} > ${this.config.maxCodeLength}`);
    }
    
    // Check for null bytes and control characters
    if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(code)) {
      throw new SecurityError('Code contains invalid control characters');
    }
  }

  _analyzeSecurityThreats(code) {
    const threats = [];
    
    // Check dangerous patterns
    for (const [category, patterns] of Object.entries(this.dangerousPatterns)) {
      for (const pattern of patterns) {
        const matches = code.match(pattern.regex);
        if (matches && pattern.severity !== 'none') {
          threats.push({
            category,
            type: 'dangerous',
            pattern: pattern.name,
            severity: pattern.severity,
            matches: matches.length,
            description: pattern.description,
            firstMatch: matches[0]
          });
        }
      }
    }
    
    // Check suspicious patterns
    for (const [category, patterns] of Object.entries(this.suspiciousPatterns)) {
      for (const pattern of patterns) {
        const matches = code.match(pattern.regex);
        if (matches && pattern.severity !== 'none') {
          threats.push({
            category,
            type: 'suspicious',
            pattern: pattern.name,
            severity: pattern.severity,
            matches: matches.length,
            description: pattern.description,
            firstMatch: matches[0]
          });
        }
      }
    }
    
    // Check file system access
    if (!this.config.allowFileSystem) {
      for (const pattern of this.fileSystemPatterns) {
        const matches = code.match(pattern.regex);
        if (matches) {
          threats.push({
            category: 'filesystem',
            type: 'blocked',
            pattern: pattern.name,
            severity: 'high',
            matches: matches.length,
            description: 'File system access not allowed',
            firstMatch: matches[0]
          });
        }
      }
    }
    
    // Check network access
    if (!this.config.allowNetwork) {
      for (const pattern of this.networkPatterns) {
        const matches = code.match(pattern.regex);
        if (matches) {
          threats.push({
            category: 'network',
            type: 'blocked',
            pattern: pattern.name,
            severity: 'high',
            matches: matches.length,
            description: 'Network access not allowed',
            firstMatch: matches[0]
          });
        }
      }
    }
    
    return threats;
  }

  _assessRiskLevel(threats) {
    if (threats.length === 0) {
      return 'low';
    }
    
    const criticalThreats = threats.filter(t => t.severity === 'critical').length;
    const highThreats = threats.filter(t => t.severity === 'high').length;
    const mediumThreats = threats.filter(t => t.severity === 'medium').length;
    
    if (criticalThreats > 0) {
      return 'critical';
    } else if (highThreats > 2) {
      return 'critical';
    } else if (highThreats > 0) {
      return 'high';
    } else if (mediumThreats > 3) {
      return 'high';
    } else if (mediumThreats > 0) {
      return 'medium';
    }
    
    return 'low';
  }

  _enforceSecurityPolicy(code, threats, riskLevel, options = {}) {
    const { allowWarnings = false, bypassValidation = false } = options;
    
    if (bypassValidation && !this.config.strictMode) {
      this.emit('policy:bypassed', { riskLevel, threats: threats.length });
      return {
        allowed: true,
        bypassed: true,
        riskLevel,
        threats,
        warnings: ['Security validation was bypassed']
      };
    }
    
    // Critical operations are blocked, high-risk only in strict mode
    if (riskLevel === 'critical' || (riskLevel === 'high' && this.config.strictMode)) {
      this.stats.blockedOperations++;
      this.emit('policy:blocked', { riskLevel, threats: threats.length });

      // Emit detailed threat information for debugging
      this.emit('threats:detected', threats.map(t => ({
        pattern: t.pattern,
        severity: t.severity,
        description: t.description,
        firstMatch: t.firstMatch
      })));

      throw new SecurityError(`Security policy violation: ${riskLevel} risk detected with ${threats.length} threats`);
    }

    // Emit high-risk operations as warnings when not in strict mode
    if (riskLevel === 'high' && !this.config.strictMode) {
      this.emit('policy:warning', { riskLevel, threats: threats.length, message: 'High-risk operation allowed in non-strict mode' });
    }
    
    // Medium-risk operations may be allowed with warnings
    if (riskLevel === 'medium') {
      if (allowWarnings) {
        this.stats.warningsIssued++;
        this.emit('policy:warning', { riskLevel, threats: threats.length });
        
        return {
          allowed: true,
          riskLevel,
          threats,
          warnings: threats.map(t => `${t.category}: ${t.description}`)
        };
      } else if (this.config.strictMode) {
        this.stats.blockedOperations++;
        throw new SecurityError(`Security policy violation: ${riskLevel} risk detected (strict mode)`);
      }
    }
    
    // Low-risk operations are allowed
    return {
      allowed: true,
      riskLevel,
      threats,
      warnings: riskLevel === 'low' ? [] : threats.map(t => `${t.category}: ${t.description}`)
    };
  }

  _isValidTensorType(tensor) {
    return ArrayBuffer.isView(tensor) || 
           tensor instanceof ArrayBuffer || 
           Array.isArray(tensor) ||
           (tensor && typeof tensor === 'object' && tensor.constructor && tensor.constructor.name.includes('Array'));
  }

  _getTensorElementCount(tensor) {
    if (ArrayBuffer.isView(tensor)) {
      return tensor.length;
    } else if (tensor instanceof ArrayBuffer) {
      return tensor.byteLength / 4; // Assume 32-bit elements
    } else if (Array.isArray(tensor)) {
      return tensor.length;
    }
    return 0;
  }

  _estimateTensorMemoryMB(tensor) {
    const elementCount = this._getTensorElementCount(tensor);
    const bytesPerElement = ArrayBuffer.isView(tensor) ? tensor.BYTES_PER_ELEMENT : 4;
    return (elementCount * bytesPerElement) / (1024 * 1024);
  }

  _validateOperationParams(params) {
    const validated = {};
    
    for (const [key, value] of Object.entries(params || {})) {
      if (typeof key !== 'string' || key.length > 100) {
        throw new SecurityError(`Invalid parameter key: ${key}`);
      }
      
      if (typeof value === 'string' && value.length > this.config.maxStringLength) {
        throw new SecurityError(`Parameter value too long: ${key}`);
      }
      
      validated[key] = value;
    }
    
    return validated;
  }

  _validateOperationOptions(options) {
    const validated = {};
    const allowedOptions = [
      'timeout', 'strategy', 'workgroupSize', 'dataType', 
      'precision', 'optimization', 'caching', 'parallel'
    ];
    
    for (const [key, value] of Object.entries(options || {})) {
      if (!allowedOptions.includes(key)) {
        this.emit('option:unknown', { key, value });
        continue; // Skip unknown options rather than throwing
      }
      
      validated[key] = value;
    }
    
    return validated;
  }

  _updateValidationStats(threats, riskLevel, result) {
    this.stats.lastValidation = {
      timestamp: Date.now(),
      threats: threats.length,
      riskLevel,
      allowed: result.allowed
    };
    
    // Update threat category statistics
    for (const threat of threats) {
      if (!this.stats.threatCategories[threat.category]) {
        this.stats.threatCategories[threat.category] = 0;
      }
      this.stats.threatCategories[threat.category]++;
    }
  }

  _initializeDangerousPatterns() {
    return {
      codeExecution: [
        {
          name: 'eval',
          regex: /(?:^|[^.\w])eval\s*\(/g,
          severity: 'critical',
          description: 'Dynamic code execution with eval()'
        },
        {
          name: 'exec',
          regex: /\bexec\s*\(/g,
          severity: 'critical',
          description: 'Dynamic code execution with exec()'
        },
        {
          name: 'compile',
          regex: /\bcompile\s*\(/g,
          severity: 'high',
          description: 'Code compilation detected'
        }
      ],
      imports: [
        {
          name: 'dynamic_import',
          regex: /\b__import__\s*\(/g,
          severity: 'high',
          description: 'Dynamic import detected'
        },
        {
          name: 'importlib',
          regex: /\bimportlib\./g,
          severity: 'medium',
          description: 'Import library usage'
        }
      ],
      subprocess: [
        {
          name: 'subprocess',
          regex: /\bsubprocess\./g,
          severity: 'critical',
          description: 'Subprocess execution'
        },
        {
          name: 'os_system',
          regex: /\bos\.system\s*\(/g,
          severity: 'critical',
          description: 'OS system command execution'
        },
        {
          name: 'popen',
          regex: /\bos\.popen\s*\(/g,
          severity: 'critical',
          description: 'Process execution with popen'
        }
      ]
    };
  }

  _initializeSuspiciousPatterns() {
    return {
      reflection: [
        {
          name: 'getattr',
          regex: /\bgetattr\s*\(/g,
          severity: 'low', // Reduced from medium - common in ML code
          description: 'Attribute access via getattr'
        },
        {
          name: 'setattr',
          regex: /\bsetattr\s*\(/g,
          severity: 'medium',
          description: 'Attribute modification via setattr'
        },
        {
          name: 'hasattr',
          regex: /\bhasattr\s*\(/g,
          severity: 'none', // Disabled - very common and safe in ML code
          description: 'Attribute existence check'
        }
      ],
      globals: [
        {
          name: 'globals',
          regex: /\bglobals\s*\(\s*\)/g,
          severity: 'medium',
          description: 'Access to global namespace'
        },
        {
          name: 'locals',
          regex: /\blocals\s*\(\s*\)/g,
          severity: 'medium',
          description: 'Access to local namespace'
        },
        {
          name: 'vars',
          regex: /\bvars\s*\(/g,
          severity: 'low',
          description: 'Variable inspection'
        }
      ]
    };
  }

  _initializeFileSystemPatterns() {
    return [
      {
        name: 'open',
        regex: /\bopen\s*\(/g,
        severity: 'high',
        description: 'File opening operation'
      },
      {
        name: 'file',
        regex: /\bfile\s*\(/g,
        severity: 'high',
        description: 'File object creation'
      },
      {
        name: 'pathlib',
        regex: /\bpathlib\./g,
        severity: 'medium',
        description: 'Path manipulation'
      }
    ];
  }

  _initializeNetworkPatterns() {
    return [
      {
        name: 'urllib',
        regex: /\burllib\./g,
        severity: 'high',
        description: 'URL library usage'
      },
      {
        name: 'requests',
        regex: /\brequests\./g,
        severity: 'high',
        description: 'HTTP requests library'
      },
      {
        name: 'socket',
        regex: /\bsocket\./g,
        severity: 'high',
        description: 'Socket networking'
      }
    ];
  }
}

// Custom security error class
class SecurityError extends Error {
  constructor(message, category = 'security', severity = 'high') {
    super(message);
    this.name = 'SecurityError';
    this.category = category;
    this.severity = severity;
    this.timestamp = new Date().toISOString();
  }
}

export default SecurityValidator;
export { SecurityError };