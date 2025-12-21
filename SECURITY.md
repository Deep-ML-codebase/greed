# Security Policy

## Supported Versions

We actively support the following versions of Greed.js with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 3.1.x   | :white_check_mark: |
| 3.0.x   | :white_check_mark: |
| < 3.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in Greed.js, please follow these steps:

### 1. Do Not Disclose Publicly

Please do not create a public GitHub issue for security vulnerabilities. This could put users at risk.

### 2. Contact Us Privately

Send details of the vulnerability to: **khalkaraditya8@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)
- Your contact information

### 3. What to Expect

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium: Within 60 days
  - Low: Next scheduled release

### 4. Disclosure Process

1. We will investigate and validate the report
2. We will develop and test a fix
3. We will release a patched version
4. We will publish a security advisory
5. After the fix is released, we will credit you (unless you prefer to remain anonymous)

## Security Best Practices

When using Greed.js:

### Input Validation

- Always validate user input before passing to Python execution
- Be cautious with dynamic code execution
- Sanitize data from untrusted sources

```javascript
// Good: Validate input
const userInput = sanitize(rawInput);
await greed.runPython(`result = ${userInput}`);

// Bad: Direct execution of user input
await greed.runPython(untrustedCode); // Never do this!
```

### Browser Security

- Use Content Security Policy (CSP) headers
- Enable HTTPS for production deployments
- Restrict WebGPU access to trusted origins

### Dependency Security

- Keep Greed.js updated to the latest version
- Regularly audit your dependencies with `npm audit`
- Monitor security advisories

### Data Privacy

- GPU memory may persist between operations
- Clear sensitive data explicitly
- Use `greed.clearState()` after processing sensitive information

```javascript
// Process sensitive data
await greed.runPython('process_sensitive_data()');

// Clean up
await greed.clearState();
```

## Known Security Considerations

### WebGPU Timing Attacks

WebGPU timing information could potentially be used for side-channel attacks. Be aware when processing sensitive data in shared environments.

**Mitigation**: Process sensitive operations server-side when possible.

### Python Code Execution

Greed.js executes Python code in the browser. Never execute untrusted code from external sources.

**Mitigation**: Validate and sanitize all code before execution.

### Memory Management

GPU buffers may retain data after operations complete.

**Mitigation**: Use `clearState()` to clean up sensitive data.

### Browser Extension Interference

Some browser extensions may interfere with WebGPU operations or inject code.

**Mitigation**: Test in clean browser profiles, advise users about potential conflicts.

## Security Updates

Security updates are released as patch versions (e.g., 3.1.1, 3.1.2) and published through:

- GitHub Security Advisories
- NPM package updates
- Release notes on GitHub

Subscribe to releases to stay informed: https://github.com/adityakhalkar/greed/releases

## Bug Bounty Program

We currently do not have a formal bug bounty program, but we deeply appreciate security researchers who responsibly disclose vulnerabilities. Contributors who report valid security issues will be:

- Credited in the security advisory (with permission)
- Mentioned in release notes
- Listed as security contributors

## Third-Party Dependencies

Greed.js relies on:

- Pyodide (Python runtime)
- WebGPU (browser API)
- Various NPM packages

We monitor these dependencies for security issues and update them promptly when vulnerabilities are discovered.

## Security Audit History

| Date | Auditor | Scope | Findings |
|------|---------|-------|----------|
| TBD  | TBD     | TBD   | TBD      |

## Contact

For security concerns: khalkaraditya8@gmail.com

For general questions: Open an issue on GitHub

---

Thank you for helping keep Greed.js and its users safe!
