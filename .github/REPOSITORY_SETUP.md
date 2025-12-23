# GitHub Repository Setup Guide

This guide will help you configure your GitHub repository for Greed.js with all the necessary settings for public contributions.

## Initial Setup

### 1. Repository Settings

Go to **Settings** → **General**:

- [ ] Set description: "High-performance PyTorch execution in browsers with WebGPU acceleration"
- [ ] Add website: Your demo URL
- [ ] Add topics: `pytorch`, `webgpu`, `machine-learning`, `browser`, `gpu-acceleration`, `python`, `javascript`, `wasm`, `deep-learning`
- [ ] Enable **Issues**
- [ ] Enable **Discussions** (recommended for Q&A)
- [ ] Disable **Wiki** (use docs folder instead)
- [ ] Disable **Projects** (unless you want to use them)

### 2. Branch Protection

Go to **Settings** → **Branches** → **Add branch protection rule**:

**For `main` branch:**
- Branch name pattern: `main`
- [ ] Require pull request reviews before merging
  - Required approving reviews: 1
  - Dismiss stale pull request approvals when new commits are pushed
- [ ] Require status checks to pass before merging
  - Required checks: `build`, `test-browser`, `security`
- [ ] Require branches to be up to date before merging
- [ ] Require conversation resolution before merging
- [ ] Do not allow bypassing the above settings

**For `v3` branch (if using):**
- Same settings as main

### 3. GitHub Actions

Go to **Settings** → **Actions** → **General**:

- [ ] Allow all actions and reusable workflows
- [ ] Workflow permissions: Read and write permissions
- [ ] Allow GitHub Actions to create and approve pull requests

### 4. Secrets Configuration

Go to **Settings** → **Secrets and variables** → **Actions**:

Add the following secrets:

- [ ] `NPM_TOKEN`: Your NPM automation token
  - Get from: https://www.npmjs.com/settings/YOUR_USERNAME/tokens
  - Type: Automation token

To create NPM token:
```bash
npm login
# Go to npmjs.com → Settings → Access Tokens → Generate New Token
# Choose "Automation" type
# Copy token and add to GitHub secrets
```

### 5. GitHub Pages (Optional)

If you want to host documentation or demos:

Go to **Settings** → **Pages**:

- Source: Deploy from a branch
- Branch: `main` (or `gh-pages`)
- Folder: `/docs` or `/` (root)

### 6. Discussions Setup

Go to **Settings** → **Features**:

- [ ] Enable **Discussions**

Then go to **Discussions** tab:

Create categories:
- **Announcements** (announcement type)
- **General** (open discussion)
- **Q&A** (Q&A type)
- **Show and Tell** (open discussion)
- **Ideas** (open discussion)

### 7. Issue Labels

Go to **Issues** → **Labels**:

Add these custom labels (beyond defaults):

**Type:**
- `webgpu` - WebGPU related issues (color: #0366d6)
- `pytorch` - PyTorch compatibility (color: #ee4c2c)
- `shader` - WebGPU shader development (color: #1d76db)
- `performance` - Performance improvements (color: #fbca04)
- `browser-compatibility` - Browser support issues (color: #0e8a16)
- `api` - API design/changes (color: #d4c5f9)

**Priority:**
- `priority: critical` - Critical issues (color: #b60205)
- `priority: high` - High priority (color: #d93f0b)
- `priority: medium` - Medium priority (color: #fbca04)
- `priority: low` - Low priority (color: #0e8a16)

**Status:**
- `status: needs-reproduction` - Needs repro steps (color: #e99695)
- `status: needs-investigation` - Needs research (color: #f9d0c4)
- `status: blocked` - Blocked by dependency (color: #d93f0b)

**Community:**
- `good first issue` - Good for newcomers (color: #7057ff)
- `help wanted` - Extra attention needed (color: #008672)
- `question` - Questions (color: #d876e3)

### 8. Repository Topics

Add these topics to your repository:

```
pytorch webgpu machine-learning deep-learning browser gpu-acceleration
javascript python wasm pyodide neural-networks tensor ml web-ml
browser-ml client-side-ml webassembly gpu-compute
```

### 9. Social Preview

Go to **Settings** → **Social preview**:

- [ ] Upload an image (1280×640px recommended)
- Use the Greed.js logo or a screenshot of the demo

### 10. Code and Automation

Go to **Settings** → **Code and automation**:

**Automatically delete head branches:**
- [ ] Enable (cleans up merged PR branches)

**Pull Requests:**
- [ ] Allow merge commits
- [ ] Allow squash merging (recommended as default)
- [ ] Allow rebase merging
- [ ] Always suggest updating pull request branches
- [ ] Automatically delete head branches

### 11. Security

Go to **Settings** → **Security**:

**Code security and analysis:**
- [ ] Enable Dependency graph
- [ ] Enable Dependabot alerts
- [ ] Enable Dependabot security updates
- [ ] Enable Secret scanning
- [ ] Enable Push protection

**Security policy:**
- Already created in SECURITY.md ✅

### 12. Insights

Go to **Insights** → **Community**:

Verify checklist is complete:
- [x] Description
- [x] README
- [x] Code of conduct
- [x] Contributing guidelines
- [x] License
- [x] Security policy
- [x] Issue templates
- [x] Pull request template

## Post-Setup Tasks

### 1. Test GitHub Actions

```bash
# Push a commit to trigger CI
git add .
git commit -m "test: verify GitHub Actions setup"
git push
```

Go to **Actions** tab and verify workflows run successfully.

### 2. Create First Release

```bash
# Tag version
git tag -a v3.1.0 -m "Release v3.1.0: GPU acceleration and professional setup"
git push origin v3.1.0
```

Go to **Releases** → **Draft a new release**:
- Tag: v3.1.0
- Title: v3.1.0 - GPU Acceleration
- Description: Add release notes
- [ ] Publish release

This will trigger the NPM publish workflow.

### 3. Pin Important Issues/Discussions

- Pin the roadmap issue
- Pin getting started discussion
- Pin any important announcements

### 4. Configure Repository Notifications

For maintainers:
- Go to **Watch** → **Custom**
- Select: Issues, Pull requests, Releases, Discussions

### 5. Set Up Repository Insights

Go to **Insights** → **Traffic**:
- Monitor clones, visitors, popular content
- Use this data to improve documentation

## Ongoing Maintenance

### Weekly Tasks
- [ ] Review and respond to new issues
- [ ] Review open pull requests
- [ ] Check GitHub Actions for failures
- [ ] Monitor Dependabot alerts

### Monthly Tasks
- [ ] Review and close stale issues
- [ ] Update dependencies
- [ ] Check security advisories
- [ ] Update documentation

### Per-Release Tasks
- [ ] Update version in package.json
- [ ] Update CHANGELOG.md
- [ ] Create GitHub release
- [ ] Verify NPM publish succeeded
- [ ] Announce on social media
- [ ] Update documentation

## Troubleshooting

### GitHub Actions Failing

1. Check workflow logs in **Actions** tab
2. Verify secrets are configured correctly
3. Ensure branch protection rules don't block Actions
4. Check Node.js version compatibility

### NPM Publish Failing

1. Verify NPM_TOKEN secret is valid
2. Check package.json version is incremented
3. Ensure you're logged into NPM: `npm whoami`
4. Verify 2FA is set up correctly

### Dependabot Issues

1. Review Dependabot alerts in **Security** tab
2. Create PR for security updates
3. Test thoroughly before merging
4. Update lock files if needed

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [NPM Publishing Guide](https://docs.npmjs.com/cli/v9/commands/npm-publish)
- [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [Semantic Versioning](https://semver.org/)

---

Your repository is now configured for open-source contributions! 🎉
