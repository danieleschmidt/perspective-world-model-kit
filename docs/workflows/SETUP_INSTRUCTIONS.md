# GitHub Workflows Setup Instructions

This document provides step-by-step instructions for setting up GitHub Actions workflows for the PWMK repository.

## Prerequisites

Before setting up workflows, ensure you have:

1. **Repository Admin Access**: You need admin permissions to create workflows and configure secrets
2. **GitHub CLI** (optional but recommended): `gh auth login`
3. **Required Secrets**: Access to configure repository secrets

## Required Repository Secrets

Configure the following secrets in your GitHub repository settings:

### Core Secrets
```bash
# PyPI publishing
PYPI_API_TOKEN=pypi-...
TEST_PYPI_API_TOKEN=pypi-...

# Docker Hub (optional)
DOCKER_HUB_USERNAME=your-username
DOCKER_HUB_TOKEN=dckr_pat_...

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SECURITY_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Email notifications
EMAIL_USERNAME=notifications@pwmk.org
EMAIL_PASSWORD=your-app-password

# Database access for tests
POSTGRES_PASSWORD=secure-password
POSTGRES_GRAFANA_PASSWORD=grafana-password

# External services
HONEYCOMB_API_KEY=your-honeycomb-key
INFLUXDB_TOKEN=your-influxdb-token
GITLEAKS_LICENSE=your-gitleaks-license (if using pro)
```

### Setting Secrets via GitHub CLI
```bash
# Set PyPI token
gh secret set PYPI_API_TOKEN --body "pypi-your-token-here"

# Set Slack webhook
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/..."

# Set multiple secrets from file
echo "PYPI_API_TOKEN=pypi-token" > .env.secrets
echo "SLACK_WEBHOOK_URL=https://hooks.slack.com/..." >> .env.secrets
gh secret set -f .env.secrets
rm .env.secrets  # Clean up
```

## Workflow Setup Steps

### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Files

Copy the example workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Copy main CI workflow
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml

# Copy release workflow
cp docs/workflows/examples/release.yml .github/workflows/release.yml

# Copy security workflow
cp docs/workflows/examples/security.yml .github/workflows/security.yml
```

### Step 3: Create Additional Configuration Files

#### CodeQL Configuration
Create `.github/codeql/codeql-config.yml`:

```yaml
name: "PWMK CodeQL Config"

disable-default-queries: false

queries:
  - uses: security-extended
  - uses: security-and-quality

paths-ignore:
  - "docs/**"
  - "tests/fixtures/**"
  - "**/*.md"

paths:
  - "pwmk/**"
  - "scripts/**"
```

#### GitLeaks Configuration
Create `.gitleaks.toml`:

```toml
title = "PWMK GitLeaks Config"

[extend]
useDefault = true

[[rules]]
description = "PWMK API Keys"
id = "pwmk-api-key"
regex = '''(?i)pwmk[_-]?api[_-]?key[_-\s]*[:=][_-\s]*['"]?([a-zA-Z0-9]{32,})['"]?'''
tags = ["key", "API", "PWMK"]

[[rules]]
description = "Database URLs"
id = "database-url"
regex = '''(?i)(postgres|mysql|mongodb)://[^:\s]+:[^@\s]+@[^:\s]+:[0-9]+/[^\s]+'''
tags = ["database", "url"]

[allowlist]
description = "Allowlisted files"
files = [
    '''(.*?)(jpg|gif|doc|pdf|bin)$''',
    '''.gitleaks.toml'''
]

paths = [
    '''tests/fixtures/'''
]
```

### Step 4: Configure Branch Protection

Set up branch protection rules for the main branches:

```bash
# Via GitHub CLI
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Code Quality","Tests (Python 3.11, ubuntu-latest)","Security Scanning","Build Package"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null
```

### Step 5: Create GitHub Environments

Create deployment environments for staging and production:

```bash
# Create staging environment
gh api repos/:owner/:repo/environments/staging --method PUT

# Create production environment with protection rules
gh api repos/:owner/:repo/environments/production --method PUT \
  --field protection_rules='[{"type":"required_reviewers","reviewers":[{"type":"User","id":123}]}]'
```

### Step 6: Configure Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 10
    reviewers:
      - "team-leads"
    assignees:
      - "security-team"
    commit-message:
      prefix: "deps"
      prefix-development: "deps-dev"
      include: "scope"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

## Workflow Customization

### Customizing Test Matrix

Edit the test matrix in `ci.yml` to match your requirements:

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.9', '3.10', '3.11']
    # Add custom configurations
    include:
      - os: ubuntu-latest
        python-version: '3.12'
        experimental: true
```

### Adding Custom Notifications

To add Microsoft Teams notifications:

```yaml
- name: Notify Teams
  if: failure()
  uses: aliencube/microsoft-teams-actions@v0.8.0
  with:
    webhook_uri: ${{ secrets.MSTEAMS_WEBHOOK }}
    title: "PWMK Build Failed"
    text: "Build failed for commit ${{ github.sha }}"
```

### Configuring Self-Hosted Runners

For GPU tests, configure self-hosted runners:

1. Set up runner with GPU support
2. Add labels: `[self-hosted, gpu, linux]`
3. Update workflow to use the runner:

```yaml
gpu-tests:
  runs-on: [self-hosted, gpu]
```

## Monitoring and Maintenance

### Workflow Health Checks

Create a script to monitor workflow health:

```bash
#!/bin/bash
# check-workflow-health.sh

echo "Checking workflow runs..."
gh run list --limit 50 --json status,conclusion,workflowName

echo "Recent failures:"
gh run list --limit 10 --json status,conclusion,workflowName \
  | jq '.[] | select(.conclusion == "failure")'
```

### Performance Optimization

1. **Use caching**: Ensure all workflows use appropriate caching
2. **Parallel execution**: Run independent jobs in parallel
3. **Conditional execution**: Use path filters and conditions
4. **Artifact management**: Clean up old artifacts regularly

### Security Best Practices

1. **Pin action versions**: Use specific versions, not `@main`
2. **Minimal permissions**: Use least-privilege principle
3. **Secret scanning**: Enable GitHub secret scanning
4. **Review dependencies**: Regularly audit workflow dependencies

## Troubleshooting Common Issues

### Permission Errors

```bash
# Check repository permissions
gh api repos/:owner/:repo --jq '.permissions'

# Check if secrets are set
gh secret list
```

### Build Failures

1. Check if all required secrets are configured
2. Verify Python version compatibility
3. Check for missing system dependencies
4. Review test failures in detail

### Deployment Issues

1. Verify environment configuration
2. Check deployment secrets
3. Review target environment status
4. Validate deployment scripts

## Getting Help

If you encounter issues:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow run logs for detailed error messages
3. Check repository settings and permissions
4. Consult the troubleshooting section in this document
5. Create an issue in the repository for persistent problems

## Workflow Examples

Additional workflow examples are available in the `docs/workflows/examples/` directory:

- `ci.yml` - Comprehensive CI/CD pipeline
- `release.yml` - Automated release management
- `security.yml` - Security scanning and compliance
- `docs.yml` - Documentation build and deployment
- `dependency-update.yml` - Automated dependency updates

Each workflow is designed to be modular and can be customized based on your specific requirements.