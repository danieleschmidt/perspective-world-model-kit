# Manual Setup Requirements

## Repository Settings

### Branch Protection
- Configure main branch protection rules
- Require pull request reviews
- Require status checks to pass
- Reference: [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)

### GitHub Actions Workflows
- Create CI/CD workflows (see [docs/workflows/README.md](workflows/README.md))
- Configure PyPI publishing secrets
- Set up documentation deployment

### Repository Topics & Description
- Add relevant topics: `python`, `machine-learning`, `theory-of-mind`, `multi-agent`
- Set repository description and homepage URL
- Configure social preview image

### Security Settings
- Enable Dependabot alerts and updates
- Configure CodeQL analysis
- Set up secret scanning

### External Integrations
- Configure monitoring tools (DataDog, New Relic)
- Set up code coverage reporting (Codecov)
- Enable documentation hosting (Read the Docs)

## Required by Administrators

These items require repository admin permissions and cannot be automated.