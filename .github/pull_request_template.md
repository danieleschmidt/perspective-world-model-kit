# Pull Request

## Description

Brief description of the changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test improvement
- [ ] Research implementation

## Related Issues

Closes #issue_number
Related to #issue_number

## Changes Made

### Core Functionality
- [ ] Modified world model architecture
- [ ] Updated belief reasoning system
- [ ] Enhanced planning algorithms
- [ ] Added new environment support
- [ ] Improved visualization capabilities

### Technical Implementation
- [ ] Added new dependencies (list them)
- [ ] Modified existing APIs (document breaking changes)
- [ ] Updated configuration options
- [ ] Enhanced error handling
- [ ] Optimized performance critical paths

## Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance benchmarks added/updated
- [ ] End-to-end tests added/updated

### Test Results
```bash
# Include relevant test output
pytest tests/ -v
Coverage: X%
```

### Manual Testing
- [ ] Tested on Python 3.9
- [ ] Tested on Python 3.10
- [ ] Tested on Python 3.11  
- [ ] Tested on Python 3.12
- [ ] Tested with Unity environments (if applicable)
- [ ] Verified backwards compatibility

## Performance Impact

- [ ] No performance impact
- [ ] Performance improvement (quantify if possible)
- [ ] Acceptable performance degradation (explain why)
- [ ] Needs performance optimization (create follow-up issue)

**Benchmark Results** (if applicable):
```
Before: X ms/iteration
After: Y ms/iteration
Improvement: Z%
```

## Documentation

- [ ] Updated docstrings
- [ ] Updated README.md
- [ ] Updated API documentation
- [ ] Added usage examples
- [ ] Updated CHANGELOG.md
- [ ] Added ADR (Architecture Decision Record) if significant

## Security Considerations

- [ ] No security implications
- [ ] Security-reviewed by team
- [ ] Added security tests
- [ ] Updated security documentation
- [ ] Ran security scanners (bandit, safety)

## Checklist

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] Pre-commit hooks pass

### Dependencies
- [ ] New dependencies are justified and documented
- [ ] License compatibility verified for new dependencies
- [ ] Dependencies pinned to appropriate versions
- [ ] Updated requirements documentation

### Research Validation (if applicable)
- [ ] Implementation follows cited research papers
- [ ] Results validate against published benchmarks
- [ ] Methodology is clearly documented
- [ ] Limitations and assumptions are noted

## Deployment Notes

- [ ] No deployment changes required
- [ ] Requires environment variable updates
- [ ] Requires database migrations
- [ ] Requires infrastructure changes
- [ ] Breaking changes require version bump

**Migration Guide** (if breaking changes):
```
# Steps for users to migrate from previous version
```

## Screenshots/Visualizations

If applicable, add screenshots or visualizations to help explain your changes.

## Reviewer Notes

**Specific Areas for Review:**
- Focus on X functionality
- Pay attention to Y performance implications
- Review Z security considerations

**Testing Instructions:**
```bash
# Step-by-step instructions for reviewers to test the changes
git checkout this-branch
pip install -e ".[dev]"
pytest tests/specific_test.py
```

---

**For Maintainers:**
- [ ] PR title follows conventional commit format
- [ ] Appropriate labels applied
- [ ] Milestone assigned (if applicable)
- [ ] Breaking changes documented in CHANGELOG.md