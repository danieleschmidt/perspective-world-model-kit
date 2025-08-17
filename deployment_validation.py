#!/usr/bin/env python3
"""
Deployment Validation Framework
Validates production deployment readiness
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any


class DeploymentValidator:
    """Validates production deployment configuration."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.results = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all deployment validation checks."""
        print("🚀 DEPLOYMENT VALIDATION FRAMEWORK")
        print("=" * 60)
        
        self.validate_docker_config()
        self.validate_monitoring_setup()
        self.validate_security_config()
        self.validate_build_process()
        
        return self.results
    
    def validate_docker_config(self):
        """Validate Docker configuration."""
        print("\n🐳 DOCKER CONFIGURATION")
        print("-" * 40)
        
        # Check Dockerfile
        dockerfile = self.repo_root / "Dockerfile"
        if dockerfile.exists():
            with open(dockerfile, 'r') as f:
                content = f.read()
                
            checks = {
                "multi_stage": "FROM" in content and "AS" in content,
                "security": "USER" in content,
                "optimization": "COPY --from=" in content or "RUN" in content,
                "health_check": "HEALTHCHECK" in content,
                "env_vars": "ENV" in content
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check.replace('_', ' ').title()}: {passed}")
            
            dockerfile_score = sum(checks.values()) / len(checks)
        else:
            print("❌ Dockerfile: Missing")
            dockerfile_score = 0
        
        # Check docker-compose
        compose_files = [
            "docker-compose.yml",
            "docker-compose.prod.yml",
            "docker-compose.dev.yml"
        ]
        
        compose_scores = []
        for compose_file in compose_files:
            path = self.repo_root / compose_file
            if path.exists():
                print(f"✅ {compose_file}: Exists")
                compose_scores.append(1)
            else:
                print(f"❌ {compose_file}: Missing")
                compose_scores.append(0)
        
        self.results["docker"] = {
            "dockerfile_score": dockerfile_score,
            "compose_score": sum(compose_scores) / len(compose_scores),
            "overall": (dockerfile_score + sum(compose_scores) / len(compose_scores)) / 2
        }
    
    def validate_monitoring_setup(self):
        """Validate monitoring and observability."""
        print("\n📊 MONITORING SETUP")
        print("-" * 40)
        
        monitoring_components = {
            "prometheus": ["monitoring/prometheus.yml", "monitoring/prometheus-prod.yml"],
            "grafana": ["monitoring/grafana/"],
            "alerts": ["monitoring/alert_rules.yml"],
            "logging": ["pwmk/utils/logging.py"],
            "health_checks": ["scripts/docker-health-check.sh"]
        }
        
        component_scores = {}
        for component, paths in monitoring_components.items():
            scores = []
            for path_str in paths:
                path = self.repo_root / path_str
                if path.exists():
                    print(f"✅ {component}: {path_str}")
                    scores.append(1)
                else:
                    print(f"❌ {component}: {path_str} missing")
                    scores.append(0)
            component_scores[component] = sum(scores) / len(scores) if scores else 0
        
        self.results["monitoring"] = {
            **component_scores,
            "overall": sum(component_scores.values()) / len(component_scores)
        }
    
    def validate_security_config(self):
        """Validate security configuration."""
        print("\n🔒 SECURITY CONFIGURATION")
        print("-" * 40)
        
        security_files = [
            "security/runtime-monitoring.yml",
            "pwmk/security/input_sanitizer.py",
            "pwmk/security/belief_validator.py",
            "SECURITY.md"
        ]
        
        security_scores = []
        for file_path in security_files:
            path = self.repo_root / file_path
            if path.exists():
                print(f"✅ {file_path}: Exists")
                security_scores.append(1)
            else:
                print(f"❌ {file_path}: Missing")
                security_scores.append(0)
        
        # Check for secrets in common locations
        sensitive_patterns = [".env", "secrets", "keys", "passwords"]
        env_files = list(self.repo_root.glob("**/.*env*"))
        
        if env_files:
            print(f"⚠️ Environment files found: {len(env_files)}")
            secrets_score = 0.5
        else:
            print("✅ No environment files in repo")
            secrets_score = 1.0
        
        self.results["security"] = {
            "config_score": sum(security_scores) / len(security_scores),
            "secrets_score": secrets_score,
            "overall": (sum(security_scores) / len(security_scores) + secrets_score) / 2
        }
    
    def validate_build_process(self):
        """Validate CI/CD and build configuration."""
        print("\n🔧 BUILD PROCESS")
        print("-" * 40)
        
        build_files = [
            "pyproject.toml",
            "Makefile",
            "scripts/build.sh",
            "tox.ini",
            "pytest.ini"
        ]
        
        build_scores = []
        for file_path in build_files:
            path = self.repo_root / file_path
            if path.exists():
                print(f"✅ {file_path}: Exists")
                build_scores.append(1)
            else:
                print(f"❌ {file_path}: Missing")
                build_scores.append(0)
        
        # Check for CI configuration
        ci_paths = [
            ".github/workflows/",
            ".gitlab-ci.yml",
            "Jenkinsfile"
        ]
        
        ci_score = 0
        for ci_path in ci_paths:
            path = self.repo_root / ci_path
            if path.exists():
                print(f"✅ CI/CD: {ci_path}")
                ci_score = 1
                break
        else:
            print("❌ CI/CD: No configuration found")
        
        self.results["build"] = {
            "config_score": sum(build_scores) / len(build_scores),
            "ci_score": ci_score,
            "overall": (sum(build_scores) / len(build_scores) + ci_score) / 2
        }
    
    def generate_summary(self):
        """Generate deployment readiness summary."""
        docker_score = self.results["docker"]["overall"]
        monitoring_score = self.results["monitoring"]["overall"] 
        security_score = self.results["security"]["overall"]
        build_score = self.results["build"]["overall"]
        
        overall_score = (docker_score + monitoring_score + security_score + build_score) / 4
        
        print("\n" + "=" * 60)
        print("📊 DEPLOYMENT READINESS SUMMARY")
        print("=" * 60)
        print(f"🐳 Docker Configuration: {docker_score:.2%}")
        print(f"📊 Monitoring Setup: {monitoring_score:.2%}")
        print(f"🔒 Security Configuration: {security_score:.2%}")
        print(f"🔧 Build Process: {build_score:.2%}")
        print("-" * 60)
        print(f"🎯 OVERALL DEPLOYMENT SCORE: {overall_score:.2%}")
        
        if overall_score >= 0.9:
            status = "🟢 PRODUCTION READY"
            print(f"{status} - System ready for production deployment")
        elif overall_score >= 0.8:
            status = "🟡 MOSTLY READY"
            print(f"{status} - Minor configuration adjustments needed")
        elif overall_score >= 0.7:
            status = "🟠 NEEDS WORK"
            print(f"{status} - Several deployment issues to address")
        else:
            status = "🔴 NOT READY"
            print(f"{status} - Significant deployment work required")
        
        self.results["overall"] = {
            "score": overall_score,
            "status": status
        }
        
        return overall_score


def main():
    """Run deployment validation."""
    validator = DeploymentValidator()
    results = validator.validate_all()
    overall_score = validator.generate_summary()
    
    # Save results
    with open("deployment_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to deployment_validation_results.json")
    return results


if __name__ == "__main__":
    main()